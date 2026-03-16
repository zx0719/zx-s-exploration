from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.common import resolve_optional_path
from src.utils.remoteclip import load_remoteclip_classes


def resolve_torch_dtype(value: str | None):
    """Map config values to torch dtypes."""
    if value in (None, "auto"):
        return value
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported torch dtype: {value}")
    return mapping[value]


class RemoteQwenBase(nn.Module):
    """Shared Remote-Clip + Qwen multimodal backbone."""

    def __init__(
        self,
        qwen_path: str,
        remoteclip_ckpt_path: str,
        remoteclip_repo_path: str | None = None,
        image_token: str = "<image>",
        trust_remote_code: bool = False,
        torch_dtype: str | None = "auto",
        remoteclip_mm_hidden_size: int = 1024,
        projector_weights_path: str | None = None,
    ):
        super().__init__()
        self.image_token = image_token

        self.tokenizer = AutoTokenizer.from_pretrained(
            qwen_path,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})

        model_dtype = resolve_torch_dtype(torch_dtype)
        llm_kwargs = {"trust_remote_code": trust_remote_code}
        if model_dtype == "auto":
            llm_kwargs["torch_dtype"] = "auto"
        elif model_dtype is not None:
            llm_kwargs["torch_dtype"] = model_dtype

        self.llm = AutoModelForCausalLM.from_pretrained(qwen_path, **llm_kwargs)
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)

        RemoteVisionTower, TowerConfig = load_remoteclip_classes(remoteclip_repo_path)
        llm_hidden_size = int(self.llm.config.hidden_size)
        tower_config = TowerConfig(hidden_size=llm_hidden_size, mm_hidden_size=remoteclip_mm_hidden_size)
        self.vision_tower = RemoteVisionTower(tower_config, model_path=remoteclip_ckpt_path)

        projector_path = resolve_optional_path(projector_weights_path)
        if projector_path:
            state_dict = torch.load(projector_path, map_location="cpu")
            self.vision_tower.projector.load_state_dict(state_dict, strict=True)
            print(f"[RemoteQwenBase] loaded projector from {projector_path}")

    @property
    def preprocess(self):
        """Expose the vision tower preprocessing pipeline."""
        return self.vision_tower.preprocess

    def freeze_llm(self) -> None:
        """Freeze all LLM parameters."""
        for parameter in self.llm.parameters():
            parameter.requires_grad = False

    def freeze_vision_backbone(self) -> None:
        """Freeze all vision parameters."""
        for parameter in self.vision_tower.parameters():
            parameter.requires_grad = False

    def enable_projector_training(self) -> None:
        """Unfreeze the projector after freezing the full vision tower."""
        for parameter in self.vision_tower.projector.parameters():
            parameter.requires_grad = True

    def build_multimodal_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vision_tokens: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Replace the first image token with projected vision tokens."""
        batch_size = input_ids.shape[0]
        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        vision_tokens = vision_tokens.to(dtype=text_embeddings.dtype, device=text_embeddings.device)

        merged_embeddings = []
        merged_attention = []
        merged_labels = []
        sequence_lengths = []

        for batch_index in range(batch_size):
            ids = input_ids[batch_index]
            attn = attention_mask[batch_index]
            lbl = labels[batch_index] if labels is not None else None

            positions = (ids == self.image_token_id).nonzero(as_tuple=False)
            if positions.numel() == 0:
                embedding = torch.cat([vision_tokens[batch_index], text_embeddings[batch_index]], dim=0)
                attention = torch.cat(
                    [
                        torch.ones(vision_tokens.shape[1], device=attn.device, dtype=attn.dtype),
                        attn,
                    ],
                    dim=0,
                )
                if lbl is not None:
                    vision_labels = torch.full(
                        (vision_tokens.shape[1],),
                        -100,
                        device=lbl.device,
                        dtype=lbl.dtype,
                    )
                    label_tensor = torch.cat([vision_labels, lbl], dim=0)
            else:
                image_position = int(positions[0].item())
                embedding = torch.cat(
                    [
                        text_embeddings[batch_index, :image_position, :],
                        vision_tokens[batch_index],
                        text_embeddings[batch_index, image_position + 1 :, :],
                    ],
                    dim=0,
                )
                attention = torch.cat(
                    [
                        attn[:image_position],
                        torch.ones(vision_tokens.shape[1], device=attn.device, dtype=attn.dtype),
                        attn[image_position + 1 :],
                    ],
                    dim=0,
                )
                if lbl is not None:
                    label_tensor = torch.cat(
                        [
                            lbl[:image_position],
                            torch.full(
                                (vision_tokens.shape[1],),
                                -100,
                                device=lbl.device,
                                dtype=lbl.dtype,
                            ),
                            lbl[image_position + 1 :],
                        ],
                        dim=0,
                    )

            merged_embeddings.append(embedding)
            merged_attention.append(attention)
            sequence_lengths.append(int(attention.sum().item()))
            if lbl is not None:
                merged_labels.append(label_tensor)

        max_length = max(item.shape[0] for item in merged_embeddings)
        padded_embeddings = []
        padded_attention = []
        padded_labels = []

        for batch_index in range(batch_size):
            embedding = merged_embeddings[batch_index]
            attention = merged_attention[batch_index]
            pad_length = max_length - embedding.shape[0]

            if pad_length > 0:
                embedding = torch.cat(
                    [
                        embedding,
                        torch.zeros(
                            pad_length,
                            embedding.shape[1],
                            device=embedding.device,
                            dtype=embedding.dtype,
                        ),
                    ],
                    dim=0,
                )
                attention = torch.cat(
                    [
                        attention,
                        torch.zeros(pad_length, device=attention.device, dtype=attention.dtype),
                    ],
                    dim=0,
                )

            padded_embeddings.append(embedding)
            padded_attention.append(attention)

            if labels is not None:
                label_tensor = merged_labels[batch_index]
                if pad_length > 0:
                    label_tensor = torch.cat(
                        [
                            label_tensor,
                            torch.full(
                                (pad_length,),
                                -100,
                                device=label_tensor.device,
                                dtype=label_tensor.dtype,
                            ),
                        ],
                        dim=0,
                    )
                padded_labels.append(label_tensor)

        output = {
            "inputs_embeds": torch.stack(padded_embeddings, dim=0),
            "attention_mask": torch.stack(padded_attention, dim=0),
            "sequence_lengths": sequence_lengths,
        }
        if labels is not None:
            output["labels"] = torch.stack(padded_labels, dim=0)
        return output

    @torch.no_grad()
    def generate_from_prompt(
        self,
        prompt_text: str,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.7,
    ) -> str:
        """Generate text for one image and one prompt."""
        device = next(self.parameters()).device
        self.eval()

        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        vision_tokens = self.vision_tower(pixel_values.to(device))
        merged = self.build_multimodal_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_tokens=vision_tokens,
        )
        prompt_length = merged["sequence_lengths"][0]

        generated_ids = self.llm.generate(
            inputs_embeds=merged["inputs_embeds"],
            attention_mask=merged["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if generated_ids.shape[1] > prompt_length:
            answer_ids = generated_ids[0, prompt_length:]
        else:
            answer_ids = generated_ids[0]
        return self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    def save_projector(self, output_path: str) -> None:
        """Save the projector state dict."""
        torch.save(self.vision_tower.projector.state_dict(), output_path)


class Stage1AlignModel(RemoteQwenBase):
    """Stage1 model that trains the projector with LM loss and alignment loss."""

    def __init__(
        self,
        qwen_path: str,
        remoteclip_ckpt_path: str,
        remoteclip_repo_path: str | None = None,
        image_token: str = "<image>",
        trust_remote_code: bool = False,
        torch_dtype: str | None = "auto",
        remoteclip_mm_hidden_size: int = 1024,
        projector_weights_path: str | None = None,
        align_weight: float = 1.0,
        align_temp: float = 0.07,
    ):
        super().__init__(
            qwen_path=qwen_path,
            remoteclip_ckpt_path=remoteclip_ckpt_path,
            remoteclip_repo_path=remoteclip_repo_path,
            image_token=image_token,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            remoteclip_mm_hidden_size=remoteclip_mm_hidden_size,
            projector_weights_path=projector_weights_path,
        )
        self.align_weight = align_weight
        self.align_temp = align_temp
        self.align_loss_fn = nn.CrossEntropyLoss()
        self.freeze_llm()
        self.freeze_vision_backbone()
        self.enable_projector_training()
        self._last_losses = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pixel_values: torch.Tensor,
        **_: Any,
    ):
        vision_tokens = self.vision_tower(pixel_values)
        text_embeddings = self.llm.get_input_embeddings()(input_ids)

        caption_mask = labels.ne(-100).unsqueeze(-1).to(dtype=text_embeddings.dtype)
        caption_sum = (text_embeddings * caption_mask).sum(dim=1)
        caption_count = labels.ne(-100).sum(dim=1).clamp(min=1).unsqueeze(-1).to(dtype=text_embeddings.dtype)
        caption_centroid = caption_sum / caption_count
        vision_centroid = vision_tokens.mean(dim=1)

        caption_norm = caption_centroid / caption_centroid.norm(dim=1, keepdim=True).clamp(min=1e-6)
        vision_norm = vision_centroid / vision_centroid.norm(dim=1, keepdim=True).clamp(min=1e-6)
        similarity = torch.matmul(vision_norm, caption_norm.t()) / float(self.align_temp)
        targets = torch.arange(similarity.size(0), device=similarity.device, dtype=torch.long)
        align_loss = 0.5 * (
            self.align_loss_fn(similarity, targets) + self.align_loss_fn(similarity.t(), targets)
        )

        merged = self.build_multimodal_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            vision_tokens=vision_tokens,
        )
        outputs = self.llm(
            inputs_embeds=merged["inputs_embeds"],
            attention_mask=merged["attention_mask"],
            labels=merged["labels"],
        )
        lm_loss = outputs.loss if outputs.loss is not None else None
        total_loss = align_loss * self.align_weight if lm_loss is None else lm_loss + self.align_weight * align_loss

        self._last_losses = {
            "lm_loss": float(lm_loss.detach().cpu()) if lm_loss is not None else None,
            "align_loss": float(align_loss.detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
        }
        outputs.loss = total_loss
        return outputs


class Stage2ConnectorLoRAModel(RemoteQwenBase):
    """Stage2 model that trains the projector and optional Qwen LoRA adapters."""

    def __init__(
        self,
        qwen_path: str,
        remoteclip_ckpt_path: str,
        remoteclip_repo_path: str | None = None,
        image_token: str = "<image>",
        trust_remote_code: bool = False,
        torch_dtype: str | None = "auto",
        remoteclip_mm_hidden_size: int = 1024,
        projector_weights_path: str | None = None,
        enable_lora: bool = True,
        train_lora: bool = True,
        lora_weights_path: str | None = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        print_trainable_parameters: bool = True,
    ):
        super().__init__(
            qwen_path=qwen_path,
            remoteclip_ckpt_path=remoteclip_ckpt_path,
            remoteclip_repo_path=remoteclip_repo_path,
            image_token=image_token,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            remoteclip_mm_hidden_size=remoteclip_mm_hidden_size,
            projector_weights_path=projector_weights_path,
        )
        self.freeze_vision_backbone()
        self.enable_projector_training()
        self.freeze_llm()

        self.enable_lora = enable_lora
        self._last_losses = None
        self._last_sample = None

        if enable_lora:
            lora_path = resolve_optional_path(lora_weights_path)
            if lora_path:
                self.llm = PeftModel.from_pretrained(self.llm, lora_path, is_trainable=train_lora)
                print(f"[Stage2ConnectorLoRAModel] loaded LoRA from {lora_path}")
            else:
                if lora_target_modules is None:
                    lora_target_modules = [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ]
                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=lora_target_modules,
                )
                self.llm = get_peft_model(self.llm, lora_config)

            if print_trainable_parameters and hasattr(self.llm, "print_trainable_parameters"):
                self.llm.print_trainable_parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pixel_values: torch.Tensor,
        **_: Any,
    ):
        vision_tokens = self.vision_tower(pixel_values)
        merged = self.build_multimodal_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            vision_tokens=vision_tokens,
        )
        outputs = self.llm(
            inputs_embeds=merged["inputs_embeds"],
            attention_mask=merged["attention_mask"],
            labels=merged["labels"],
        )
        self._last_losses = {"lm_loss": float(outputs.loss.detach().cpu())}
        return outputs

    @torch.no_grad()
    def generate_from_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 16,
    ) -> str:
        """Generate from a batch item using the multimodal prompt."""
        device = next(self.parameters()).device
        dummy_labels = torch.full_like(input_ids, -100, device=input_ids.device)
        vision_tokens = self.vision_tower(pixel_values.to(device))
        merged = self.build_multimodal_inputs(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            labels=dummy_labels.to(device),
            vision_tokens=vision_tokens,
        )
        prompt_length = merged["sequence_lengths"][0]
        generated_ids = self.llm.generate(
            inputs_embeds=merged["inputs_embeds"][:1],
            attention_mask=merged["attention_mask"][:1],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if generated_ids.shape[1] > prompt_length:
            answer_ids = generated_ids[0, prompt_length:]
        else:
            answer_ids = generated_ids[0]
        return self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    def save_lora(self, output_dir: str) -> None:
        """Save LoRA adapter weights when PEFT is enabled."""
        if self.enable_lora and hasattr(self.llm, "save_pretrained"):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.llm.save_pretrained(output_dir)
