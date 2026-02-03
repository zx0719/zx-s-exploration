# train_stage2_osvqa.py
import os
import re
import sys
import json
from collections import defaultdict

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model
import swanlab

from osvqa_dataset import OSVQAStage2RGBDataset, Stage2Collator

# RemoteClip
sys.path.insert(0, "/home/xingyueao/RemoteClip")
from RemoteVisionTower import RemoteVisionTower, TowerConfig

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# -------------------------
# Utils
# -------------------------
def normalize_answer(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[\,\.\?\!\:\;\"\'\(\)\[\]\{\}]", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    yes_set = {"yes", "y", "yeah", "yep", "true", "correct"}
    no_set = {"no", "n", "nope", "false", "incorrect"}
    if s in yes_set:
        return "yes"
    if s in no_set:
        return "no"
    return s


def count_trainable(model: nn.Module):
    n_train, n_all = 0, 0
    for p in model.parameters():
        n_all += p.numel()
        if p.requires_grad:
            n_train += p.numel()
    return n_train, n_all, (n_train / n_all if n_all > 0 else 0.0)


# -------------------------
# Model: train connector + LoRA
# -------------------------
class Stage2ConnectorLoRAModel(nn.Module):
    """
    Stage2: train vision_tower.projector (connector) + Qwen LoRA
    """
    def __init__(
        self,
        qwen_path: str,
        remoteclip_ckpt_path: str,
        image_token: str = "<image>",
        trust_remote_code: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules=None,
    ):
        super().__init__()
        self.image_token = image_token

        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path, use_fast=True, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})

        self.llm = AutoModelForCausalLM.from_pretrained(
            qwen_path,
            torch_dtype="auto",
            trust_remote_code=trust_remote_code,
        )
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)

        llm_hidden = int(self.llm.config.hidden_size)
        print(f"[Init] Qwen hidden_size={llm_hidden}")

        tower_cfg = TowerConfig(hidden_size=llm_hidden, mm_hidden_size=1024)
        self.vision_tower = RemoteVisionTower(tower_cfg, model_path=remoteclip_ckpt_path)

        # freeze vision backbone; train projector only
        for p in self.vision_tower.parameters():
            p.requires_grad = False
        for p in self.vision_tower.projector.parameters():
            p.requires_grad = True

        # freeze LLM base weights
        for p in self.llm.parameters():
            p.requires_grad = False

        # LoRA
        if lora_target_modules is None:
            # 如果你觉得 LoRA 太大，改成只 attention：["q_proj","k_proj","v_proj","o_proj"]
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules,
        )
        self.llm = get_peft_model(self.llm, lora_cfg)
        self.llm.print_trainable_parameters()

        # caches for logging
        self._last_losses = None
        self._last_sample = None

    def _merge(self, input_ids, attention_mask, labels, vision_tokens):
        """
        Replace first <image> token with vision_tokens, and set vision labels=-100
        """
        B, T = input_ids.shape
        Bv, Tv, _ = vision_tokens.shape
        assert B == Bv

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        vision_tokens = vision_tokens.to(dtype=text_embeds.dtype, device=text_embeds.device)

        new_embeds, new_attn, new_labels = [], [], []
        for b in range(B):
            ids = input_ids[b]
            attn = attention_mask[b]
            lbl = labels[b]

            pos = (ids == self.image_token_id).nonzero(as_tuple=False)
            if pos.numel() == 0:
                merged_e = torch.cat([vision_tokens[b], text_embeds[b]], dim=0)
                merged_a = torch.cat([torch.ones(Tv, device=attn.device, dtype=attn.dtype), attn], dim=0)
                merged_l = torch.cat([torch.full((Tv,), -100, device=lbl.device, dtype=lbl.dtype), lbl], dim=0)
            else:
                p0 = int(pos[0].item())
                merged_e = torch.cat([text_embeds[b, :p0, :], vision_tokens[b], text_embeds[b, p0 + 1 :, :]], dim=0)
                merged_a = torch.cat([attn[:p0], torch.ones(Tv, device=attn.device, dtype=attn.dtype), attn[p0 + 1 :]], dim=0)
                merged_l = torch.cat([lbl[:p0], torch.full((Tv,), -100, device=lbl.device, dtype=lbl.dtype), lbl[p0 + 1 :]], dim=0)

            new_embeds.append(merged_e)
            new_attn.append(merged_a)
            new_labels.append(merged_l)

        max_len = max(x.size(0) for x in new_embeds)
        pe, pa, pl = [], [], []
        for b in range(B):
            e, a, l = new_embeds[b], new_attn[b], new_labels[b]
            pad = max_len - e.size(0)
            if pad > 0:
                e = torch.cat([e, torch.zeros(pad, e.size(1), device=e.device, dtype=e.dtype)], dim=0)
                a = torch.cat([a, torch.zeros(pad, device=a.device, dtype=a.dtype)], dim=0)
                l = torch.cat([l, torch.full((pad,), -100, device=l.device, dtype=l.dtype)], dim=0)
            pe.append(e); pa.append(a); pl.append(l)

        return torch.stack(pe, 0), torch.stack(pa, 0), torch.stack(pl, 0)

    def forward(self, input_ids, attention_mask, labels, pixel_values, **kwargs):
        vision_tokens = self.vision_tower(pixel_values)
        inputs_embeds, attn, new_labels = self._merge(input_ids, attention_mask, labels, vision_tokens)
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=new_labels)

        # cache losses for logging
        try:
            self._last_losses = {"lm_loss": float(outputs.loss.detach().cpu())}
        except Exception:
            self._last_losses = None

        return outputs

    @torch.no_grad()
    def generate_from_batch(
        self,
        input_ids, attention_mask, pixel_values,
        max_new_tokens: int = 16,
    ):
        """
        用 batch[0] 做一次生成（把 <image> 换成 vision tokens），返回完整输出文本。
        """
        device = next(self.parameters()).device
        self.eval()

        vision_tokens = self.vision_tower(pixel_values.to(device))
        dummy_labels = torch.full_like(input_ids, -100).to(device)

        inputs_embeds, attn, _ = self._merge(
            input_ids.to(device),
            attention_mask.to(device),
            dummy_labels,
            vision_tokens,
        )

        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds[:1],
            attention_mask=attn[:1],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()


# -------------------------
# SwanLab callback
# -------------------------
class SwanLabCallback(TrainerCallback):
    def __init__(self, project: str, exp_name: str, config: dict, log_sample: bool = True):
        super().__init__()
        self.project = project
        self.exp_name = exp_name
        self.config = config
        self.log_sample = log_sample
        self.run = None
        self.model_ref = None  # will be set by main()

    def on_train_begin(self, args, state, control, **kwargs):
        self.run = swanlab.init(project=self.project, experiment_name=self.exp_name, config=self.config)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.run is None or logs is None:
            return
        step = int(state.global_step)

        merged = dict(logs)

        # merge model cached losses
        if self.model_ref is not None and getattr(self.model_ref, "_last_losses", None) is not None:
            merged.update(self.model_ref._last_losses)

        # ✅ 不上传文本 sample，只上传一个数值：sample 是否预测正确（归一化后）
        if self.log_sample and self.model_ref is not None and getattr(self.model_ref, "_last_sample", None) is not None:
            s = self.model_ref._last_sample
            if isinstance(s, dict) and ("gt_norm" in s) and ("pred_norm" in s):
                merged["sample/is_correct"] = float(s["gt_norm"] == s["pred_norm"])

        self.run.log(merged, step=step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.run is not None:
            self.run.finish()


# -------------------------
# Custom trainer: generate sample every N steps and cache it in model + save to local jsonl
# -------------------------
class CustomTrainer(Trainer):
    def __init__(self, *args, log_sample_every: int = 50, max_sample_new_tokens: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_sample_every = int(log_sample_every)
        self.max_sample_new_tokens = int(max_sample_new_tokens)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            pixel_values=inputs["pixel_values"],
        )
        loss = outputs.loss

        # 周期性生成 sample（不要每步都做）
        try:
            step = int(self.state.global_step)
            if self.log_sample_every > 0 and (step % self.log_sample_every == 0):
                with torch.no_grad():
                    full_text = model.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

                    lbl = inputs["labels"][0]
                    gt_ids = lbl[lbl != -100]
                    gt_text = model.tokenizer.decode(gt_ids, skip_special_tokens=True).strip()

                    pred_full = model.generate_from_batch(
                        input_ids=inputs["input_ids"][:1],
                        attention_mask=inputs["attention_mask"][:1],
                        pixel_values=inputs["pixel_values"][:1],
                        max_new_tokens=self.max_sample_new_tokens,
                    )

                    # 归一化 pred：如果你 prompt 里有 "Answer:"，可只取 Answer 后面
                    pred_ans = pred_full.split("Answer:", 1)[-1].strip() if "Answer:" in pred_full else pred_full

                    sample = {
                        "step": step,
                        "full_input": full_text[:800],
                        "gt": gt_text[:200],
                        "pred_full": pred_full[:800],
                        "gt_norm": normalize_answer(gt_text),
                        "pred_norm": normalize_answer(pred_ans),
                    }

                    model._last_sample = sample

                    # ✅ 保存到本地 jsonl（追加）
                    os.makedirs(self.args.output_dir, exist_ok=True)
                    sample_path = os.path.join(self.args.output_dir, "train_samples.jsonl")
                    with open(sample_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        except Exception:
            pass

        return (loss, outputs) if return_outputs else loss


# -------------------------
# main
# -------------------------
def main():
    # ===== OSVQA paths =====
    osvqa_root = "/mnt/data/mm_data/OSVQA_开源/OSVQA_1.0"
    ann_train = os.path.join(osvqa_root, "annotations", "all_train_annotation.json")
    rgb_dir = os.path.join(osvqa_root, "images", "rgb")

    # ===== model paths =====
    qwen_path = "/mnt/data/zhuxiang/Qwen/Qwen3-4B"
    remoteclip_ckpt = "/home/xingyueao/RemoteClip/chendelong/RemoteClip/RemoteCLIP-ViT-L-14.pt"

    # stage1 projector (continue training)
    stage1_projector = "/mnt/data/zhuxiang/Qwen/Remote-Clip+Qwen/outputs/stage2/V1/stage1_rsicd_projector/projector.pt"
    out_dir = "/mnt/data/zhuxiang/Qwen/Remote-Clip+Qwen/outputs/stage2/V1/stage2_osvqa_rgb_connector_lora"

    model = Stage2ConnectorLoRAModel(
        qwen_path=qwen_path,
        remoteclip_ckpt_path=remoteclip_ckpt,
        image_token="<image>",
        trust_remote_code=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        # 如果你觉得 LoRA 太大：只 attention
        # lora_target_modules=["q_proj","k_proj","v_proj","o_proj"],
    )

    # load stage1 connector
    if stage1_projector and os.path.isfile(stage1_projector):
        sd = torch.load(stage1_projector, map_location="cpu")
        model.vision_tower.projector.load_state_dict(sd, strict=True)
        print(f"[Resume] loaded stage1 projector: {stage1_projector}")
    else:
        print(f"[Resume] stage1 projector not found: {stage1_projector} (train from scratch)")

    n_train, n_all, ratio = count_trainable(model)
    print(f"[Params] trainable params: {n_train:,} || all params: {n_all:,} || trainable%: {ratio*100:.4f}%")

    # ===== dataset =====
    train_ds = OSVQAStage2RGBDataset(
        ann_path=ann_train,
        rgb_dir=rgb_dir,
        preprocess=model.vision_tower.preprocess,
        tokenizer=model.tokenizer,
        image_token="<image>",
        max_length=512,
        add_task_prefix=True,
        return_meta=False,
    )
    collator = Stage2Collator(pad_token_id=model.tokenizer.pad_token_id)

    # ===== training args =====
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=3,
        bf16=True,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
    )

    # ===== swanlab =====
    swan_project = os.environ.get("SWAN_PROJECT", "qwen3-remoteclip-stage2")
    swan_exp = os.environ.get("SWAN_EXP", "osvqa_connector_lora")
    swan_cfg = {
        "qwen_path": qwen_path,
        "remoteclip_ckpt": remoteclip_ckpt,
        "stage1_projector": stage1_projector,
        "out_dir": out_dir,
        "bs": args.per_device_train_batch_size,
        "grad_accum": args.gradient_accumulation_steps,
        "lr": args.learning_rate,
        "epochs": args.num_train_epochs,
        "trainable_params": n_train,
        "trainable_ratio": ratio,
    }

    swan_cb = SwanLabCallback(project=swan_project, exp_name=swan_exp, config=swan_cfg, log_sample=True)
    swan_cb.model_ref = model  # important

    # ===== trainer =====
    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
        log_sample_every=20,           # 每 20 step 保存一次输出到本地 train_samples.jsonl
        max_sample_new_tokens=16,
    )
    trainer.add_callback(swan_cb)

    trainer.train()

    # ===== save =====
    os.makedirs(out_dir, exist_ok=True)

    # save connector/projector
    torch.save(model.vision_tower.projector.state_dict(), os.path.join(out_dir, "connector.pt"))
    print("Saved connector:", os.path.join(out_dir, "connector.pt"))

    # save LoRA adapter
    lora_dir = os.path.join(out_dir, "lora")
    model.llm.save_pretrained(lora_dir)
    print("Saved LoRA:", lora_dir)

    # save tokenizer (important for <image>)
    model.tokenizer.save_pretrained(out_dir)
    print("Saved tokenizer:", out_dir)

    print("Training finished. Samples saved to:", os.path.join(out_dir, "train_samples.jsonl"))


if __name__ == "__main__":
    main()