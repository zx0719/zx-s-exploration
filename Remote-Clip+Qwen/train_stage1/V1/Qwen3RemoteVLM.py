import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.insert(0, "/home/xingyueao/RemoteClip")
from RemoteVisionTower import RemoteVisionTower, TowerConfig


class Qwen3RemoteVLM(nn.Module):
    """
    prompt 里包含 <image>，用 RemoteVisionTower 生成的视觉 tokens 替换它，
    再把拼好的 inputs_embeds 喂给 Qwen3 生成。
    """
    def __init__(
        self,
        qwen_name_or_path: str = "Qwen/Qwen3-4B",
        remoteclip_ckpt_path: str = "/home/xingyueao/RemoteClip/chendelong/RemoteClip/RemoteCLIP-ViT-L-14.pt",
        image_token: str = "<image>",
        device: str = "cuda",
        trust_remote_code: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.image_token = image_token

        # --- Tokenizer / LLM ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            qwen_name_or_path, use_fast=True, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # add <image> token
        if image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})

        self.llm = AutoModelForCausalLM.from_pretrained(
            qwen_name_or_path,
            torch_dtype="auto",
            device_map=None,
            trust_remote_code=trust_remote_code,
        )
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # ✅ 动态获取 Qwen hidden_size（你这里就是 2560）
        llm_hidden = int(self.llm.config.hidden_size)
        print(f"[Init] Qwen hidden_size = {llm_hidden}")

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)

        # --- Vision tower（用 llm_hidden 来建 projector 输出维度） ---
        tower_cfg = TowerConfig(hidden_size=llm_hidden, mm_hidden_size=1024)
        self.vision_tower = RemoteVisionTower(tower_cfg, model_path=remoteclip_ckpt_path)

        self.to(self.device)

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: (B,3,224,224)
        returns: (B,196,llm_hidden)
        """
        self.vision_tower.eval()
        feats = self.vision_tower(pixel_values)  # (B,196,H)
        return feats

    def _merge_text_and_vision(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vision_tokens: torch.Tensor,
    ):
        B, T = input_ids.shape
        Bv, Tv, Dv = vision_tokens.shape
        assert B == Bv

        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B,T,D)
        vision_tokens = vision_tokens.to(dtype=text_embeds.dtype, device=text_embeds.device)

        Dt = text_embeds.shape[-1]
        if Dt != Dv:
            raise RuntimeError(f"Hidden mismatch: text_embeds={Dt}, vision_tokens={Dv}. "
                               f"Check projector output dim == llm hidden_size.")

        new_embeds = []
        new_attn = []

        for b in range(B):
            ids = input_ids[b]
            attn = attention_mask[b]

            pos = (ids == self.image_token_id).nonzero(as_tuple=False)
            if pos.numel() == 0:
                new_embeds.append(text_embeds[b])
                new_attn.append(attn)
                continue

            p0 = int(pos[0].item())

            left_e = text_embeds[b, :p0, :]
            right_e = text_embeds[b, p0 + 1 :, :]
            merged_e = torch.cat([left_e, vision_tokens[b], right_e], dim=0)  # (T-1+Tv, D)

            left_a = attn[:p0]
            right_a = attn[p0 + 1 :]
            vis_a = torch.ones(Tv, device=attn.device, dtype=attn.dtype)
            merged_a = torch.cat([left_a, vis_a, right_a], dim=0)

            new_embeds.append(merged_e)
            new_attn.append(merged_a)

        max_len = max(x.size(0) for x in new_embeds)
        padded_e = []
        padded_a = []
        for b in range(B):
            e = new_embeds[b]
            a = new_attn[b]
            pad_len = max_len - e.size(0)
            if pad_len > 0:
                e = torch.cat([e, torch.zeros(pad_len, e.size(1), device=e.device, dtype=e.dtype)], dim=0)
                a = torch.cat([a, torch.zeros(pad_len, device=a.device, dtype=a.dtype)], dim=0)
            padded_e.append(e)
            padded_a.append(a)

        return torch.stack(padded_e, dim=0), torch.stack(padded_a, dim=0)

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.7,
    ) -> str:
        self.eval()
        image = image.to(self.device)

        # 1) tokenize
        enc = self.tokenizer(prompt, return_tensors="pt", padding=False)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # ✅ 先拿到文本 embedding 的 dtype（通常 bf16）
        text_dtype = self.llm.get_input_embeddings().weight.dtype

        # 2) vision tokens
        vision_tokens = self.encode_image(image)  # (1,196,H) 可能是 fp32
        # ✅ 对齐 dtype/device
        vision_tokens = vision_tokens.to(device=self.device, dtype=text_dtype)

        # 3) merge -> inputs_embeds
        inputs_embeds, attn = self._merge_text_and_vision(input_ids, attention_mask, vision_tokens)

        # 4) generate
        out_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )


        print("llm embed dtype:", self.llm.get_input_embeddings().weight.dtype)
        print("vision dtype before:", vision_tokens.dtype)


        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

