# train_stage1_rsicd.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import sys
import swanlab
from torch.utils.data import ConcatDataset
from aid_stage1_dataset import AIDStage1Dataset
from transformers import TrainerCallback

sys.path.insert(0, "/home/xingyueao/RemoteClip")
from RemoteVisionTower import RemoteVisionTower, TowerConfig
from rsicd_dataset import RSICDStage1DatasetPair as RSICDStage1Dataset, Stage1Collator


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class SwanLabCallback(TrainerCallback):
    def __init__(self, project: str, exp_name: str, config: dict):
        super().__init__()
        self.project = project
        self.exp_name = exp_name
        self.config = config
        self.run = None

    def on_train_begin(self, args, state, control, **kwargs):
        # 初始化 swanlab run
        self.run = swanlab.init(
            project=self.project,
            experiment_name=self.exp_name,
            config=self.config,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.run is None or logs is None:
            return
        step = int(state.global_step)
        # logs 里通常包含 loss/learning_rate/grad_norm/epoch 等
        # 如果 callback 被赋予 model，并且 model 保存了最近 step 的子损失，则合并到上传的 logs
        try:
            if hasattr(self, 'model') and getattr(self.model, '_last_losses', None) is not None:
                merged = dict(logs)
                merged.update(self.model._last_losses)
                self.run.log(merged, step=step)
                return
        except Exception:
            pass

        self.run.log(logs, step=step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.run is not None:
            self.run.finish()



class Stage1ProjectorAlignModel(nn.Module):
    def __init__(
        self,
        qwen_path: str,
        remoteclip_ckpt_path: str,
        image_token: str = "<image>",
        trust_remote_code: bool = False,
        align_weight: float = 1.0,
        align_temp: float = 0.07,
    ):
        super().__init__()
        self.image_token = image_token
        self.align_weight = align_weight
        self.align_temp = align_temp

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
        print(f"[Init] Qwen hidden_size = {llm_hidden}")

        tower_cfg = TowerConfig(hidden_size=llm_hidden, mm_hidden_size=1024)
        self.vision_tower = RemoteVisionTower(tower_cfg, model_path=remoteclip_ckpt_path)

        # 冻结 LLM
        for p in self.llm.parameters():
            p.requires_grad = False

        # 只训 projector
        for p in self.vision_tower.projector.parameters():
            p.requires_grad = True

        # alignment loss settings (InfoNCE)
        self.align_loss_fn = nn.CrossEntropyLoss()

    def _merge(self, input_ids, attention_mask, labels, vision_tokens):
        """
        将第一个 <image> token 替换为 vision_tokens (B,Tv,D)
        并同步扩展 labels：视觉 token 对应位置 labels=-100
        """
        B, T = input_ids.shape
        Bv, Tv, Dv = vision_tokens.shape
        assert B == Bv

        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B,T,D)
        vision_tokens = vision_tokens.to(dtype=text_embeds.dtype, device=text_embeds.device)

        new_embeds, new_attn, new_labels = [], [], []
        for b in range(B):
            ids = input_ids[b]
            attn = attention_mask[b]
            lbl = labels[b]

            pos = (ids == self.image_token_id).nonzero(as_tuple=False)
            if pos.numel() == 0:
                # 如果 prompt 没写 <image>，就把视觉 token 插到最前面（更鲁棒）
                merged_e = torch.cat([vision_tokens[b], text_embeds[b]], dim=0)
                vis_a = torch.ones(Tv, device=attn.device, dtype=attn.dtype)
                merged_a = torch.cat([vis_a, attn], dim=0)
                vis_l = torch.full((Tv,), -100, device=lbl.device, dtype=lbl.dtype)
                merged_l = torch.cat([vis_l, lbl], dim=0)
            else:
                p0 = int(pos[0].item())
                left_e = text_embeds[b, :p0, :]
                right_e = text_embeds[b, p0 + 1 :, :]
                merged_e = torch.cat([left_e, vision_tokens[b], right_e], dim=0)

                left_a = attn[:p0]
                right_a = attn[p0 + 1 :]
                vis_a = torch.ones(Tv, device=attn.device, dtype=attn.dtype)
                merged_a = torch.cat([left_a, vis_a, right_a], dim=0)

                left_l = lbl[:p0]
                right_l = lbl[p0 + 1 :]
                vis_l = torch.full((Tv,), -100, device=lbl.device, dtype=lbl.dtype)
                merged_l = torch.cat([left_l, vis_l, right_l], dim=0)

            new_embeds.append(merged_e)
            new_attn.append(merged_a)
            new_labels.append(merged_l)

        max_len = max(x.size(0) for x in new_embeds)
        padded_e, padded_a, padded_l = [], [], []
        for b in range(B):
            e, a, l = new_embeds[b], new_attn[b], new_labels[b]
            pad_len = max_len - e.size(0)
            if pad_len > 0:
                e = torch.cat([e, torch.zeros(pad_len, e.size(1), device=e.device, dtype=e.dtype)], dim=0)
                a = torch.cat([a, torch.zeros(pad_len, device=a.device, dtype=a.dtype)], dim=0)
                l = torch.cat([l, torch.full((pad_len,), -100, device=l.device, dtype=l.dtype)], dim=0)
            padded_e.append(e); padded_a.append(a); padded_l.append(l)

        return torch.stack(padded_e, 0), torch.stack(padded_a, 0), torch.stack(padded_l, 0)

    def forward(self, input_ids, attention_mask, labels, pixel_values, **kwargs):
        vision_tokens = self.vision_tower(pixel_values)  # (B,196,H)

        # 1) 计算对比对齐损失（InfoNCE）
        # caption centroid: 使用 labels!=-100 的 token 的 embedding 均值
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B,T,D)
        caption_mask = (labels != -100)  # (B,T)
        caption_mask_f = caption_mask.unsqueeze(-1).to(dtype=text_embeds.dtype)
        caption_sum = (text_embeds * caption_mask_f).sum(dim=1)  # (B,D)
        caption_count = caption_mask.sum(dim=1).clamp(min=1).unsqueeze(-1).to(dtype=text_embeds.dtype)  # (B,1)
        caption_centroid = caption_sum / caption_count  # (B,D)

        vision_centroid = vision_tokens.mean(dim=1)  # (B,D)

        # L2-normalize for cosine similarity
        caption_norm = caption_centroid / (caption_centroid.norm(dim=1, keepdim=True).clamp(min=1e-6))
        vision_norm = vision_centroid / (vision_centroid.norm(dim=1, keepdim=True).clamp(min=1e-6))

        logits = torch.matmul(vision_norm, caption_norm.t()) / float(self.align_temp)  # (B,B)
        labels_pos = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
        # 对称 InfoNCE：image->text 和 text->image
        i2t_loss = self.align_loss_fn(logits, labels_pos)
        t2i_loss = self.align_loss_fn(logits.t(), labels_pos)
        align_loss = 0.5 * (i2t_loss + t2i_loss)

        # 2) 合并并计算语言建模损失
        inputs_embeds, attn, new_labels = self._merge(input_ids, attention_mask, labels, vision_tokens)
        lm_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=new_labels)

        lm_loss = lm_outputs.loss if getattr(lm_outputs, "loss", None) is not None else None

        if lm_loss is None:
            total_loss = align_loss * self.align_weight
        else:
            total_loss = lm_loss + self.align_weight * align_loss

        # 保存子损失用于 callback 上报
        try:
            self._last_losses = {
                'lm_loss': float(lm_loss.detach().cpu()) if lm_loss is not None else None,
                'align_loss': float(align_loss.detach().cpu()),
                'total_loss': float(total_loss.detach().cpu()),
            }
        except Exception:
            self._last_losses = None

        lm_outputs.loss = total_loss
        return lm_outputs


def main():
    # ====== 你给的真实路径 ======
    rsicd_root = "/mnt/data/xingyueao/BGM_IL/data/RSICD/RSICD"
    ann_path = os.path.join(rsicd_root, "dataset_rsicd.json")
    images_dir = os.path.join(rsicd_root, "RSICD_images")
    aid_root = "/mnt/data/mm_data/AID/AID Data Set/AID/AID_dataset/AID"
    qwen_path = "/mnt/data/zhuxiang/Qwen/Qwen3-4B"
    remoteclip_ckpt = "/home/xingyueao/RemoteClip/chendelong/RemoteClip/RemoteCLIP-ViT-L-14.pt"

    out_dir = "/mnt/data/zhuxiang/Qwen/Remote-Clip+Qwen/outputs/stage1/V2/stage1_rsicd_projector"

    model = Stage1ProjectorAlignModel(
        qwen_path=qwen_path,
        remoteclip_ckpt_path=remoteclip_ckpt,
        image_token="<image>",
        trust_remote_code=False,
        align_weight=1.0,
    )

    rsicd_dataset = RSICDStage1Dataset(
    ann_path=ann_path,
    images_dir=images_dir,
    preprocess=model.vision_tower.preprocess,
    tokenizer=model.tokenizer,
    image_token="<image>",
    max_length=256,
)

    aid_dataset = AIDStage1Dataset(
        aid_root=aid_root,
        preprocess=model.vision_tower.preprocess,
        tokenizer=model.tokenizer,
        image_token="<image>",
        max_length=256,
    )

    # 直接拼接：总体样本数 = RSICD + AID
    dataset = ConcatDataset([rsicd_dataset, aid_dataset])

    collator = Stage1Collator(pad_token_id=model.tokenizer.pad_token_id)


    print("[Trainable params]")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print("  ", n)


    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=8,   # global batch = 128
        learning_rate=2e-4,
        num_train_epochs=3,

        lr_scheduler_type="cosine",
        warmup_ratio=0.03,

        logging_steps=1,
        save_strategy="no",

        bf16=True,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        max_grad_norm=1.0,

        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
    )


    swan_cb = SwanLabCallback(
        project="qwen3-remoteclip-stage1",
        exp_name="rsicd_projector_align/loss1.0",
        config={
            "qwen_path": qwen_path,
            "remoteclip_ckpt": remoteclip_ckpt,
            "rsicd_root": rsicd_root,
            "batch_size": args.per_device_train_batch_size,
            "grad_accum": args.gradient_accumulation_steps,
            "lr": args.learning_rate,
            "epochs": args.num_train_epochs,
            "scheduler": args.lr_scheduler_type,
            "warmup_ratio": args.warmup_ratio,
            "max_length": 256,
            "image_tokens": 196,
                "llm_hidden": int(model.llm.config.hidden_size),
                "align_weight": float(model.align_weight),
                "align_temp": float(model.align_temp),
        },
    )

    # 让 callback 能访问 model 的 _last_losses
    swan_cb.model = model


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[swan_cb], 
    )

    trainer.train()

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.vision_tower.projector.state_dict(), os.path.join(out_dir, "projector.pt"))
    model.tokenizer.save_pretrained(out_dir)
    print("Saved projector:", os.path.join(out_dir, "projector.pt"))


if __name__ == "__main__":
    main()
