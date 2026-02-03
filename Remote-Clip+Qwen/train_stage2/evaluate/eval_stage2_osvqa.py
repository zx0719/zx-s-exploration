# train_stage2_osvqa.py
import os
import re
import sys
import json
import math
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import swanlab
from transformers import TrainerCallback

from osvqa_dataset import OSVQAStage2RGBDataset, Stage2Collator

# 你的 RemoteClip 视觉塔
sys.path.insert(0, "/home/xingyueao/RemoteClip")
from RemoteVisionTower import RemoteVisionTower, TowerConfig


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def normalize_answer(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # 去掉常见标点
    s = re.sub(r"[\,\.\?\!\:\;\"\'\(\)\[\]\{\}]", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    # 归一 yes/no
    yes_set = {"yes", "y", "yeah", "yep", "true", "correct"}
    no_set = {"no", "n", "nope", "false", "incorrect"}
    if s in yes_set:
        return "yes"
    if s in no_set:
        return "no"
    return s


def is_yesno(gt_norm: str) -> bool:
    return gt_norm in {"yes", "no"}


class Stage2ConnectorLoRAModel(nn.Module):
    """
    Stage2: 训练 vision_tower.projector (connector) + Qwen LoRA
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

        # 冻结 vision backbone，只训 projector
        for p in self.vision_tower.parameters():
            p.requires_grad = False
        for p in self.vision_tower.projector.parameters():
            p.requires_grad = True

        # 冻结 llm 本体
        for p in self.llm.parameters():
            p.requires_grad = False

        # LoRA
        if lora_target_modules is None:
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


class SwanLabCallback(TrainerCallback):
    def __init__(self, project: str, exp_name: str, config: dict):
        super().__init__()
        self.project = project
        self.exp_name = exp_name
        self.config = config
        self.run = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.run = swanlab.init(project=self.project, experiment_name=self.exp_name, config=self.config)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.run is None or logs is None:
            return
        step = int(state.global_step)
        try:
            if hasattr(self, 'model') and getattr(self.model, '_last_losses', None) is not None:
                merged = dict(logs)
                merged.update(self.model._last_losses)
                # 如果有示例输出也一并上传（字符串）
                if getattr(self.model, '_last_sample', None) is not None:
                    merged['sample_output'] = self.model._last_sample
                self.run.log(merged, step=step)
                return
        except Exception:
            pass
        self.run.log(logs, step=step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.run is not None:
            self.run.finish()

    def _merge(self, input_ids, attention_mask, labels, vision_tokens):
        """
        用 vision_tokens 替换第一个 <image> token，并把视觉 token 的 labels 置 -100
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

    def _merge_for_generation(self, input_ids, attention_mask, vision_tokens):
        """
        generation 用：不需要 labels
        """
        B, T = input_ids.shape
        Bv, Tv, _ = vision_tokens.shape
        assert B == Bv

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        vision_tokens = vision_tokens.to(dtype=text_embeds.dtype, device=text_embeds.device)

        new_embeds, new_attn, new_lens = [], [], []
        for b in range(B):
            ids = input_ids[b]
            attn = attention_mask[b]

            pos = (ids == self.image_token_id).nonzero(as_tuple=False)
            if pos.numel() == 0:
                merged_e = torch.cat([vision_tokens[b], text_embeds[b]], dim=0)
                vis_a = torch.ones(Tv, device=attn.device, dtype=attn.dtype)
                merged_a = torch.cat([vis_a, attn], dim=0)
            else:
                p0 = int(pos[0].item())
                left_e = text_embeds[b, :p0, :]
                right_e = text_embeds[b, p0 + 1 :, :]
                merged_e = torch.cat([left_e, vision_tokens[b], right_e], dim=0)

                left_a = attn[:p0]
                right_a = attn[p0 + 1 :]
                vis_a = torch.ones(Tv, device=attn.device, dtype=attn.dtype)
                merged_a = torch.cat([left_a, vis_a, right_a], dim=0)

            new_embeds.append(merged_e)
            new_attn.append(merged_a)
            new_lens.append(int(merged_a.sum().item()))

        max_len = max(x.size(0) for x in new_embeds)
        padded_e, padded_a = [], []
        for b in range(B):
            e, a = new_embeds[b], new_attn[b]
            pad_len = max_len - e.size(0)
            if pad_len > 0:
                e = torch.cat([e, torch.zeros(pad_len, e.size(1), device=e.device, dtype=e.dtype)], dim=0)
                a = torch.cat([a, torch.zeros(pad_len, device=a.device, dtype=a.dtype)], dim=0)
            padded_e.append(e); padded_a.append(a)

        return torch.stack(padded_e, 0), torch.stack(padded_a, 0), new_lens

    def forward(self, input_ids, attention_mask, labels, pixel_values, **kwargs):
        vision_tokens = self.vision_tower(pixel_values)  # (B,196,H) 一般是
        inputs_embeds, attn, new_labels = self._merge(input_ids, attention_mask, labels, vision_tokens)
        lm_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=new_labels)

        # 保存子损失以便回调上报
        try:
            lm_loss = lm_outputs.loss if getattr(lm_outputs, "loss", None) is not None else None
            self._last_losses = {
                'lm_loss': float(lm_loss.detach().cpu()) if lm_loss is not None else None,
            }
        except Exception:
            self._last_losses = None

        # 尝试生成一个示例输出（只对 batch 首样本，no_grad 避免影响梯度）
        try:
            with torch.no_grad():
                gen_ids = self.llm.generate(
                    inputs_embeds=inputs_embeds[:1].detach(),
                    attention_mask=attn[:1],
                    max_new_tokens=16,
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                pred = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
                # 截断过长
                if len(pred) > 512:
                    pred = pred[:512]
                self._last_sample = pred
        except Exception:
            self._last_sample = None

        return lm_outputs

    @torch.no_grad()
    def generate_answer_from_prompt(self, prompt_text: str, pixel_values, max_new_tokens: int = 16) -> str:
        """
        单样本生成：给 prompt（不含答案），生成 answer 文本
        """
        device = next(self.parameters()).device
        self.eval()

        enc = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        pixel_values = pixel_values.to(device)

        vision_tokens = self.vision_tower(pixel_values)  # (1,Tv,D)
        inputs_embeds, attn, lens = self._merge_for_generation(input_ids, attention_mask, vision_tokens)
        prompt_len = lens[0]

        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # gen_ids: (1, prompt_len + new_tokens)（通常如此）
        # 切掉 prompt 部分
        if gen_ids.shape[1] > prompt_len:
            out_ids = gen_ids[0, prompt_len:]
        else:
            out_ids = gen_ids[0, :]

        text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        return text.strip()


def evaluate_osvqa(model: Stage2ConnectorLoRAModel, dataset: OSVQAStage2RGBDataset, batch_size: int = 1):
    """
    生成式评测：exact match accuracy（归一化后）
    - overall
    - yes/no 子集
    - 按 task_type 分组（整体 + yes/no）
    """
    assert dataset.return_meta is True, "Eval dataset must set return_meta=True"

    device = next(model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=batch_size,  # 建议 1，最稳
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=Stage2Collator(pad_token_id=model.tokenizer.pad_token_id),
    )

    # 统计结构
    stats = {
        "overall": {"n": 0, "correct": 0, "yn_n": 0, "yn_correct": 0},
        "by_task": defaultdict(lambda: {"n": 0, "correct": 0, "yn_n": 0, "yn_correct": 0}),
    }

    model.eval()
    for batch in loader:
        # batch_size=1 推荐；若你想 >1，也能跑，但这里按逐条生成写得更清晰
        pixel_values = batch["pixel_values"].to(device)
        metas = batch["meta"]

        for i, meta in enumerate(metas):
            task = meta["task_type"]
            q = meta["question"]
            gt = meta["answer"]

            prompt = dataset.build_prompt(task, q)  # 不含答案
            pred = model.generate_answer_from_prompt(prompt, pixel_values[i:i+1], max_new_tokens=16)

            gt_n = normalize_answer(gt)
            pred_n = normalize_answer(pred)

            ok = int(pred_n == gt_n)

            stats["overall"]["n"] += 1
            stats["overall"]["correct"] += ok

            stats["by_task"][task]["n"] += 1
            stats["by_task"][task]["correct"] += ok

            if is_yesno(gt_n):
                stats["overall"]["yn_n"] += 1
                stats["overall"]["yn_correct"] += ok
                stats["by_task"][task]["yn_n"] += 1
                stats["by_task"][task]["yn_correct"] += ok

    def acc(c, n):
        return (c / n) if n > 0 else float("nan")

    report = {
        "overall_acc": acc(stats["overall"]["correct"], stats["overall"]["n"]),
        "overall_yesno_acc": acc(stats["overall"]["yn_correct"], stats["overall"]["yn_n"]),
        "overall_n": stats["overall"]["n"],
        "overall_yesno_n": stats["overall"]["yn_n"],
        "by_task": {},
    }

    for task, st in stats["by_task"].items():
        report["by_task"][task] = {
            "acc": acc(st["correct"], st["n"]),
            "n": st["n"],
            "yesno_acc": acc(st["yn_correct"], st["yn_n"]),
            "yesno_n": st["yn_n"],
        }

    return report


def pretty_print_report(title: str, report: dict):
    print(f"\n========== {title} ==========")
    print(f"Overall: acc={report['overall_acc']:.4f}  (n={report['overall_n']})")
    if report["overall_yesno_n"] > 0:
        print(f"Yes/No: acc={report['overall_yesno_acc']:.4f} (n={report['overall_yesno_n']})")

    print("\n-- By task_type --")
    # 按样本数排序
    items = sorted(report["by_task"].items(), key=lambda x: x[1]["n"], reverse=True)
    for task, st in items:
        line = f"{task:10s}  acc={st['acc']:.4f} (n={st['n']})"
        if st["yesno_n"] > 0:
            line += f" | yes/no acc={st['yesno_acc']:.4f} (n={st['yesno_n']})"
        print(line)


def main():
    # ===== OSVQA 路径（按你贴的目录）=====
    osvqa_root = "/mnt/data/mm_data/OSVQA_开源/OSVQA_1.0"
    ann_train = os.path.join(osvqa_root, "annotations", "all_train_annotation.json")
    ann_val = os.path.join(osvqa_root, "annotations", "all_val_annotation.json")
    ann_test = os.path.join(osvqa_root, "annotations", "all_test_annotation.json")
    rgb_dir = os.path.join(osvqa_root, "images", "rgb")

    # ===== 模型与 ckpt =====
    qwen_path = "/mnt/data/zhuxiang/Qwen/Qwen3-4B"
    remoteclip_ckpt = "/home/xingyueao/RemoteClip/chendelong/RemoteClip/RemoteCLIP-ViT-L-14.pt"

    # (可选) 继续训练 Stage1 projector/connector
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
        # 显存紧张可只 LoRA 注意力：
        # lora_target_modules=["q_proj","k_proj","v_proj","o_proj"],
    )

    # 加载 Stage1 projector（强烈建议）
    if stage1_projector and os.path.isfile(stage1_projector):
        sd = torch.load(stage1_projector, map_location="cpu")
        model.vision_tower.projector.load_state_dict(sd, strict=True)
        print(f"[Resume] loaded projector from: {stage1_projector}")
    else:
        print(f"[Resume] stage1 projector not found: {stage1_projector} (train from scratch)")

    # ===== Dataset（训练不带 meta；评测带 meta）=====
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

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,  # global=32
        learning_rate=2e-4,
        num_train_epochs=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,

        logging_steps=10,
        save_strategy="steps",
        save_steps=1000,
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

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    # SwanLab callback：新建项目并与 swanlab 同步上报（将 model 绑定到 callback）
    swan_project = os.environ.get('SWAN_PROJECT', 'qwen3-remoteclip-stage2')
    swan_exp = os.environ.get('SWAN_EXP', 'osvqa_connector')
    swan_cfg = {
        'qwen_path': qwen_path,
        'remoteclip_ckpt': remoteclip_ckpt,
        'stage1_projector': stage1_projector,
        'out_dir': out_dir,
        'batch_size': args.per_device_train_batch_size,
        'grad_accum': args.gradient_accumulation_steps,
    }
    swan_cb = SwanLabCallback(project=swan_project, exp_name=swan_exp, config=swan_cfg)
    swan_cb.model = model

    trainer.add_callback(swan_cb)

    trainer.train()

    os.makedirs(out_dir, exist_ok=True)

    # 保存 connector/projector
    torch.save(model.vision_tower.projector.state_dict(), os.path.join(out_dir, "connector.pt"))
    print("Saved connector:", os.path.join(out_dir, "connector.pt"))

    # 保存 LoRA
    lora_dir = os.path.join(out_dir, "lora")
    model.llm.save_pretrained(lora_dir)
    print("Saved LoRA:", lora_dir)

    # tokenizer
    model.tokenizer.save_pretrained(out_dir)
    print("Saved tokenizer:", out_dir)

    print("Training finished. For evaluation, run the separate eval script: train_stage2/V1/eval_stage2_osvqa.py")


if __name__ == "__main__":
    main()
