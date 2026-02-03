import os
import json
import re
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from train_stage2_osvqa import Stage2ConnectorLoRAModel
from osvqa_dataset import OSVQAStage2RGBDataset, Stage2Collator


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


def is_yesno(gt_norm: str) -> bool:
    return gt_norm in {"yes", "no"}


def evaluate_osvqa(model: Stage2ConnectorLoRAModel, dataset: OSVQAStage2RGBDataset, batch_size: int = 1):
    assert dataset.return_meta is True, "Eval dataset must set return_meta=True"

    device = next(model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=Stage2Collator(pad_token_id=model.tokenizer.pad_token_id),
    )

    stats = {
        "overall": {"n": 0, "correct": 0, "yn_n": 0, "yn_correct": 0},
        "by_task": defaultdict(lambda: {"n": 0, "correct": 0, "yn_n": 0, "yn_correct": 0}),
    }

    model.eval()
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        metas = batch["meta"]

        for i, meta in enumerate(metas):
            task = meta["task_type"]
            q = meta["question"]
            gt = meta["answer"]

            prompt = dataset.build_prompt(task, q)
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
    items = sorted(report["by_task"].items(), key=lambda x: x[1]["n"], reverse=True)
    for task, st in items:
        line = f"{task:10s}  acc={st['acc']:.4f} (n={st['n']})"
        if st["yesno_n"] > 0:
            line += f" | yes/no acc={st['yesno_acc']:.4f} (n={st['yesno_n']})"
        print(line)


if __name__ == "__main__":
    # Configure paths to match your environment
    osvqa_root = "/mnt/data/mm_data/OSVQA_开源/OSVQA_1.0"
    ann_val = os.path.join(osvqa_root, "annotations", "all_val_annotation.json")
    ann_test = os.path.join(osvqa_root, "annotations", "all_test_annotation.json")
    rgb_dir = os.path.join(osvqa_root, "images", "rgb")

    qwen_path = "/mnt/data/zhuxiang/Qwen/Qwen3-4B"
    remoteclip_ckpt = "/home/xingyueao/RemoteClip/chendelong/RemoteClip/RemoteCLIP-ViT-L-14.pt"
    stage1_projector = "/mnt/data/zhuxiang/Qwen/Remote-Clip+Qwen/outputs/stage2/V1/stage1_rsicd_projector/projector.pt"

    # The outputs from training
    out_dir = "/mnt/data/zhuxiang/Qwen/Remote-Clip+Qwen/outputs/stage2/V1/stage2_osvqa_rgb_connector_lora"
    connector_ckpt = os.path.join(out_dir, "connector.pt")
    lora_dir = os.path.join(out_dir, "lora")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Stage2ConnectorLoRAModel(
        qwen_path=qwen_path,
        remoteclip_ckpt_path=remoteclip_ckpt,
        image_token="<image>",
        trust_remote_code=False,
    )

    # load stage1 projector if exists
    if stage1_projector and os.path.isfile(stage1_projector):
        sd = torch.load(stage1_projector, map_location="cpu")
        model.vision_tower.projector.load_state_dict(sd, strict=True)

    # load trained connector if exists
    if os.path.isfile(connector_ckpt):
        sd = torch.load(connector_ckpt, map_location="cpu")
        model.vision_tower.projector.load_state_dict(sd, strict=True)
        print(f"[Load] connector from {connector_ckpt}")

    # load LoRA
    if os.path.isdir(lora_dir):
        # PeftModel wrapper will load adapter weights
        model.llm = PeftModel.from_pretrained(model.llm, lora_dir)
        print(f"[Load] LoRA from {lora_dir}")

    model.to(device)

    val_ds = OSVQAStage2RGBDataset(
        ann_path=ann_val,
        rgb_dir=rgb_dir,
        preprocess=model.vision_tower.preprocess,
        tokenizer=model.tokenizer,
        image_token="<image>",
        max_length=512,
        add_task_prefix=True,
        return_meta=True,
    )
    test_ds = OSVQAStage2RGBDataset(
        ann_path=ann_test,
        rgb_dir=rgb_dir,
        preprocess=model.vision_tower.preprocess,
        tokenizer=model.tokenizer,
        image_token="<image>",
        max_length=512,
        add_task_prefix=True,
        return_meta=True,
    )

    val_report = evaluate_osvqa(model, val_ds, batch_size=1)
    test_report = evaluate_osvqa(model, test_ds, batch_size=1)

    pretty_print_report("VAL", val_report)
    pretty_print_report("TEST", test_report)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval_val.json"), "w", encoding="utf-8") as f:
        json.dump(val_report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "eval_test.json"), "w", encoding="utf-8") as f:
        json.dump(test_report, f, ensure_ascii=False, indent=2)

    print("Saved eval reports to:", out_dir)
