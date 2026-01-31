#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, glob, argparse, time, random
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any

from PIL import Image
from tqdm import tqdm

import swanlab

# vLLM
from vllm import LLM, SamplingParams

# HF processor only (for chat template)
from transformers import AutoProcessor


# ===================== SwanLab 固定配置（你只改这里） =====================
SWAN_PROJECT = "qwen3vl_aid"
SWAN_EXPERIMENT = "aid_cls_vllm"
SWAN_TAGS = ["qwen3vl", "AID", "cls", "vllm"]
SWAN_MODE = "cloud"   # 联网 cloud；离线 offline
# =======================================================================


def build_chat_text(processor, prompt: str) -> str:
    """
    关键：Qwen3-VL 需要 prompt 中有 image placeholder token。
    仍然用 HF 的 apply_chat_template 生成。
    """
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_first_json(text: str) -> Dict[str, Any]:
    """
    尽量从模型输出里抽取第一个 JSON。
    你现在 parse_ok_rate=1.0，说明格式基本稳定。
    """
    s = text.find("{")
    e = text.rfind("}")
    if s >= 0 and e > s:
        blob = text[s:e + 1]
        try:
            return json.loads(blob)
        except Exception:
            return {"raw": text}
    return {"raw": text}


def collect_aid_pairs(aid_class_root: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    从 aid_class_root/<class>/*.jpg 收集 (img_path, label)
    """
    subdirs = [d for d in glob.glob(os.path.join(aid_class_root, "*")) if os.path.isdir(d)]
    subdirs = [d for d in subdirs if os.path.basename(d).lower() not in ["imagesets", "__macosx"]]

    labels = []
    pairs = []
    for d in subdirs:
        lb = os.path.basename(d)
        imgs = (
            glob.glob(os.path.join(d, "*.jpg"))
            + glob.glob(os.path.join(d, "*.jpeg"))
            + glob.glob(os.path.join(d, "*.png"))
        )
        if not imgs:
            continue
        labels.append(lb)
        for p in imgs:
            pairs.append((p, lb))

    labels = sorted(set(labels))
    return pairs, labels


def init_vllm(model_dir: str, max_model_len: int, max_num_seqs: int):
    """
    vLLM 初始化：
    - trust_remote_code=True（Qwen3-VL 必须）
    - limit_mm_per_prompt={"image": 1}（每个请求 1 张图）
    - max_num_seqs 建议 >= batch_size（否则会排队）
    """
    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        limit_mm_per_prompt={"image": 1},
    )
    return llm


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_dir", default="/mnt/data/zhuxiang/Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument(
        "--aid_root",
        default="/mnt/data/mm_data/AID/AID Data Set/AID/AID_dataset/AID",
        help="AID类别根目录（应为 <root>/<class>/*.jpg）",
    )
    ap.add_argument("--out", default="aid_pred.jsonl")

    ap.add_argument("--limit", type=int, default=0, help="只跑前N张做冒烟测试，0=全量")
    ap.add_argument("--seed", type=int, default=42, help="shuffle 随机种子（保证可复现）")

    # vLLM 相关
    ap.add_argument("--batch_size", type=int, default=4, help="vLLM batch size（越大吞吐越高）")
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--max_new_tokens", type=int, default=128)

    args = ap.parse_args()

    # ---------------- SwanLab init（不走命令行参数） ----------------
    run = swanlab.init(
        project=SWAN_PROJECT,
        experiment_name=SWAN_EXPERIMENT,
        tags=SWAN_TAGS,
        mode=SWAN_MODE,
        config={
            "model_dir": args.model_dir,
            "aid_root": args.aid_root,
            "limit": args.limit,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_model_len": args.max_model_len,
            "max_new_tokens": args.max_new_tokens,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    )

    try:
        if not os.path.isdir(args.aid_root):
            raise RuntimeError(f"AID目录不存在：{args.aid_root}")

        pairs, labels = collect_aid_pairs(args.aid_root)
        if len(pairs) == 0:
            raise RuntimeError(
                "没找到AID图片。\n"
                f"aid_root={args.aid_root}\n"
                "请确认存在 <root>/<class>/*.jpg 结构。"
            )

        # ✅ 关键：先 shuffle 再 limit，避免前 N 张全是一个类
        random.seed(args.seed)
        random.shuffle(pairs)

        if args.limit and args.limit > 0:
            pairs = pairs[:args.limit]

        dist = Counter([gt for _, gt in pairs]).most_common(10)

        # SwanLab：注意字符串用 Text，避免类型错误
        swanlab.log({
            "num_classes": len(labels),
            "num_images_total": len(pairs),
            "subset_class_dist_top10": swanlab.Text(json.dumps(dist, ensure_ascii=False), caption="top10"),
            "aid_root": swanlab.Text(args.aid_root, caption="aid_root"),
            "model_dir": swanlab.Text(args.model_dir, caption="model_dir"),
        })

        # processor 只用于 chat_template
        processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)

        label_list = ", ".join(labels)
        prompt = (
            "You are an expert in aerial scene classification.\n"
            f"Choose exactly ONE label from this list:\n[{label_list}]\n\n"
            "Output MUST be a single JSON object with keys: label, confidence, evidence.\n"
            "Rules:\n"
            "- label must be EXACTLY one of the provided labels.\n"
            "- confidence is a float in [0,1].\n"
            "- evidence is a short phrase (<= 12 words).\n"
            "- Do NOT output any other text.\n"
        )
        swanlab.log({"prompt_preview": swanlab.Text(prompt[:8000], caption="prompt")})

        # vLLM init
        llm = init_vllm(
            model_dir=args.model_dir,
            max_model_len=args.max_model_len,
            max_num_seqs=max(args.batch_size, 4),
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=args.max_new_tokens,
        )

        correct = 0
        total = 0
        total_time = 0.0

        # per-class
        cls_tot = defaultdict(int)
        cls_ok = defaultdict(int)

        sample_logged = False

        def run_batch(batch_items: List[Tuple[str, str]]):
            nonlocal correct, total, total_time, sample_logged

            reqs = []
            metas = []

            for img_path, gt in batch_items:
                image = Image.open(img_path).convert("RGB")
                chat_text = build_chat_text(processor, prompt)

                reqs.append({
                    "prompt": chat_text,
                    "multi_modal_data": {"image": image},
                })
                metas.append((img_path, gt))

            t0 = time.time()
            outputs = llm.generate(reqs, sampling_params=sampling_params)
            dt = time.time() - t0
            total_time += dt

            # outputs 顺序与 reqs 一一对应
            for out, (img_path, gt) in zip(outputs, metas):
                # vLLM 输出结构：out.outputs[0].text
                raw_text = out.outputs[0].text if out.outputs else ""

                pred = extract_first_json(raw_text)
                pred_label = str(pred.get("label", "")).strip()

                ok = int(pred_label == gt)
                correct += ok
                total += 1
                cls_tot[gt] += 1
                cls_ok[gt] += ok

                rec = {"image": img_path, "gt": gt, "pred": pred, "raw_text": raw_text}
                return rec, ok, dt / max(1, len(batch_items))

        # 主循环：batch 推理
        with open(args.out, "w", encoding="utf-8") as f:
            pbar = tqdm(range(0, len(pairs), args.batch_size), desc="AID vLLM infer")
            step = 0
            for start in pbar:
                batch = pairs[start:start + args.batch_size]
                # 这里一次 llm.generate
                reqs = []
                metas = []
                images = []

                for img_path, gt in batch:
                    images.append(Image.open(img_path).convert("RGB"))
                    metas.append((img_path, gt))
                    reqs.append({
                        "prompt": build_chat_text(processor, prompt),
                        "multi_modal_data": {"image": images[-1]},
                    })

                t0 = time.time()
                outs = llm.generate(reqs, sampling_params=sampling_params)
                dt_batch = time.time() - t0
                avg_dt = dt_batch / max(1, len(batch))
                total_time += dt_batch

                for out, (img_path, gt) in zip(outs, metas):
                    step += 1
                    raw_text = out.outputs[0].text if out.outputs else ""
                    pred = extract_first_json(raw_text)
                    pred_label = str(pred.get("label", "")).strip()

                    ok = int(pred_label == gt)
                    correct += ok
                    total += 1
                    cls_tot[gt] += 1
                    cls_ok[gt] += ok

                    rec = {"image": img_path, "gt": gt, "pred": pred, "raw_text": raw_text}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    swanlab.log({
                        "step": step,
                        "latency_sec": avg_dt,
                        "correct": ok,
                    })

                    if not sample_logged:
                        swanlab.log({
                            "sample_image": swanlab.Image(img_path, caption="sample"),
                            "sample_gt": swanlab.Text(gt, caption="gt"),
                            "sample_pred": swanlab.Text(json.dumps(pred, ensure_ascii=False), caption="pred"),
                            "sample_raw": swanlab.Text(raw_text[:4000], caption="raw"),
                        })
                        sample_logged = True

                pbar.set_postfix({
                    "acc": f"{correct/max(1,total):.3f}",
                    "avg_lat": f"{(total_time/max(1,total)):.2f}s",
                })

        acc = correct / max(1, total)
        avg_latency = total_time / max(1, total)

        per_cls = []
        for c in sorted(cls_tot.keys()):
            per_cls.append([c, cls_ok[c], cls_tot[c], round(cls_ok[c] / cls_tot[c], 4)])
        per_cls_sorted = sorted(per_cls, key=lambda x: x[2], reverse=True)[:10]

        print(f"[INFO] AID root = {args.aid_root}")
        print(f"[DONE] Accuracy: {correct}/{total} = {acc:.4f}")
        print(f"[STAT] avg_latency={avg_latency:.3f}s/image")
        print(f"[DONE] Saved: {args.out}")
        print(f"[STAT] subset class dist top10: {dist}")

        swanlab.log({
            "accuracy": acc,
            "avg_latency_sec": avg_latency,
            "total": total,
            "per_class_top10": swanlab.Text(json.dumps(per_cls_sorted, ensure_ascii=False), caption="per_class_top10"),
        })

    finally:
        try:
            swanlab.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
