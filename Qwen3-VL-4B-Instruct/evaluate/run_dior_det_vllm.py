#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import time
import random
import argparse
from typing import List, Dict, Optional, Tuple, Any

from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoProcessor

import swanlab
from vllm import LLM, SamplingParams


# ===================== SwanLab 固定配置（只改这里，不走命令行） =====================
SWAN_PROJECT = "qwen3vl_dior"
SWAN_EXPERIMENT = "dior_det_vllm"
SWAN_TAGS = ["qwen3vl", "DIOR", "det", "vllm"]
SWAN_MODE = "cloud"                 # 有网用 cloud；没网用 offline
# ===================================================================================


# -------------------------- DIOR 数据扫描 --------------------------
def find_images_recursive(root: str, subdirs: List[str]) -> List[str]:
    imgs = []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    for sd in subdirs:
        p = os.path.join(root, sd)
        if os.path.isdir(p):
            for ext in exts:
                imgs += glob.glob(os.path.join(p, "**", ext), recursive=True)
    return sorted(set(imgs))


def build_image_id_map(image_paths: List[str]) -> Dict[str, str]:
    mp = {}
    for p in image_paths:
        image_id = os.path.splitext(os.path.basename(p))[0]
        if image_id not in mp:
            mp[image_id] = p
    return mp


def find_split_file(imagesets_root: str, split: str) -> Optional[str]:
    if not os.path.isdir(imagesets_root):
        return None
    pats = [
        os.path.join(imagesets_root, "**", f"{split}.txt"),
        os.path.join(imagesets_root, "**", "Main", f"{split}.txt"),
        os.path.join(imagesets_root, "**", "main", f"{split}.txt"),
    ]
    hits = []
    for pat in pats:
        hits += glob.glob(pat, recursive=True)
    if not hits:
        return None
    hits = sorted(hits, key=lambda x: len(x))
    return hits[0]


def read_image_ids_from_split(split_file: str) -> List[str]:
    ids = []
    with open(split_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(line.split()[0])
    return ids


def infer_classes_from_imagesets(imagesets_root: str) -> List[str]:
    """
    DIOR 一般能从 ImageSets/Main/*.txt 推断 20 类。
    更鲁棒：Main / main 都搜，且支持任意 split 后缀。
    """
    if not os.path.isdir(imagesets_root):
        return []

    patterns = [
        os.path.join(imagesets_root, "**", "Main", "*.txt"),
        os.path.join(imagesets_root, "**", "main", "*.txt"),
    ]
    files = []
    for pat in patterns:
        files += glob.glob(pat, recursive=True)

    classes = set()
    for p in files:
        name = os.path.basename(p)
        m = re.match(r"(.+?)_(train|trainval|test|val)\.txt$", name, flags=re.I)
        if m:
            cls = m.group(1).strip()
            if cls and cls.lower() not in ["train", "test", "val", "trainval"]:
                classes.add(cls)

    return sorted(classes)


def infer_classes_from_voc_xml(ann_root: str, max_xml: int = 8000) -> List[str]:
    if not os.path.isdir(ann_root):
        return []
    import xml.etree.ElementTree as ET

    xmls = glob.glob(os.path.join(ann_root, "**", "*.xml"), recursive=True)
    classes = set()
    for x in xmls[:max_xml]:
        try:
            root = ET.parse(x).getroot()
            for obj in root.findall("object"):
                name = obj.findtext("name")
                if name:
                    classes.add(name.strip())
        except Exception:
            continue
    return sorted(classes)


# -------------------------- Prompt & Chat Template --------------------------
def make_prompt(classes: List[str], score_thr: float) -> str:
    if classes:
        class_list = ", ".join(classes)
        class_hint = f"Detect objects ONLY from this class list:\n[{class_list}]\n"
    else:
        class_hint = "Detect objects in the image (no fixed class list available).\n"

    return (
        "You are an expert in aerial object detection.\n"
        f"{class_hint}\n"
        "Output MUST be a single JSON object with keys: objects, count.\n"
        "Schema:\n"
        '{"objects":[{"class":"<label>","bbox":[x1,y1,x2,y2],"score":0.0}], "count":0}\n'
        "Rules:\n"
        "- bbox are integer pixel coords in the image (x1 < x2, y1 < y2).\n"
        "- score is a float in [0,1].\n"
        f"- Only include objects with score >= {score_thr}.\n"
        "- Do NOT output any other text (no markdown, no explanations).\n"
    )


def build_chat_text(processor, prompt: str) -> str:
    """
    Qwen3-VL 必须插入 image placeholder token
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# -------------------------- JSON 解析（更鲁棒） --------------------------
_JSON_BLOCK_RE = re.compile(r"\{.*\}", flags=re.S)


def _extract_json_blob(text: str) -> Optional[str]:
    if not text:
        return None

    idx = text.rfind("assistant")
    if idx >= 0:
        tail = text[idx:]
        m = _JSON_BLOCK_RE.search(tail)
        if m:
            return m.group(0)

    m = _JSON_BLOCK_RE.search(text)
    if m:
        return m.group(0)

    return None


def _try_fix_common_json_errors(blob: str) -> str:
    if not blob:
        return blob

    last = blob.rfind("}")
    if last >= 0:
        blob = blob[: last + 1]

    def _split_bbox_string(m):
        s = m.group(1)
        parts = [p.strip().strip('"').strip("'") for p in s.split(",")]
        parts = [p for p in parts if p != ""]
        if len(parts) >= 4:
            parts = parts[:4]
            nums = []
            for p in parts:
                try:
                    nums.append(int(float(p)))
                except Exception:
                    nums.append(0)
            return f'"bbox":[{nums[0]},{nums[1]},{nums[2]},{nums[3]}]'
        return m.group(0)

    blob = re.sub(r'"bbox"\s*:\s*\[\s*("?[\d\s\.,-]+"?)\s*\]', _split_bbox_string, blob)

    blob = re.sub(
        r'"bbox"\s*:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]',
        r'"bbox":[\1,\2,\3,\4]',
        blob,
    )

    blob = re.sub(r'"bbox"\s*:\s*\[\s*\[', r'"bbox":[', blob)

    # 纠错：tennicourt -> tenniscourt
    blob = blob.replace("tennicourt", "tenniscourt")

    return blob


def parse_pred(text: str) -> Tuple[Dict[str, Any], bool]:
    blob = _extract_json_blob(text)
    if not blob:
        return {"raw": text}, False

    blob2 = _try_fix_common_json_errors(blob)
    try:
        obj = json.loads(blob2)
        if isinstance(obj, dict):
            return obj, True
        return {"raw": text}, False
    except Exception:
        return {"raw": text}, False


def clamp_and_validate_objects(pred: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, int]:
    if not isinstance(pred, dict):
        return pred, False, 0

    objs = pred.get("objects", [])
    if not isinstance(objs, list):
        pred["objects"] = []
        pred["count"] = 0
        return pred, True, 0

    new_objs = []
    for o in objs:
        if not isinstance(o, dict):
            continue
        cls = str(o.get("class", "")).strip()
        score = o.get("score", None)
        bbox = o.get("bbox", None)

        if not cls or bbox is None:
            continue

        if isinstance(bbox, list) and len(bbox) == 4:
            vals = []
            ok = True
            for v in bbox:
                try:
                    vals.append(int(float(v)))
                except Exception:
                    ok = False
                    break
            if not ok:
                continue
            x1, y1, x2, y2 = vals
        else:
            continue

        if not (x1 < x2 and y1 < y2):
            continue

        try:
            score_f = float(score)
        except Exception:
            score_f = 0.0
        score_f = max(0.0, min(1.0, score_f))

        new_objs.append({"class": cls, "bbox": [x1, y1, x2, y2], "score": score_f})

    pred["objects"] = new_objs
    pred["count"] = len(new_objs)
    return pred, True, len(new_objs)


# -------------------------- vLLM 初始化 --------------------------
def init_vllm(
    model_dir: str,
    max_model_len: int,
    max_num_seqs: int,
    gpu_mem_util: float,
    enforce_eager: bool,
):
    """
    避免你之前的启动失败：
    默认 gpu_memory_utilization=0.85（可通过参数改）
    """
    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        limit_mm_per_prompt={"image": 1},
        # gpu_memory_utilization=gpu_mem_util,
        # enforce_eager=enforce_eager,
        disable_log_stats=True,
        enable_prefix_caching=True,
    )
    return llm


# -------------------------- 主流程 --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="/mnt/data/zhuxiang/Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--dior_root", default="/mnt/data/mm_data/DIOR")
    ap.add_argument("--split", default="test", choices=["test", "trainval", "train", "all"])
    ap.add_argument("--out", default="dior_pred.jsonl")
    ap.add_argument("--limit", type=int, default=0)

    ap.add_argument("--seed", type=int, default=0, help="limit 抽样前是否打散；0=不打散，>0 用该 seed 打散")
    ap.add_argument("--batch_size", type=int, default=4, help="vLLM 一次并发请求数")
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--score_thr", type=float, default=0.3)

    # vLLM 显存相关
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--enforce_eager", action="store_true")

    args = ap.parse_args()

    # ---------------- SwanLab init ----------------
    run = swanlab.init(
        project=SWAN_PROJECT,
        experiment_name=SWAN_EXPERIMENT,
        tags=SWAN_TAGS,
        mode=SWAN_MODE,
    )
    run.log({
        "limit": int(args.limit),
        "batch_size": int(args.batch_size),
        "max_model_len": int(args.max_model_len),
        "max_new_tokens": int(args.max_new_tokens),
        "score_thr": float(args.score_thr),
        "gpu_mem_util": float(args.gpu_mem_util),
        "enforce_eager": int(bool(args.enforce_eager)),
        "torch_cuda": int(torch.cuda.is_available()),
    })
    run.log({
        "model_dir_text": swanlab.Text(str(args.model_dir)),
        "dior_root_text": swanlab.Text(str(args.dior_root)),
        "split_text": swanlab.Text(str(args.split)),
        "out_text": swanlab.Text(str(args.out)),
    })

    try:
        processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)

        # 递归找图（兼容 JPEGImages-test/JPEGImages-test/*.jpg）
        image_subdirs = ["JPEGImages-test", "JPEGImages-trainval"]
        all_images = find_images_recursive(args.dior_root, image_subdirs)
        if not all_images:
            raise RuntimeError(
                f"没找到任何图片。请检查：{args.dior_root}/JPEGImages-test 或 JPEGImages-trainval 下是否有 jpg/png"
            )
        id2path = build_image_id_map(all_images)

        imagesets_root = os.path.join(args.dior_root, "ImageSets")
        split_file = None
        selected_images: List[str] = []

        if args.split != "all":
            split_file = find_split_file(imagesets_root, args.split)
            if split_file:
                ids = read_image_ids_from_split(split_file)
                for image_id in ids:
                    p = id2path.get(image_id)
                    if p:
                        selected_images.append(p)
                if not selected_images:
                    print(f"[WARN] 找到 split 文件 {split_file} 但无法匹配图片路径，改用全量图片。")
                    selected_images = all_images
            else:
                print(f"[WARN] 没找到 {args.split}.txt，改用全量图片。")
                selected_images = all_images
        else:
            selected_images = all_images

        if args.seed and args.seed > 0:
            rnd = random.Random(args.seed)
            rnd.shuffle(selected_images)

        if args.limit and args.limit > 0:
            selected_images = selected_images[: args.limit]

        # 类别推断：优先 ImageSets/Main，其次 XML；失败则不限制类别继续跑
        classes = infer_classes_from_imagesets(imagesets_root)
        if not classes:
            classes = infer_classes_from_voc_xml(os.path.join(args.dior_root, "Annotations"))

        if classes:
            print(f"[INFO] Classes inferred: {len(classes)} (e.g. {classes[:10]})")
            run.log({"num_classes": int(len(classes))})
            run.log({"classes_text": swanlab.Text(", ".join(classes))})
        else:
            print("[WARN] 没能推断出类别集合：将以“不限制类别”模式运行（不推荐，但可跑通）。")
            run.log({"num_classes": 0})
            run.log({"classes_text": swanlab.Text("UNKNOWN")})

        prompt = make_prompt(classes, args.score_thr)
        run.log({"prompt_preview": swanlab.Text(prompt[:2000])})

        print(f"[INFO] Total images to run: {len(selected_images)}")
        if split_file:
            print(f"[INFO] Using split file: {split_file}")
            run.log({"split_file_text": swanlab.Text(split_file)})

        # ---------------- vLLM init ----------------
        llm = init_vllm(
            model_dir=args.model_dir,
            max_model_len=args.max_model_len,
            max_num_seqs=max(args.batch_size, 4),
            gpu_mem_util=args.gpu_mem_util,
            enforce_eager=args.enforce_eager,
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=args.max_new_tokens,
        )

        # 统计
        total = 0
        parse_ok = 0
        has_objects = 0
        total_objects = 0
        total_time = 0.0

        sample_logged = False

        with open(args.out, "w", encoding="utf-8") as f:
            pbar = tqdm(range(0, len(selected_images), args.batch_size), desc="DIOR vLLM infer")
            step = 0

            for start in pbar:
                batch_paths = selected_images[start:start + args.batch_size]

                reqs = []
                metas = []

                for img_path in batch_paths:
                    image = Image.open(img_path).convert("RGB")
                    chat_text = build_chat_text(processor, prompt)
                    reqs.append({
                        "prompt": chat_text,
                        "multi_modal_data": {"image": image},
                    })
                    metas.append(img_path)

                t0 = time.time()
                outs = llm.generate(reqs, sampling_params=sampling_params)
                dt_batch = time.time() - t0
                total_time += dt_batch
                avg_dt = dt_batch / max(1, len(batch_paths))

                for out, img_path in zip(outs, metas):
                    step += 1
                    raw_text = out.outputs[0].text if out.outputs else ""

                    pred, ok_parse = parse_pred(raw_text)
                    if ok_parse:
                        parse_ok += 1

                    pred, _norm_ok, nobj = clamp_and_validate_objects(pred)
                    if nobj > 0:
                        has_objects += 1
                        total_objects += nobj

                    total += 1

                    rec = {"image": img_path, "pred": pred, "raw_text": raw_text}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    run.log({
                        "step": int(step),
                        "latency_sec": float(avg_dt),
                        "parse_ok": int(ok_parse),
                        "num_objects": int(nobj),
                    })

                    if not sample_logged:
                        run.log({
                            "sample_image_path": swanlab.Text(img_path),
                            "sample_pred": swanlab.Text(json.dumps(pred, ensure_ascii=False)[:4000]),
                            "sample_raw": swanlab.Text(raw_text[:4000]),
                        })
                        sample_logged = True

                pbar.set_postfix({
                    "parse_ok": f"{parse_ok/max(1,total):.2f}",
                    "has_obj": f"{has_objects/max(1,total):.2f}",
                })

        avg_latency = total_time / max(1, total)
        parse_ok_rate = parse_ok / max(1, total)
        has_objects_rate = has_objects / max(1, total)
        avg_objects = total_objects / max(1, total)

        print(f"[DONE] Saved: {args.out}")
        print(
            f"[STAT] avg_latency={avg_latency:.3f}s, "
            f"parse_ok={parse_ok}/{total} ({parse_ok_rate:.3f}), "
            f"has_objects={has_objects}/{total} ({has_objects_rate:.3f}), "
            f"avg_objects={avg_objects:.2f}"
        )

        run.log({
            "avg_latency_sec": float(avg_latency),
            "parse_ok_rate": float(parse_ok_rate),
            "has_objects_rate": float(has_objects_rate),
            "avg_objects": float(avg_objects),
            "total_images": int(total),
        })

    finally:
        try:
            swanlab.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
