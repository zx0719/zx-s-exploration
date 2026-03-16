# rsicd_dataset.py
import os
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from PIL import Image
from torch.utils.data import Dataset


def _norm_name(s: str) -> str:
    # 统一大小写、去空格
    return os.path.basename(s).strip().lower()

def load_rsicd_items(ann_path: str, images_dir: str) -> List[Tuple[str, List[str]]]:
    """
    保证“图片目录为全集”：
    - 遍历 images_dir 拿到所有图片（N=目录真实图片数）
    - 用标注尽可能匹配 caption
    - 输出统计信息帮助定位缺失原因
    """
    # 1) 扫描所有图片（全集）
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    all_img_paths = []
    for fn in os.listdir(images_dir):
        if fn.lower().endswith(exts):
            all_img_paths.append(os.path.join(images_dir, fn))
    all_img_paths.sort()

    if not all_img_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    # 2) 读标注，构建 caption_map
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict) and "images" in data, f"Unexpected json format keys={list(data.keys())}"

    caption_map: Dict[str, List[str]] = {}  # norm_fn -> [caps]
    ann_filenames = set()

    for im in data["images"]:
        fn = im.get("filename") or im.get("file_name") or im.get("name")
        if not fn:
            continue
        k = _norm_name(fn)
        ann_filenames.add(k)

        sents = im.get("sentences") or im.get("captions") or []
        caps = []
        for s in sents:
            if isinstance(s, dict):
                c = s.get("raw") or s.get("caption") or s.get("sent") or ""
            else:
                c = str(s)
            c = c.strip()
            if c:
                caps.append(c)

        if caps:
            caption_map.setdefault(k, []).extend(caps)

    # 3) 把图片目录中的每张图都转成 item
    items: List[Tuple[str, List[str]]] = []
    img_filenames = set()

    missing_caption = 0
    for p in all_img_paths:
        k = _norm_name(p)
        img_filenames.add(k)
        caps = caption_map.get(k, [])
        if not caps:
            missing_caption += 1
        items.append((p, caps))

    # 4) 统计：标注里有但图片目录没找到的文件
    ann_not_found = sorted(list(ann_filenames - img_filenames))[:20]

    print(f"[RSICD] images in dir: {len(all_img_paths)}")
    print(f"[RSICD] images with captions: {len(all_img_paths) - missing_caption}")
    print(f"[RSICD] images WITHOUT captions: {missing_caption}")
    print(f"[RSICD] annotated filenames not found in dir: {len(ann_filenames - img_filenames)}")
    if ann_not_found:
        print("[RSICD] examples of ann-not-found:", ann_not_found)

    return items


class RSICDStage1DatasetPair(Dataset):
    """
    把一图多句展平：每张图的每条 caption 都变成一条样本。
    labels：prompt 部分 -100，只对 caption 计算 loss。
    """
    def __init__(self, ann_path: str, images_dir: str, preprocess, tokenizer,
                 image_token: str = "<image>", max_length: int = 256):
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.max_length = max_length

        # 复用你现有 load_rsicd_items，但它返回 (img_path, [caps])
        raw_items = load_rsicd_items(ann_path, images_dir)

        # 展平
        self.pairs = []
        skipped = 0
        for img_path, caps in raw_items:
            if not caps:
                skipped += 1
                continue
            for cap in caps:
                self.pairs.append((img_path, cap))

        print(f"[RSICD Pair] pairs={len(self.pairs)} (skipped no-caption images={skipped})")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, cap = self.pairs[idx]

        img = Image.open(img_path).convert("RGB")
        pixel_values = self.preprocess(img)

        prompt = f"Describe the remote sensing image: {self.image_token}\nCaption: "
        full_text = prompt + cap

        enc_full = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=self.max_length)
        enc_prompt = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)

        input_ids = enc_full["input_ids"][0]
        attention_mask = enc_full["attention_mask"][0]
        labels = input_ids.clone()

        prompt_len = enc_prompt["input_ids"].shape[1]
        labels[:prompt_len] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }



@dataclass
class Stage1Collator:
    pad_token_id: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch
        max_len = max(x["input_ids"].shape[0] for x in batch)

        def pad_1d(x, pad_value):
            pad_len = max_len - x.shape[0]
            if pad_len <= 0:
                return x
            return torch.cat([x, torch.full((pad_len,), pad_value, dtype=x.dtype)], dim=0)

        input_ids = torch.stack([pad_1d(x["input_ids"], self.pad_token_id) for x in batch], dim=0)
        attention_mask = torch.stack([pad_1d(x["attention_mask"], 0) for x in batch], dim=0)
        labels = torch.stack([pad_1d(x["labels"], -100) for x in batch], dim=0)
        pixel_values = torch.stack([x["pixel_values"] for x in batch], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }
