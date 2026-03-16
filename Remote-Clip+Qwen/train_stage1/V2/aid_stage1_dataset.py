# aid_stage1_dataset.py
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset


AID_TEMPLATES = [
    "A remote sensing image of {cls}.",
    "This aerial scene shows {cls}.",
    "The scene type is {cls}.",
    "An overhead view of {cls}.",
    "{cls} in a remote sensing image.",
]

def list_aid_items(aid_root: str) -> List[Tuple[str, str]]:
    """
    AID 目录结构：aid_root/<ClassName>/*.jpg
    返回 [(img_path, class_name), ...]
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    items = []
    classes = [d for d in os.listdir(aid_root) if os.path.isdir(os.path.join(aid_root, d))]
    classes.sort()
    for cls in classes:
        cls_dir = os.path.join(aid_root, cls)
        for fn in os.listdir(cls_dir):
            if fn.lower().endswith(exts):
                items.append((os.path.join(cls_dir, fn), cls))
    if not items:
        raise RuntimeError(f"No images found in AID root: {aid_root}")
    print(f"[AID] classes={len(classes)} images={len(items)} root={aid_root}")
    return items


class AIDStage1Dataset(Dataset):
    """
    Stage1：AID 没有自然语言标注，用类别名生成弱 caption，保持与 RSICDStage1Dataset 同接口：
    返回 pixel_values, input_ids, attention_mask, labels（prompt 部分 labels=-100）
    """
    def __init__(
        self,
        aid_root: str,
        preprocess,
        tokenizer,
        image_token: str = "<image>",
        max_length: int = 256,
        seed: int = 42,
        templates: List[str] = None,
    ):
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.max_length = max_length
        random.seed(seed)

        self.templates = templates or AID_TEMPLATES
        self.items = list_aid_items(aid_root)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, cls = self.items[idx]
        cls_txt = str(cls).replace("_", " ").strip()

        cap = random.choice(self.templates).format(cls=cls_txt)

        img = Image.open(img_path).convert("RGB")
        pixel_values = self.preprocess(img)

        # prompt 结构与 RSICD Stage1 一致 :contentReference[oaicite:2]{index=2}
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
