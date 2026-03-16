# osvqa_dataset.py
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset


def load_osvqa_listlist(path: str) -> List[List[Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    bad = [i for i, x in enumerate(data) if not (isinstance(x, list) and len(x) >= 4)]
    if bad:
        raise ValueError(f"Bad records indices (first 5): {bad[:5]}")
    return data


def _norm_img_name(x: str) -> str:
    x = str(x).strip()
    if x.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        return x
    return x + ".png"


class OSVQAStage2RGBDataset(Dataset):
    """
    OSVQA RGB only
    ann record: [img_name, task_type, question, answer]
    - return_meta=False: 训练用（只返回张量）
    - return_meta=True : 评测用（额外返回 meta）
    """
    def __init__(
        self,
        ann_path: str,
        rgb_dir: str,        # .../images/rgb
        preprocess,
        tokenizer,
        image_token: str = "<image>",
        max_length: int = 512,
        add_task_prefix: bool = True,
        return_meta: bool = False,
    ):
        self.ann_path = ann_path
        self.rgb_dir = rgb_dir
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.max_length = max_length
        self.add_task_prefix = add_task_prefix
        self.return_meta = return_meta

        self.items = load_osvqa_listlist(ann_path)
        print(f"[OSVQA RGB] ann={ann_path} items={len(self.items)}")
        print(f"[OSVQA RGB] rgb_dir={rgb_dir}")

    def __len__(self):
        return len(self.items)

    def build_prompt(self, task_type: str, question: str) -> str:
        if self.add_task_prefix and task_type:
            return (
                "You are a remote sensing VQA assistant.\n"
                f"Task: {task_type}\n"
                f"Image: {self.image_token}\n"
                f"Question: {question}\n"
                "Answer: "
            )
        return (
            "You are a remote sensing VQA assistant.\n"
            f"Image: {self.image_token}\n"
            f"Question: {question}\n"
            "Answer: "
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_name, task_type, question, answer = self.items[idx][:4]
        img_name = _norm_img_name(img_name)

        img_path = os.path.join(self.rgb_dir, img_name)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"RGB image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        pixel_values = self.preprocess(img)

        task_type = str(task_type).strip()
        question = str(question).strip()
        answer = str(answer).strip()

        prompt = self.build_prompt(task_type, question)
        full_text = prompt + answer

        enc_full = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=self.max_length)
        enc_prompt = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)

        input_ids = enc_full["input_ids"][0]
        attention_mask = enc_full["attention_mask"][0]
        labels = input_ids.clone()

        prompt_len = enc_prompt["input_ids"].shape[1]
        labels[:prompt_len] = -100

        out = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if self.return_meta:
            out["meta"] = {
                "img_name": img_name,
                "task_type": task_type,
                "question": question,
                "answer": answer,
            }
        return out


@dataclass
class Stage2Collator:
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

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }

        # eval 才会带 meta
        if "meta" in batch[0]:
            out["meta"] = [x["meta"] for x in batch]
        return out
