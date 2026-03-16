import json
import os
from typing import Any

from src.datasets.base import BaseRemoteQwenDataset


def load_osvqa_records(annotation_path: str) -> list[list[Any]]:
    """Load OSVQA annotations stored as list records."""
    with open(annotation_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {annotation_path}, got {type(payload)}")
    invalid = [index for index, item in enumerate(payload) if not (isinstance(item, list) and len(item) >= 4)]
    if invalid:
        raise ValueError(f"Invalid OSVQA records at indices: {invalid[:5]}")
    return payload


def normalize_image_name(image_name: str) -> str:
    """Ensure OSVQA image names include a file extension."""
    value = str(image_name).strip()
    if value.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        return value
    return value + ".png"


class OSVQARemoteVQADataset(BaseRemoteQwenDataset):
    """OSVQA dataset with dict-batch outputs for training and evaluation."""

    def __init__(
        self,
        ann_path: str,
        rgb_dir: str,
        preprocess,
        tokenizer,
        image_token: str = "<image>",
        max_length: int = 512,
        add_task_prefix: bool = True,
        return_meta: bool = False,
    ):
        super().__init__(preprocess=preprocess, tokenizer=tokenizer, image_token=image_token, max_length=max_length)
        self.rgb_dir = rgb_dir
        self.add_task_prefix = add_task_prefix
        self.return_meta = return_meta
        self.items = load_osvqa_records(ann_path)
        print(f"[OSVQARemoteVQADataset] ann={ann_path} items={len(self.items)}")

    def __len__(self) -> int:
        return len(self.items)

    def build_prompt(self, task_type: str, question: str) -> str:
        """Build the instruction prompt for OSVQA."""
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

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_name, task_type, question, answer = self.items[index][:4]
        image_name = normalize_image_name(image_name)
        image_path = os.path.join(self.rgb_dir, image_name)

        task_text = str(task_type).strip()
        question_text = str(question).strip()
        answer_text = str(answer).strip()

        sample = self.encode_causal_example(
            image_path=image_path,
            prompt=self.build_prompt(task_text, question_text),
            target_text=answer_text,
        )

        if self.return_meta:
            sample["meta"] = {
                "img_name": image_name,
                "task_type": task_text,
                "question": question_text,
                "answer": answer_text,
            }

        return sample
