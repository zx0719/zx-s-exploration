import os
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


class BaseRemoteQwenDataset(Dataset):
    """Base dataset that builds dict batches for Remote-Clip + Qwen training."""

    def __init__(self, preprocess, tokenizer, image_token: str = "<image>", max_length: int = 512):
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.max_length = max_length

    def load_image(self, image_path: str):
        """Load an RGB image from disk."""
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        return Image.open(image_path).convert("RGB")

    def encode_causal_example(self, image_path: str, prompt: str, target_text: str) -> dict[str, Any]:
        """Encode an image-text sample into the shared dict-batch contract."""
        image = self.load_image(image_path)
        pixel_values = self.preprocess(image)

        full_text = prompt + target_text
        full_enc = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        prompt_enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = full_enc["input_ids"][0]
        attention_mask = full_enc["attention_mask"][0]
        labels = input_ids.clone()

        prompt_len = prompt_enc["input_ids"].shape[1]
        labels[:prompt_len] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
