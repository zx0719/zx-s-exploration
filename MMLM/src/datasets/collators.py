from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CausalLMCollator:
    """Pad dict-based causal language modeling batches."""

    pad_token_id: int

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(item["input_ids"].shape[0] for item in batch)

        def pad_1d(tensor: torch.Tensor, pad_value: int) -> torch.Tensor:
            pad_len = max_len - tensor.shape[0]
            if pad_len <= 0:
                return tensor
            return torch.cat(
                [tensor, torch.full((pad_len,), pad_value, dtype=tensor.dtype)],
                dim=0,
            )

        output = {
            "input_ids": torch.stack([pad_1d(item["input_ids"], self.pad_token_id) for item in batch], dim=0),
            "attention_mask": torch.stack([pad_1d(item["attention_mask"], 0) for item in batch], dim=0),
            "labels": torch.stack([pad_1d(item["labels"], -100) for item in batch], dim=0),
            "pixel_values": torch.stack([item["pixel_values"] for item in batch], dim=0),
        }

        if "meta" in batch[0]:
            output["meta"] = [item["meta"] for item in batch]

        return output
