import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed common RNGs for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    """Append one JSON record to a JSONL file."""
    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_optional_path(path: str | None) -> str | None:
    """Normalize empty strings to None for config-driven paths."""
    if path is None:
        return None
    value = str(path).strip()
    return value or None
