import json
import os
import random
from typing import Any

from src.datasets.base import BaseRemoteQwenDataset


STAGE1_PROMPT = "Describe the remote sensing image: <image>\nCaption: "
AID_TEMPLATES = [
    "A remote sensing image of {cls}.",
    "This aerial scene shows {cls}.",
    "The scene type is {cls}.",
    "An overhead view of {cls}.",
    "{cls} in a remote sensing image.",
]


def _normalize_name(filename: str) -> str:
    return os.path.basename(filename).strip().lower()


def load_rsicd_items(ann_path: str, images_dir: str) -> list[tuple[str, list[str]]]:
    """Load RSICD as image-to-captions records."""
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_paths = sorted(
        os.path.join(images_dir, name)
        for name in os.listdir(images_dir)
        if name.lower().endswith(extensions)
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    with open(ann_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict) or "images" not in payload:
        raise ValueError(f"Unexpected RSICD annotation format: {ann_path}")

    caption_map: dict[str, list[str]] = {}
    for record in payload["images"]:
        filename = record.get("filename") or record.get("file_name") or record.get("name")
        if not filename:
            continue

        captions = []
        for sentence in record.get("sentences") or record.get("captions") or []:
            if isinstance(sentence, dict):
                text = sentence.get("raw") or sentence.get("caption") or sentence.get("sent") or ""
            else:
                text = str(sentence)
            text = text.strip()
            if text:
                captions.append(text)

        if captions:
            caption_map.setdefault(_normalize_name(filename), []).extend(captions)

    return [(path, caption_map.get(_normalize_name(path), [])) for path in image_paths]


def load_aid_pairs(aid_root: str, seed: int, templates: list[str]) -> list[tuple[str, str]]:
    """Generate weak captions for AID from class folders."""
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []

    classes = sorted(
        class_name
        for class_name in os.listdir(aid_root)
        if os.path.isdir(os.path.join(aid_root, class_name))
    )

    for class_name in classes:
        class_dir = os.path.join(aid_root, class_name)
        image_names = sorted(name for name in os.listdir(class_dir) if name.lower().endswith(extensions))
        for image_name in image_names:
            image_path = os.path.join(class_dir, image_name)
            class_text = class_name.replace("_", " ").strip()
            caption = rng.choice(templates).format(cls=class_text)
            pairs.append((image_path, caption))

    if not pairs:
        raise RuntimeError(f"No AID images found in {aid_root}")

    return pairs


class Stage1MixedDataset(BaseRemoteQwenDataset):
    """Stage1 training dataset that mixes RSICD captions and optional AID weak labels."""

    def __init__(
        self,
        rsicd_ann_path: str,
        rsicd_images_dir: str,
        preprocess,
        tokenizer,
        aid_root: str | None = None,
        include_rsicd: bool = True,
        include_aid: bool = True,
        image_token: str = "<image>",
        max_length: int = 256,
        seed: int = 42,
        aid_templates: list[str] | None = None,
    ):
        super().__init__(preprocess=preprocess, tokenizer=tokenizer, image_token=image_token, max_length=max_length)
        self.prompt = STAGE1_PROMPT.replace("<image>", image_token)
        self.items: list[dict[str, str]] = []

        if include_rsicd:
            skipped = 0
            rsicd_pairs = 0
            for image_path, captions in load_rsicd_items(rsicd_ann_path, rsicd_images_dir):
                if not captions:
                    skipped += 1
                    continue
                for caption in captions:
                    self.items.append({"image_path": image_path, "caption": caption, "source": "rsicd"})
                    rsicd_pairs += 1
            print(f"[Stage1MixedDataset] RSICD pairs={rsicd_pairs} skipped_no_caption={skipped}")

        if include_aid and aid_root:
            aid_pairs = load_aid_pairs(aid_root, seed=seed, templates=aid_templates or AID_TEMPLATES)
            aid_items = [
                {"image_path": image_path, "caption": caption, "source": "aid"}
                for image_path, caption in aid_pairs
            ]
            self.items.extend(aid_items)
            print(f"[Stage1MixedDataset] AID pairs={len(aid_items)}")

        if not self.items:
            raise RuntimeError("Stage1MixedDataset is empty. Check RSICD/AID paths and include flags.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        return self.encode_causal_example(
            image_path=item["image_path"],
            prompt=self.prompt,
            target_text=item["caption"],
        )
