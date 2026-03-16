import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.utils.common import ensure_dir
from src.utils.text import normalize_answer


class SingleImageGenerationRunner:
    """Run single-image generation for a configured prompt."""

    def __init__(
        self,
        image_path: str,
        prompt_text: str,
        device: str = "cuda",
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.7,
        output_path: str | None = None,
    ):
        self.image_path = image_path
        self.prompt_text = prompt_text
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.output_path = output_path

    def run(self, model, dataset=None, collator=None) -> None:
        model.to(self.device)
        model.eval()

        image = Image.open(self.image_path).convert("RGB")
        pixel_values = model.preprocess(image).unsqueeze(0).to(self.device)
        prediction = model.generate_from_prompt(
            prompt_text=self.prompt_text,
            pixel_values=pixel_values,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
        )
        print(prediction)

        if self.output_path:
            output_file = Path(self.output_path)
            ensure_dir(output_file.parent)
            with output_file.open("w", encoding="utf-8") as handle:
                json.dump({"prediction": prediction}, handle, ensure_ascii=False, indent=2)


class OSVQAEvaluationRunner:
    """Evaluate exact-match accuracy on OSVQA with generative decoding."""

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 1,
        max_new_tokens: int = 16,
        report_path: str | None = None,
    ):
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.report_path = report_path

    def run(self, model, dataset, collator) -> None:
        if dataset is None:
            raise ValueError("OSVQAEvaluationRunner requires a dataset config.")
        if not getattr(dataset, "return_meta", False):
            raise ValueError("OSVQAEvaluationRunner expects dataset.return_meta=True.")

        model.to(self.device)
        model.eval()

        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collator,
        )

        stats = {
            "overall": {"n": 0, "correct": 0, "yesno_n": 0, "yesno_correct": 0},
            "by_task": {},
        }

        def update_stats(task_name: str, is_correct: int, gt_answer: str) -> None:
            task_stats = stats["by_task"].setdefault(
                task_name,
                {"n": 0, "correct": 0, "yesno_n": 0, "yesno_correct": 0},
            )
            stats["overall"]["n"] += 1
            stats["overall"]["correct"] += is_correct
            task_stats["n"] += 1
            task_stats["correct"] += is_correct

            if gt_answer in {"yes", "no"}:
                stats["overall"]["yesno_n"] += 1
                stats["overall"]["yesno_correct"] += is_correct
                task_stats["yesno_n"] += 1
                task_stats["yesno_correct"] += is_correct

        with torch.no_grad():
            for batch in data_loader:
                pixel_values = batch["pixel_values"].to(self.device)
                metas = batch["meta"]

                for index, meta in enumerate(metas):
                    prompt = dataset.build_prompt(meta["task_type"], meta["question"])
                    prediction = model.generate_from_prompt(
                        prompt_text=prompt,
                        pixel_values=pixel_values[index : index + 1],
                        max_new_tokens=self.max_new_tokens,
                    )
                    gt_norm = normalize_answer(meta["answer"])
                    pred_norm = normalize_answer(prediction)
                    update_stats(meta["task_type"], int(gt_norm == pred_norm), gt_norm)

        report = self.build_report(stats)
        print(json.dumps(report, ensure_ascii=False, indent=2))

        if self.report_path:
            report_file = Path(self.report_path)
            ensure_dir(report_file.parent)
            with report_file.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def build_report(stats: dict) -> dict:
        """Convert raw counters into accuracy metrics."""
        def accuracy(correct: int, total: int) -> float | None:
            return None if total == 0 else correct / total

        report = {
            "overall_acc": accuracy(stats["overall"]["correct"], stats["overall"]["n"]),
            "overall_n": stats["overall"]["n"],
            "overall_yesno_acc": accuracy(
                stats["overall"]["yesno_correct"],
                stats["overall"]["yesno_n"],
            ),
            "overall_yesno_n": stats["overall"]["yesno_n"],
            "by_task": {},
        }

        for task_name, task_stats in stats["by_task"].items():
            report["by_task"][task_name] = {
                "acc": accuracy(task_stats["correct"], task_stats["n"]),
                "n": task_stats["n"],
                "yesno_acc": accuracy(task_stats["yesno_correct"], task_stats["yesno_n"]),
                "yesno_n": task_stats["yesno_n"],
            }

        return report
