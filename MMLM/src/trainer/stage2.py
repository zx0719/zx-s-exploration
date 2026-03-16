from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import Trainer, TrainingArguments

from src.trainer.callbacks import ExperimentLoggerCallback
from src.utils.common import append_jsonl, ensure_dir
from src.utils.text import extract_answer_text, normalize_answer


class Stage2HFTrainer(Trainer):
    """Custom Trainer that periodically logs generated samples."""

    def __init__(self, *args, sample_log_every: int = 50, sample_max_new_tokens: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_log_every = sample_log_every
        self.sample_max_new_tokens = sample_max_new_tokens

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            pixel_values=inputs["pixel_values"],
        )
        loss = outputs.loss

        try:
            step = int(self.state.global_step)
            should_log_sample = self.sample_log_every > 0 and step % self.sample_log_every == 0
            if should_log_sample:
                with torch.no_grad():
                    ground_truth_ids = inputs["labels"][0][inputs["labels"][0] != -100]
                    ground_truth = model.tokenizer.decode(ground_truth_ids, skip_special_tokens=True).strip()
                    prediction = model.generate_from_batch(
                        input_ids=inputs["input_ids"][:1],
                        attention_mask=inputs["attention_mask"][:1],
                        pixel_values=inputs["pixel_values"][:1],
                        max_new_tokens=self.sample_max_new_tokens,
                    )
                    sample = {
                        "step": step,
                        "gt": ground_truth[:200],
                        "pred_full": prediction[:800],
                        "gt_norm": normalize_answer(ground_truth),
                        "pred_norm": normalize_answer(extract_answer_text(prediction)),
                    }
                    model._last_sample = sample
                    append_jsonl(Path(self.args.output_dir) / "train_samples.jsonl", sample)
        except Exception:
            pass

        return (loss, outputs) if return_outputs else loss


class Stage2TrainerRunner:
    """Wrap Hugging Face Trainer for stage2 projector plus LoRA training."""

    def __init__(
        self,
        training_args: dict,
        save_connector_name: str = "connector.pt",
        save_lora_subdir: str = "lora",
        sample_log_every: int = 20,
        sample_max_new_tokens: int = 16,
    ):
        self.training_args = training_args
        self.save_connector_name = save_connector_name
        self.save_lora_subdir = save_lora_subdir
        self.sample_log_every = sample_log_every
        self.sample_max_new_tokens = sample_max_new_tokens

    def fit(self, model, train_dataset, collator, logger) -> None:
        args = TrainingArguments(**OmegaConf.to_container(self.training_args, resolve=True))
        callback = ExperimentLoggerCallback(
            logger=logger,
            model_ref=model,
            include_sample_correctness=True,
        )
        trainer = Stage2HFTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            data_collator=collator,
            callbacks=[callback],
            sample_log_every=self.sample_log_every,
            sample_max_new_tokens=self.sample_max_new_tokens,
        )
        trainer.train()

        output_dir = ensure_dir(args.output_dir)
        connector_path = Path(output_dir) / self.save_connector_name
        model.save_projector(str(connector_path))
        model.save_lora(str(Path(output_dir) / self.save_lora_subdir))
        model.tokenizer.save_pretrained(str(output_dir))
        print(f"[Stage2TrainerRunner] saved connector to {connector_path}")
