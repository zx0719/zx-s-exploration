from pathlib import Path

from omegaconf import OmegaConf
from transformers import Trainer, TrainingArguments

from src.trainer.callbacks import ExperimentLoggerCallback
from src.utils.common import ensure_dir


class Stage1TrainerRunner:
    """Wrap Hugging Face Trainer for stage1 projector alignment."""

    def __init__(self, training_args: dict, save_projector_name: str = "projector.pt"):
        self.training_args = training_args
        self.save_projector_name = save_projector_name

    def fit(self, model, train_dataset, collator, logger) -> None:
        args = TrainingArguments(**OmegaConf.to_container(self.training_args, resolve=True))
        callback = ExperimentLoggerCallback(logger=logger, model_ref=model)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            data_collator=collator,
            callbacks=[callback],
        )
        trainer.train()

        output_dir = ensure_dir(args.output_dir)
        projector_path = Path(output_dir) / self.save_projector_name
        model.save_projector(str(projector_path))
        model.tokenizer.save_pretrained(str(output_dir))
        print(f"[Stage1TrainerRunner] saved projector to {projector_path}")
