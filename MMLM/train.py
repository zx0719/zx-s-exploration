from pathlib import Path
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.common import seed_everything


@hydra.main(version_base=None, config_path="src/configs", config_name="train")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    model = instantiate(cfg.model)
    train_dataset = instantiate(
        cfg.datasets.train,
        tokenizer=model.tokenizer,
        preprocess=model.preprocess,
    )
    collator = instantiate(
        cfg.datasets.collator,
        pad_token_id=model.tokenizer.pad_token_id,
    )
    logger = instantiate(cfg.logger)
    trainer = instantiate(cfg.trainer)
    trainer.fit(
        model=model,
        train_dataset=train_dataset,
        collator=collator,
        logger=logger,
    )


if __name__ == "__main__":
    main()
