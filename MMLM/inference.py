from pathlib import Path
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.common import seed_everything


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    model = instantiate(cfg.model)

    dataset = None
    if "datasets" in cfg and "dataset" in cfg.datasets:
        dataset = instantiate(
            cfg.datasets.dataset,
            tokenizer=model.tokenizer,
            preprocess=model.preprocess,
        )

    collator = None
    if "datasets" in cfg and "collator" in cfg.datasets:
        collator = instantiate(
            cfg.datasets.collator,
            pad_token_id=model.tokenizer.pad_token_id,
        )

    inferencer = instantiate(cfg.inferencer)
    inferencer.run(model=model, dataset=dataset, collator=collator)


if __name__ == "__main__":
    main()
