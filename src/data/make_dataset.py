# src/data/make_dataset.py

import kagglehub
import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Download raw dataset based on config.
    Currently supports source: kaggle.
    Add new dataset configs under configs/dataset/ for other sources.
    """
    output_path = Path(cfg.paths.raw_data)

    if output_path.exists():
        print(f"Dataset already exists at {output_path}. Skipping download.")
        return

    if cfg.dataset.source == "kaggle":
        print(f"Downloading from Kaggle: {cfg.dataset.slug}")
        kaggle_path = kagglehub.dataset_download(cfg.dataset.slug)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(kaggle_path) / cfg.dataset.filename, output_path)
        print(f"Dataset saved to {output_path}")
    else:
        raise ValueError(f"Unsupported dataset source: {cfg.dataset.source}")


if __name__ == "__main__":
    main()