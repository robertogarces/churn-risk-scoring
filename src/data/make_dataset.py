# src/data/make_dataset.py

import logging
import zipfile
from pathlib import Path

import hydra
import kaggle
from omegaconf import DictConfig

import os
os.environ["KAGGLE_API_TOKEN"] = open(Path.home() / ".kaggle" / "access_token").read().strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Download raw dataset based on config.
    Currently supports source: kaggle.
    Add new dataset configs under configs/dataset/ for other sources.
    """
    output_path = Path(cfg.paths.raw_data)

    if output_path.exists():
        logger.info(f"Dataset already exists at {output_path}. Skipping download.")
        return

    if cfg.dataset.source == "kaggle":
        raw_data_dir = output_path.parent
        raw_data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading dataset from Kaggle: {cfg.dataset.slug}")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            cfg.dataset.slug,
            path=str(raw_data_dir),
            unzip=True
        )

        # Rename to our standard filename
        downloaded = list(raw_data_dir.glob("*.csv"))
        if not downloaded:
            raise FileNotFoundError("No CSV found after download")

        downloaded[0].rename(output_path)
        logger.info(f"Dataset saved to {output_path}")
    else:
        raise ValueError(f"Unsupported dataset source: {cfg.dataset.source}")


if __name__ == "__main__":
    main()