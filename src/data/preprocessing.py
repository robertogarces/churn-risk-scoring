# src/data/preprocessing.py

import logging
import pandas as pd
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.data.features import (
    DROP_COLS,
    BINARY_COLS,
    BINARY_MAP,
    CATEGORICAL_COLS,
    SERVICE_COLS,
    EXPECTED_COLS
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix data type issues detected in EDA:
    - TotalCharges: whitespace strings masking nulls -> float
    - Churn: Yes/No strings -> binary int
    """
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    logger.info("Fixed dtypes: TotalCharges -> float, Churn -> binary int")
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns decided in EDA:
    - customerID: identifier, no predictive value
    - gender: no signal, ethical concerns in scoring models
    - TotalCharges: 0.83 correlation with tenure (multicollinearity)
    """
    df = df.drop(columns=DROP_COLS)
    logger.info(f"Dropped columns: {DROP_COLS}")
    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode Yes/No binary columns to 1/0.
    SeniorCitizen is already int.
    """
    df = df.copy()
    for col in BINARY_COLS:
        df[col] = df[col].map(BINARY_MAP)
    logger.info(f"Encoded {len(BINARY_COLS)} binary columns to 0/1")
    return df


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode multiclass categorical columns.
    drop_first=True to avoid dummy variable trap.
    """
    df = df.copy()
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    new_cols = [c for c in df.columns if any(c.startswith(cat) for cat in CATEGORICAL_COLS)]
    logger.info(f"One-hot encoded {CATEGORICAL_COLS} -> {len(new_cols)} new columns")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering decisions from EDA:
    - n_services: total number of contracted services.
      Customers with more services have higher switching friction.
      Explicit feature improves SHAP interpretability vs letting
      LightGBM discover the interaction across 8 binary columns.
    """
    df = df.copy()
    df['n_services'] = df[SERVICE_COLS].sum(axis=1)
    logger.info(f"Created feature n_services from {len(SERVICE_COLS)} service columns")
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full transformation pipeline to a dataframe.
    Used by both the batch pipeline and the API for inference.
    Does not include fix_dtypes or drop_columns — those are
    only needed for the raw training data, not for inference input.
    """
    df = encode_binary_columns(df)
    df = create_features(df)
    df = encode_categorical_columns(df)

    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = 0

    return df[EXPECTED_COLS]


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    raw_path = Path(cfg.paths.raw_data)
    output_path = Path(cfg.paths.processed_data)

    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Raw dataset loaded — shape: {df.shape}")

    df = fix_dtypes(df)
    df = drop_columns(df)
    df = transform(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    logger.info(f"Processed dataset saved to {output_path}")
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()