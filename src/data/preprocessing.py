# src/data/preprocessing.py

import logging
import pandas as pd
from pathlib import Path

import hydra
from omegaconf import DictConfig

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
    cols_to_drop = ['customerID', 'gender', 'TotalCharges']
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped columns: {cols_to_drop}")
    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode Yes/No binary columns to 1/0.
    SeniorCitizen is already int.
    """
    df = df.copy()
    binary_cols = [
        'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines'
    ]
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0})
    logger.info(f"Encoded {len(binary_cols)} binary columns to 0/1")
    return df


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode multiclass categorical columns.
    drop_first=True to avoid dummy variable trap.
    """
    df = df.copy()
    cat_cols = ['Contract', 'PaymentMethod', 'InternetService']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    new_cols = [c for c in df.columns if any(c.startswith(cat) for cat in cat_cols)]
    logger.info(f"One-hot encoded {cat_cols} -> {len(new_cols)} new columns")
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
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies'
    ]
    df['n_services'] = df[service_cols].sum(axis=1)
    logger.info(f"Created feature n_services from {len(service_cols)} service columns")
    return df


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    raw_path = Path(cfg.paths.raw_data)
    output_path = Path(cfg.paths.processed_data)

    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Raw dataset loaded — shape: {df.shape}")

    df = fix_dtypes(df)
    df = drop_columns(df)
    df = encode_binary_columns(df)
    df = create_features(df)
    df = encode_categorical_columns(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    logger.info(f"Processed dataset saved to {output_path}")
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Columns: {', '.join(list(df.columns))}")


if __name__ == "__main__":
    main()