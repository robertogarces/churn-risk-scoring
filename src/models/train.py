# src/models/train.py

import logging
import pickle
from pathlib import Path

import hydra
import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    logger.info(f"Loading processed data from {path}")
    df = pd.read_parquet(path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    logger.info(f"Dataset loaded — shape: {df.shape} | Churn rate: {y.mean():.1%}")
    return X, y


def get_model(cfg: DictConfig):
    """
    Instantiate model based on config.
    Add new models here when new configs/model/*.yaml are added.
    """
    name = cfg.model.name
    params = dict(cfg.model.params)

    if name == "logistic_regression":
        return LogisticRegression(**params)
    elif name == "random_forest":
        return RandomForestClassifier(**params)
    elif name == "lightgbm":
        return LGBMClassifier(**params)
    else:
        raise ValueError(f"Unsupported model: {name}")


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")

    return metrics


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # ── Load data ────────────────────────────────────────
    X, y = load_data(cfg.paths.processed_data)

    # ── Split ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
        stratify=y
    )
    logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # ── MLflow ───────────────────────────────────────────
    mlflow.set_experiment(cfg.model.name)

    with mlflow.start_run(run_name=cfg.model.name):
        # Log params
        mlflow.log_params(dict(cfg.model.params))

        # Train
        logger.info(f"Training {cfg.model.name}...")
        model = get_model(cfg)
        model.fit(X_train, y_train)
        logger.info("Training complete")

        # Evaluate
        logger.info("Evaluating on test set...")
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # Save model
        model_path = Path(cfg.paths.models) / f"{cfg.model.name}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.sklearn.log_model(model, artifact_path=cfg.model.name)
        logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()