# src/models/evaluate.py

import logging
import pickle
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

import json
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model(path: str):
    logger.info(f"Loading model from {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_data(path: str, test_size: float, random_state: int):
    logger.info(f"Loading processed data from {path}")
    df = pd.read_parquet(path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    logger.info(f"Test set — shape: {X_test.shape} | Churn rate: {y_test.mean():.1%}")
    return X_test, y_test


def compute_shap_values(model, X: pd.DataFrame, artifacts_path: Path):
    logger.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    logger.info("SHAP values computed")

    # Save SHAP values
    shap_path = artifacts_path / "shap_values.npy"
    np.save(shap_path, shap_values.values)
    logger.info(f"SHAP values saved to {shap_path}")

    return shap_values


def plot_shap_summary(shap_values, X: pd.DataFrame, artifacts_path: Path):
    """Global feature importance — bar plot."""
    logger.info("Generating SHAP summary plot...")
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.bar(shap_values, max_display=15, show=False, ax=ax)
    ax.set_title("Global Feature Importance (SHAP)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = artifacts_path / "shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary plot saved to {path}")


def plot_shap_beeswarm(shap_values, artifacts_path: Path):
    """Beeswarm plot — shows distribution of SHAP values per feature."""
    logger.info("Generating SHAP beeswarm plot...")
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP Beeswarm Plot", fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = artifacts_path / "shap_beeswarm.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP beeswarm plot saved to {path}")


def save_metrics(model, X_test: pd.DataFrame, y_test: pd.Series, artifacts_path: Path) -> dict:
    """
    Evaluate model on test set and save metrics to artifacts/metrics.json.
    Tracked by DVC — makes model performance auditable across runs.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "test_size": len(y_test),
        "churn_rate_test": round(y_test.mean(), 4)
    }

    metrics_path = artifacts_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to {metrics_path}")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    return metrics


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    artifacts_path = Path(cfg.paths.artifacts)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────
    model = load_model(Path(cfg.paths.models) / "lightgbm.pkl")
    X_test, y_test = load_data(
        cfg.paths.processed_data,
        cfg.data.test_size,
        cfg.data.random_state
    )

    # ── SHAP ──────────────────────────────────────────────
    shap_values = compute_shap_values(model, X_test, artifacts_path)
    plot_shap_summary(shap_values, X_test, artifacts_path)
    plot_shap_beeswarm(shap_values, artifacts_path)

    # ── Metrics ──────────────────────────────────────────────
    save_metrics(model, X_test, y_test, artifacts_path)

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()