# src/models/tune.py

import logging
import pickle
from pathlib import Path
import yaml
from omegaconf import OmegaConf

import hydra
import mlflow
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold, cross_val_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Silence Optuna's info logs to keep output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    logger.info(f"Loading processed data from {path}")
    df = pd.read_parquet(path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    logger.info(f"Dataset loaded — shape: {df.shape} | Churn rate: {y.mean():.1%}")
    return X, y


def objective(trial, X: pd.DataFrame, y: pd.Series) -> float:
    """Optuna objective function — maximizes ROC-AUC via 5-fold CV."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": -1,
    }

    model = LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    X, y = load_data(cfg.paths.processed_data)

    logger.info(f"Starting hyperparameter optimization with Optuna ({cfg.tuning.n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=cfg.tuning.n_trials)

    best_params = study.best_params
    best_score = study.best_value
    logger.info(f"Best ROC-AUC (CV): {best_score:.4f}")

    # ── Log en MLflow ─────────────────────────────────────
    mlflow.set_experiment("lightgbm-tuning")
    with mlflow.start_run(run_name="optuna-best"):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_roc_auc", best_score)

    # ── Update lightgbm.yaml ──────────────────────────
    config_path = Path("configs/model/lightgbm.yaml")
    updated_config = {
        "name": "lightgbm",
        "params": {
            "n_estimators": best_params["n_estimators"],
            "learning_rate": round(best_params["learning_rate"], 6),
            "max_depth": best_params["max_depth"],
            "num_leaves": best_params["num_leaves"],
            "min_child_samples": best_params["min_child_samples"],
            "subsample": round(best_params["subsample"], 6),
            "colsample_bytree": round(best_params["colsample_bytree"], 6),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
    }

    with open(config_path, "w") as f:
        yaml.dump(updated_config, f, default_flow_style=False, sort_keys=False)

    logger.info("configs/model/lightgbm.yaml updated with best params")


if __name__ == "__main__":
    main()