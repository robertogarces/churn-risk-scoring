# src/models/predict.py

import logging
import pickle
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

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


def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading processed data from {path}")
    df = pd.read_parquet(path)
    logger.info(f"Dataset loaded — shape: {df.shape}")
    return df


def score(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Score all customers and assign risk segments.
    Returns original dataframe with churn_probability and risk_segment columns.
    """
    X = df.drop(columns=["Churn"])
    y_true = df["Churn"]

    churn_proba = model.predict_proba(X)[:, 1]

    results = df.copy()
    results["churn_probability"] = churn_proba
    results["churn_true"] = y_true
    results["risk_segment"] = pd.cut(
        churn_proba,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["low", "medium", "high"]
    )

    return results


def log_summary(results: pd.DataFrame, business_cfg) -> None:
    """Log business impact summary after scoring."""
    arpu = business_cfg.arpu_aud
    retention_cost = business_cfg.retention_cost_aud
    avg_lifetime = business_cfg.avg_lifetime_months

    high_risk = results[results["risk_segment"] == "high"]
    n_high_risk = len(high_risk)
    revenue_at_risk = n_high_risk * arpu
    clv_at_risk = revenue_at_risk * avg_lifetime
    retention_cost_total = n_high_risk * retention_cost

    logger.info("=" * 55)
    logger.info("   BATCH SCORING SUMMARY")
    logger.info("=" * 55)
    logger.info(f"   Total customers scored:      {len(results):,}")
    logger.info(f"   High risk (>60%):            {n_high_risk:,}")
    logger.info(f"   Medium risk (30-60%):        {len(results[results['risk_segment'] == 'medium']):,}")
    logger.info(f"   Low risk (<30%):             {len(results[results['risk_segment'] == 'low']):,}")
    logger.info("-" * 55)
    logger.info(f"   Monthly revenue at risk:     ${revenue_at_risk:,.0f} AUD")
    logger.info(f"   CLV at risk:                 ${clv_at_risk:,.0f} AUD")
    logger.info(f"   Cost to retain high risk:    ${retention_cost_total:,.0f} AUD")
    logger.info("=" * 55)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # ── Load ──────────────────────────────────────────────
    model = load_model(Path(cfg.paths.models) / f"{cfg.model.name}.pkl")
    df = load_data(cfg.paths.processed_data)

    # ── Score ─────────────────────────────────────────────
    logger.info("Scoring all customers...")
    results = score(model, df)

    # ── Save ──────────────────────────────────────────────
    output_path = Path(cfg.paths.outputs) / "churn_scores.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info(f"Scores saved to {output_path}")

    # ── Business summary ──────────────────────────────────
    log_summary(results, cfg.business)


if __name__ == "__main__":
    main()