# api/main.py

import logging
import os
import pickle
from pathlib import Path
from contextlib import asynccontextmanager

import shap
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import CustomerProfile, ChurnPrediction, ChurnFactor
from src.data.preprocessing import transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Load config on startup ────────────────────────────────
CONFIG_PATH = Path("configs/config.yaml")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

MODEL_NAME = cfg["scoring"]["model_name"]
MODEL_PATH = Path(cfg["paths"]["models"]) / f"{MODEL_NAME}.pkl"
THRESHOLD_HIGH = cfg["scoring"]["thresholds"]["high"]
THRESHOLD_MEDIUM = cfg["scoring"]["thresholds"]["medium"]
N_TOP_FACTORS = cfg["scoring"]["top_factors"]

model = None
explainer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, explainer
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    logger.info("Model and explainer loaded successfully")
    yield


# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="Churn Risk Scoring API",
    description="End-to-end churn prediction pipeline for an Australian telco operator.",
    version="1.0.0",
    lifespan=lifespan
)

_cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Preprocessing ─────────────────────────────────────────
def preprocess_input(profile: CustomerProfile) -> pd.DataFrame:
    df = pd.DataFrame([profile.model_dump()])
    return transform(df)


# ── Risk segment ──────────────────────────────────────────
def get_risk_segment(probability: float) -> str:
    if probability >= THRESHOLD_HIGH:
        return "high"
    elif probability >= THRESHOLD_MEDIUM:
        return "medium"
    else:
        return "low"


# ── SHAP local explanation ────────────────────────────────
def get_top_factors(df: pd.DataFrame, profile: CustomerProfile, n: int = 3) -> list[ChurnFactor]:
    shap_values = explainer(df)
    shap_array = shap_values.values[0]

    raw_values = profile.model_dump()
    feature_value_map = {
        'tenure': str(raw_values['tenure']),
        'MonthlyCharges': f"${raw_values['MonthlyCharges']:.2f}",
        'SeniorCitizen': "Yes" if raw_values['SeniorCitizen'] == 1 else "No",
        'n_services': str(int(df['n_services'].values[0])),
        'Contract_One year': raw_values['Contract'],
        'Contract_Two year': raw_values['Contract'],
        'PaymentMethod_Credit card (automatic)': raw_values['PaymentMethod'],
        'PaymentMethod_Electronic check': raw_values['PaymentMethod'],
        'PaymentMethod_Mailed check': raw_values['PaymentMethod'],
        'InternetService_Fiber optic': raw_values['InternetService'],
        'InternetService_No': raw_values['InternetService'],
    }

    feature_name_map = {
        'tenure': 'Tenure (months)',
        'MonthlyCharges': 'Monthly Charges',
        'SeniorCitizen': 'Senior Citizen',
        'n_services': 'Number of Services',
        'Contract_One year': 'Contract Type',
        'Contract_Two year': 'Contract Type',
        'PaymentMethod_Credit card (automatic)': 'Payment Method',
        'PaymentMethod_Electronic check': 'Payment Method',
        'PaymentMethod_Mailed check': 'Payment Method',
        'InternetService_Fiber optic': 'Internet Service',
        'InternetService_No': 'Internet Service',
    }

    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']:
        feature_value_map[col] = "Yes" if raw_values.get(col) == "Yes" else "No"
        feature_name_map[col] = col

    feature_names = df.columns.tolist()
    top_indices = np.argsort(np.abs(shap_array))[::-1][:n]

    factors = []
    seen_features = set()
    for idx in top_indices:
        fname = feature_names[idx]
        display_name = feature_name_map.get(fname, fname)
        if display_name in seen_features:
            continue
        seen_features.add(display_name)
        impact = "increases risk" if shap_array[idx] > 0 else "decreases risk"
        value = feature_value_map.get(fname, str(df[fname].values[0]))
        factors.append(ChurnFactor(
            feature=display_name,
            value=value,
            impact=impact
        ))

    return factors


# ── Endpoints ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict", response_model=ChurnPrediction)
def predict(profile: CustomerProfile):
    if model is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    logger.info("Received prediction request")
    df = preprocess_input(profile)
    churn_probability = float(model.predict_proba(df)[:, 1][0])
    risk_segment = get_risk_segment(churn_probability)
    top_factors = get_top_factors(df, profile, n=N_TOP_FACTORS)
    logger.info(f"Churn probability: {churn_probability:.3f} | Segment: {risk_segment}")
    return ChurnPrediction(
        churn_probability=round(churn_probability, 4),
        risk_segment=risk_segment,
        top_factors=top_factors
    )