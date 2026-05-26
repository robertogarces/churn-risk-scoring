# api/main.py

import logging
import pickle
from pathlib import Path

import shap
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import CustomerProfile, ChurnPrediction, ChurnFactor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="Churn Risk Scoring API",
    description="End-to-end churn prediction pipeline for an Australian telco operator.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model and config on startup ─────────────────────
MODEL_PATH = Path("models/lightgbm.pkl")
CONFIG_PATH = Path("configs/model/lightgbm.yaml")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(CONFIG_PATH, "r") as f:
    model_cfg = yaml.safe_load(f)

explainer = shap.TreeExplainer(model)
logger.info("Model and explainer loaded successfully")


# ── Preprocessing ─────────────────────────────────────────
def preprocess_input(profile: CustomerProfile) -> pd.DataFrame:
    """
    Apply the same preprocessing as preprocessing.py to a single customer profile.
    Input is raw — same fields as the original CSV, no encoding needed from the caller.
    """
    data = profile.model_dump()
    df = pd.DataFrame([data])

    # Binary encoding
    binary_map = {'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0}
    binary_cols = [
        'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines'
    ]
    for col in binary_cols:
        df[col] = df[col].map(binary_map)

    # n_services feature
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies'
    ]
    df['n_services'] = df[service_cols].sum(axis=1)

    # One-hot encoding
    df = pd.get_dummies(df, columns=['Contract', 'PaymentMethod', 'InternetService'])

    # Ensure all expected columns are present with correct order
    expected_cols = [
        'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
        'MonthlyCharges', 'n_services', 'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check', 'InternetService_Fiber optic', 'InternetService_No'
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    return df[expected_cols]


# ── Risk segment ──────────────────────────────────────────
def get_risk_segment(probability: float) -> str:
    if probability >= 0.6:
        return "high"
    elif probability >= 0.3:
        return "medium"
    else:
        return "low"


# ── SHAP local explanation ────────────────────────────────
def get_top_factors(df: pd.DataFrame, profile: CustomerProfile, n: int = 3) -> list[ChurnFactor]:
    """
    Compute SHAP values for a single customer and return top n factors
    with human-readable feature names and values.
    """
    shap_values = explainer(df)
    shap_array = shap_values.values[0]

    # Map encoded column names back to raw feature values
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

    # Add binary features
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']:
        feature_value_map[col] = "Yes" if raw_values.get(col) == "Yes" else "No"
        feature_name_map[col] = col

    # Sort by absolute SHAP value
    feature_names = df.columns.tolist()
    top_indices = np.argsort(np.abs(shap_array))[::-1][:n]

    factors = []
    seen_features = set()
    for idx in top_indices:
        fname = feature_names[idx]
        display_name = feature_name_map.get(fname, fname)

        # Avoid duplicate feature names (e.g. Contract appears in two columns)
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
    return {"status": "ok", "model": model_cfg["name"]}


@app.post("/predict", response_model=ChurnPrediction)
def predict(profile: CustomerProfile):
    logger.info("Received prediction request")

    df = preprocess_input(profile)
    churn_probability = float(model.predict_proba(df)[:, 1][0])
    risk_segment = get_risk_segment(churn_probability)
    top_factors = get_top_factors(df, profile)

    logger.info(f"Churn probability: {churn_probability:.3f} | Segment: {risk_segment}")

    return ChurnPrediction(
        churn_probability=round(churn_probability, 4),
        risk_segment=risk_segment,
        top_factors=top_factors
    )