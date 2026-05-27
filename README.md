# churn-risk-scoring

> End-to-end churn risk scoring pipeline simulated for an Australian telecommunications operator.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)
![DVC](https://img.shields.io/badge/Data-DVC-945DD6)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2)

---

## Overview

Customer churn is one of the highest-impact problems in the telecommunications industry. This project builds a production-grade churn risk scoring system — from raw data to a live REST API and business dashboard — framed as a real solution for an Australian telco operator (Telstra/Optus market context).

The system assigns a churn probability score to every customer, segments them by risk level, and explains each prediction using SHAP values. At an ARPU of AUD $64.76/month, the model identifies over $149,000 in monthly recurring revenue at risk.

This is not a Kaggle notebook. Every component is productionized: reproducible pipeline, versioned data, tracked experiments, containerized services, and tested API endpoints.

---

## Dashboard

![dashboard](images/dashboard-light.png)

---

## API

![api](images/api.png)

---

## Business Context

| Metric | Value |
|---|---|
| Dataset | IBM Telco Customer Churn — 7,043 customers |
| Churn Rate | 26.5% |
| ARPU | AUD $64.76/month |
| Monthly Revenue at Risk | ~AUD $149,984 |
| CLV at Risk | ~AUD $3,599,620 |
| Cost to Retain All High-Risk | AUD $34,740 |
| ROI (50% retention success) | 2.2x |

---

## Model Performance

Three models were evaluated. LightGBM was selected as the production model after Optuna hyperparameter tuning.

| Model | ROC-AUC | PR-AUC | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.839 | 0.630 | 0.778 | 0.617 |
| Random Forest | 0.844 | 0.641 | 0.781 | 0.625 |
| **LightGBM (tuned)** | **0.847** | **0.668** | **0.808** | **0.623** |

Recall is the primary business metric — missing a churner (false negative) costs AUD $1,554 in lost CLV, while a false positive costs AUD $15 in unnecessary retention spend.

---

## SHAP — Global Feature Importance

<!-- ADD SCREENSHOT: shap summary bar plot -->

Top predictors identified:

- **Contract type** — Month-to-month customers churn at 42.7% vs 2.8% for two-year contracts
- **Tenure** — Strong negative correlation (-0.35) with churn. Risk concentrates in first 5 months
- **Internet Service** — Fiber optic customers churn at 41.9% despite being the premium tier
- **Payment Method** — Electronic check: 45.3% churn vs ~15% for automatic payment methods
- **Online Security / Tech Support** — Absence of these services nearly triples churn rate

---

## Architecture

```
Raw Data (Kaggle)
      │
      ▼
make_dataset.py          ← Downloads dataset automatically via kagglehub
      │
      ▼
preprocessing.py         ← Cleaning, encoding, feature engineering
      │
      ▼
tune.py                  ← Optuna hyperparameter optimization (50 trials, 5-fold CV)
      │
      ▼
train.py                 ← LightGBM training + MLflow experiment tracking
      │
      ▼
evaluate.py              ← SHAP global explanations + artifacts
      │
      ▼
predict.py               ← Batch scoring → churn_scores.csv
      │
      ▼
api/main.py              ← FastAPI scoring endpoint + SHAP local explanations
      │
      ▼
dashboard/app.py         ← Streamlit business dashboard
```

---

## Project Structure

```
churn-risk-scoring/
│
├── data/
│   ├── raw/                  # Raw dataset (DVC tracked)
│   ├── processed/            # Engineered features (DVC tracked)
│   └── outputs/              # Batch scores
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   └── 02_business_analysis.ipynb  # Revenue impact analysis
│
├── src/
│   ├── data/
│   │   ├── features.py       # Centralized feature definitions
│   │   ├── make_dataset.py   # Kaggle data download
│   │   └── preprocessing.py  # Feature engineering pipeline
│   └── models/
│       ├── train.py          # Model training + MLflow tracking
│       ├── tune.py           # Optuna hyperparameter tuning
│       ├── evaluate.py       # SHAP evaluation
│       └── predict.py        # Batch scoring
│
├── api/
│   ├── main.py               # FastAPI app
│   ├── schemas.py            # Pydantic request/response models
│   └── tests/
│       └── test_api.py       # 11 endpoint tests
│
├── dashboard/
│   └── app.py                # Streamlit dashboard
│
├── configs/
│   ├── config.yaml           # Main Hydra config
│   ├── dataset/
│   │   └── kaggle.yaml       # Dataset source config
│   └── model/
│       └── lightgbm.yaml     # Model hyperparameters (auto-updated by tune.py)
│
├── models/                   # Trained model artifacts (DVC tracked)
├── artifacts/                # SHAP plots and values
├── dvc.yaml                  # Reproducible pipeline definition
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

---

## Quickstart

### Option A — Docker (recommended)

```bash
git clone https://github.com/robertogarces/churn-risk-scoring.git
cd churn-risk-scoring

docker compose up --build
```

- Dashboard: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

### Option B — Local

**1. Create environment**
```bash
conda create -n churn-risk-scoring python=3.11 -y
conda activate churn-risk-scoring
pip install -r requirements.txt
pip install -e .
```

**2. Run pipeline**
```bash
dvc repro
```

**3. Run API**
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**4. Run dashboard**
```bash
streamlit run dashboard/app.py
```

> **Note:** Kaggle credentials required for data download. Configure via `kaggle.json` or environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.

---

## API Usage

**Endpoint:** `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35
  }'
```

**Response:**
```json
{
  "churn_probability": 0.8399,
  "risk_segment": "high",
  "top_factors": [
    {"feature": "Tenure (months)", "value": "2", "impact": "increases risk"},
    {"feature": "Contract Type", "value": "Month-to-month", "impact": "increases risk"},
    {"feature": "Internet Service", "value": "Fiber optic", "impact": "increases risk"}
  ]
}
```

---

## Reproducing the Pipeline

```bash
# Download data
python src/data/make_dataset.py

# Full pipeline (data → model → evaluation)
dvc repro

# Hyperparameter tuning (updates configs/model/lightgbm.yaml automatically)
python src/models/tune.py

# Batch scoring
python src/models/predict.py

# Run tests
pytest api/tests/test_api.py -v
```

---

## Experiment Tracking

MLflow tracks all experiments locally. To view:

```bash
mlflow ui
```

Open `http://localhost:5000` to compare runs across Logistic Regression, Random Forest, and LightGBM.

---

## Tech Stack

| Category | Tools |
|---|---|
| Model | LightGBM, Scikit-learn |
| Explainability | SHAP |
| Tuning | Optuna |
| Experiment Tracking | MLflow |
| Pipeline | DVC |
| Configuration | Hydra |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Containerization | Docker, Docker Compose |
| Testing | Pytest |

---

## Data

**Source:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**Records:** 7,043 customers, 21 variables  
**Target:** `Churn` — whether a customer left in the last month  

> Dataset is downloaded automatically via `kagglehub` when running the pipeline. No manual download required.

---

## Author

**Roberto Garcés** — Data Scientist  
[github.com/robertogarces](https://github.com/robertogarces) · [LinkedIn](#)
