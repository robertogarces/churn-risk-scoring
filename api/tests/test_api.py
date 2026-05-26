# api/tests/test_api.py

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# ── Fixtures ──────────────────────────────────────────────
HIGH_RISK_CUSTOMER = {
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
}

LOW_RISK_CUSTOMER = {
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 45.0
}


# ── Health ────────────────────────────────────────────────
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model"] == "lightgbm"


# ── Predict ───────────────────────────────────────────────
def test_predict_returns_200():
    response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    assert response.status_code == 200


def test_predict_response_structure():
    response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    data = response.json()
    assert "churn_probability" in data
    assert "risk_segment" in data
    assert "top_factors" in data
    assert len(data["top_factors"]) == 3


def test_predict_probability_range():
    response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    prob = response.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_risk_segment_values():
    response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    segment = response.json()["risk_segment"]
    assert segment in ["low", "medium", "high"]


def test_high_risk_customer_scores_high():
    response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    data = response.json()
    assert data["churn_probability"] > 0.6
    assert data["risk_segment"] == "high"


def test_low_risk_customer_scores_low():
    response = client.post("/predict", json=LOW_RISK_CUSTOMER)
    data = response.json()
    assert data["churn_probability"] < 0.5


def test_top_factors_structure():
    response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
    factors = response.json()["top_factors"]
    for factor in factors:
        assert "feature" in factor
        assert "value" in factor
        assert "impact" in factor
        assert factor["impact"] in ["increases risk", "decreases risk"]


def test_invalid_input_returns_422():
    response = client.post("/predict", json={"tenure": "invalid"})
    assert response.status_code == 422


def test_negative_tenure_returns_422():
    invalid = HIGH_RISK_CUSTOMER.copy()
    invalid["tenure"] = -1
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_negative_monthly_charges_returns_422():
    invalid = HIGH_RISK_CUSTOMER.copy()
    invalid["MonthlyCharges"] = -50.0
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422