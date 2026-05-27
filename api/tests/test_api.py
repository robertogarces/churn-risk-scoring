# api/tests/test_api.py

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


# ── Health ────────────────────────────────────────────────
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model"] == "lightgbm"


# ── Predict ───────────────────────────────────────────────
def test_predict_returns_200(high_risk_customer):
    response = client.post("/predict", json=high_risk_customer)
    assert response.status_code == 200


def test_predict_response_structure(high_risk_customer):
    response = client.post("/predict", json=high_risk_customer)
    data = response.json()
    assert "churn_probability" in data
    assert "risk_segment" in data
    assert "top_factors" in data
    assert len(data["top_factors"]) == 3


def test_predict_probability_range(high_risk_customer):
    response = client.post("/predict", json=high_risk_customer)
    prob = response.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_risk_segment_values(high_risk_customer):
    response = client.post("/predict", json=high_risk_customer)
    segment = response.json()["risk_segment"]
    assert segment in ["low", "medium", "high"]


def test_high_risk_customer_scores_high(high_risk_customer):
    response = client.post("/predict", json=high_risk_customer)
    data = response.json()
    assert data["churn_probability"] > 0.6
    assert data["risk_segment"] == "high"


def test_low_risk_customer_scores_low(low_risk_customer, mock_model_low_risk):
    response = client.post("/predict", json=low_risk_customer)
    data = response.json()
    assert data["churn_probability"] < 0.5

def test_top_factors_structure(high_risk_customer):
    response = client.post("/predict", json=high_risk_customer)
    factors = response.json()["top_factors"]
    for factor in factors:
        assert "feature" in factor
        assert "value" in factor
        assert "impact" in factor
        assert factor["impact"] in ["increases risk", "decreases risk"]


def test_invalid_input_returns_422():
    response = client.post("/predict", json={"tenure": "invalid"})
    assert response.status_code == 422


def test_negative_tenure_returns_422(high_risk_customer):
    invalid = high_risk_customer.copy()
    invalid["tenure"] = -1
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_negative_monthly_charges_returns_422(high_risk_customer):
    invalid = high_risk_customer.copy()
    invalid["MonthlyCharges"] = -50.0
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422