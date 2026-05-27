# conftest.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ── Customer fixtures ─────────────────────────────────────
@pytest.fixture
def high_risk_customer():
    return {
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


@pytest.fixture
def low_risk_customer():
    return {
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


# ── Model mocks ───────────────────────────────────────────
def make_mock_explainer(shap_values_array):
    mock_shap_output = MagicMock()
    mock_shap_output.values = shap_values_array
    mock_explainer = MagicMock()
    mock_explainer.return_value = mock_shap_output
    return mock_explainer


@pytest.fixture(autouse=True)
def mock_model_high_risk():
    """Default mock — returns high risk score (0.84)."""
    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[0.16, 0.84]])

    shap_vals = np.array([[
        0.59, 0.0, 0.0, -0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.15, 0.08, 0.0, -0.25, 0.0, 0.19, 0.0,
        0.34, 0.0
    ]])

    with patch("api.main.model", mock), \
         patch("api.main.explainer", make_mock_explainer(shap_vals)):
        yield


@pytest.fixture
def mock_model_low_risk():
    """Mock that returns low risk score (0.12)."""
    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[0.88, 0.12]])

    shap_vals = np.array([[
        -0.59, 0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, -0.15, 0.08, 0.0, 0.25, 0.0, -0.19, 0.0,
        -0.34, 0.0
    ]])

    with patch("api.main.model", mock), \
         patch("api.main.explainer", make_mock_explainer(shap_vals)):
        yield