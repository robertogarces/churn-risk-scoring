# src/data/features.py

# ── Columns to drop ───────────────────────────────────────
DROP_COLS = ["customerID", "gender", "TotalCharges"]

# ── Binary columns ────────────────────────────────────────
BINARY_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines"
]

BINARY_MAP = {
    "Yes": 1,
    "No": 0,
    "No internet service": 0,
    "No phone service": 0
}

# ── Categorical columns (one-hot encoded) ─────────────────
CATEGORICAL_COLS = ["Contract", "PaymentMethod", "InternetService"]

# ── Service columns (used for n_services feature) ─────────
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies"
]

# ── Expected columns after preprocessing (model input) ────
EXPECTED_COLS = [
    "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
    "MonthlyCharges", "n_services", "Contract_One year", "Contract_Two year",
    "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check", "InternetService_Fiber optic", "InternetService_No"
]