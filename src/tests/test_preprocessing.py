# src/tests/test_preprocessing.py

import pandas as pd
import numpy as np
import pytest
from src.data.preprocessing import fix_dtypes, drop_columns, encode_binary_columns, create_features, encode_categorical_columns, transform
from src.data.features import EXPECTED_COLS, BINARY_COLS, SERVICE_COLS


# ── Fixtures ──────────────────────────────────────────────
@pytest.fixture
def raw_sample():
    """Minimal raw dataframe mimicking the original CSV structure."""
    return pd.DataFrame([{
        "customerID": "1234-ABCD",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": "844.2",
        "Churn": "Yes"
    }])


@pytest.fixture
def clean_sample(raw_sample):
    """Sample after fix_dtypes and drop_columns — ready for transform."""
    df = fix_dtypes(raw_sample)
    df = drop_columns(df)
    return df


# ── fix_dtypes ────────────────────────────────────────────
def test_fix_dtypes_total_charges_is_float(raw_sample):
    df = fix_dtypes(raw_sample)
    assert df["TotalCharges"].dtype == float


def test_fix_dtypes_churn_is_binary(raw_sample):
    df = fix_dtypes(raw_sample)
    assert df["Churn"].isin([0, 1]).all()


def test_fix_dtypes_handles_whitespace_total_charges():
    df = pd.DataFrame([{"TotalCharges": " ", "Churn": "No"}])
    df = fix_dtypes(df)
    assert df["TotalCharges"].iloc[0] == 0.0


# ── drop_columns ──────────────────────────────────────────
def test_drop_columns_removes_customer_id(raw_sample):
    df = fix_dtypes(raw_sample)
    df = drop_columns(df)
    assert "customerID" not in df.columns


def test_drop_columns_removes_gender(raw_sample):
    df = fix_dtypes(raw_sample)
    df = drop_columns(df)
    assert "gender" not in df.columns


def test_drop_columns_removes_total_charges(raw_sample):
    df = fix_dtypes(raw_sample)
    df = drop_columns(df)
    assert "TotalCharges" not in df.columns


# ── encode_binary_columns ─────────────────────────────────
def test_encode_binary_columns_are_numeric(clean_sample):
    df = encode_binary_columns(clean_sample)
    for col in BINARY_COLS:
        assert df[col].dtype in [np.int64, np.float64, int]


def test_encode_binary_columns_no_nulls(clean_sample):
    df = encode_binary_columns(clean_sample)
    for col in BINARY_COLS:
        assert df[col].isnull().sum() == 0


# ── create_features ───────────────────────────────────────
def test_n_services_created(clean_sample):
    df = encode_binary_columns(clean_sample)
    df = create_features(df)
    assert "n_services" in df.columns


def test_n_services_range(clean_sample):
    df = encode_binary_columns(clean_sample)
    df = create_features(df)
    assert df["n_services"].between(0, len(SERVICE_COLS)).all()


# ── transform ─────────────────────────────────────────────
def test_transform_output_columns(clean_sample):
    df = transform(clean_sample)
    assert list(df.columns) == EXPECTED_COLS


def test_transform_no_nulls(clean_sample):
    df = transform(clean_sample)
    assert df.isnull().sum().sum() == 0


def test_transform_output_shape(clean_sample):
    df = transform(clean_sample)
    assert df.shape[1] == len(EXPECTED_COLS)


def test_transform_binary_cols_are_numeric(clean_sample):
    df = transform(clean_sample)
    for col in BINARY_COLS:
        if col in df.columns:
            assert df[col].dtype in [np.int64, np.float64, int]