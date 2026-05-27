# dashboard/app.py

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import yaml
from pathlib import Path

# ── Config ────────────────────────────────────────────────
with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

ARPU = cfg["business"]["arpu_aud"]
AVG_LIFETIME = cfg["business"]["avg_lifetime_months"]
RETENTION_COST = cfg["business"]["retention_cost_aud"]
THRESHOLD_HIGH = cfg["scoring"]["thresholds"]["high"]
THRESHOLD_MEDIUM = cfg["scoring"]["thresholds"]["medium"]
API_URL = "http://localhost:8000"


# ── Load batch scores ─────────────────────────────────────
@st.cache_data
def load_scores():
    return pd.read_csv("data/outputs/churn_scores.csv")


# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Churn Risk Scoring",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Churn Risk Scoring Dashboard")
st.caption("End-to-end churn prediction pipeline — simulated for an Australian telco operator")

df = load_scores()

# Reconstruct categorical columns from one-hot
df["Contract"] = "Month-to-month"
df.loc[df["Contract_One year"] == 1, "Contract"] = "One year"
df.loc[df["Contract_Two year"] == 1, "Contract"] = "Two year"

df["InternetService"] = "DSL"
df.loc[df["InternetService_Fiber optic"] == 1, "InternetService"] = "Fiber optic"
df.loc[df["InternetService_No"] == 1, "InternetService"] = "No"

df["PaymentMethod"] = "Bank transfer (automatic)"
df.loc[df["PaymentMethod_Electronic check"] == 1, "PaymentMethod"] = "Electronic check"
df.loc[df["PaymentMethod_Mailed check"] == 1, "PaymentMethod"] = "Mailed check"
df.loc[df["PaymentMethod_Credit card (automatic)"] == 1, "PaymentMethod"] = "Credit card (automatic)"

# ═══════════════════════════════════════════════════════════
# 1. OVERVIEW
# ═══════════════════════════════════════════════════════════
st.header("1. Business Overview")

n_total = len(df)
n_high = len(df[df["risk_segment"] == "high"])
n_medium = len(df[df["risk_segment"] == "medium"])
n_low = len(df[df["risk_segment"] == "low"])
churn_rate = df["churn_true"].mean()
revenue_at_risk = n_high * ARPU
clv_at_risk = revenue_at_risk * AVG_LIFETIME
retention_cost = n_high * RETENTION_COST
roi = (revenue_at_risk * 0.5) / retention_cost

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{n_total:,}")
col2.metric("Churn Rate (observed)", f"{churn_rate:.1%}")
col3.metric("Monthly Revenue at Risk", f"${revenue_at_risk:,.0f} AUD")
col4.metric("ROI (50% retention)", f"{roi:.1f}x")

st.caption(
    f"⚠️ Revenue at Risk reflects **model-predicted** high-risk customers ({n_high:,}), "
    f"not observed churners (1,869). Difference represents customers at risk but not yet churned."
)

col1, col2, col3 = st.columns(3)
col1.metric("🔴 High Risk", f"{n_high:,}", f"{n_high/n_total:.1%} of base")
col2.metric("🟡 Medium Risk", f"{n_medium:,}", f"{n_medium/n_total:.1%} of base")
col3.metric("🟢 Low Risk", f"{n_low:,}", f"{n_low/n_total:.1%} of base")

# ═══════════════════════════════════════════════════════════
# 2. RISK SEGMENTATION
# ═══════════════════════════════════════════════════════════
st.header("2. Risk Segmentation")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        df, x="churn_probability",
        color="risk_segment",
        color_discrete_map={"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"},
        nbins=50,
        title="Churn Probability Distribution",
        labels={"churn_probability": "Churn Probability", "count": "Customers"}
    )
    fig.add_vline(x=THRESHOLD_HIGH, line_dash="dash", line_color="red", annotation_text="High threshold")
    fig.add_vline(x=THRESHOLD_MEDIUM, line_dash="dash", line_color="orange", annotation_text="Medium threshold")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    segment_counts = df["risk_segment"].value_counts().reset_index()
    segment_counts.columns = ["risk_segment", "count"]
    segment_order = ["high", "medium", "low"]
    segment_counts["risk_segment"] = pd.Categorical(
        segment_counts["risk_segment"], categories=segment_order, ordered=True
    )
    segment_counts = segment_counts.sort_values("risk_segment")
    fig = px.bar(
        segment_counts, x="risk_segment", y="count",
        color="risk_segment",
        color_discrete_map={"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"},
        title="Customers by Risk Segment",
        labels={"risk_segment": "Risk Segment", "count": "Customers"}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Churn rate by key features
col1, col2, col3 = st.columns(3)

with col1:
    churn_by_contract = df.groupby("Contract")["churn_probability"].mean().reset_index()
    fig = px.bar(churn_by_contract, x="Contract", y="churn_probability",
                 title="Avg Churn Probability by Contract",
                 color="churn_probability", color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    churn_by_internet = df.groupby("InternetService")["churn_probability"].mean().reset_index()
    fig = px.bar(churn_by_internet, x="InternetService", y="churn_probability",
                 title="Avg Churn Probability by Internet Service",
                 color="churn_probability", color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)

with col3:
    churn_by_payment = df.groupby("PaymentMethod")["churn_probability"].mean().reset_index()
    fig = px.bar(churn_by_payment, x="PaymentMethod", y="churn_probability",
                 title="Avg Churn Probability by Payment Method",
                 color="churn_probability", color_continuous_scale="Reds")
    fig.update_xaxes(tickangle=15)
    st.plotly_chart(fig, use_container_width=True)

# Top high risk customers
st.subheader("Top High Risk Customers")
top_high_risk = (
    df[df["risk_segment"] == "high"]
    .sort_values("churn_probability", ascending=False)
    [["churn_probability", "risk_segment", "Contract", "InternetService", "MonthlyCharges", "tenure"]]
    .head(10)
    .reset_index(drop=True)
)
top_high_risk["churn_probability"] = top_high_risk["churn_probability"].map("{:.1%}".format)
top_high_risk["MonthlyCharges"] = top_high_risk["MonthlyCharges"].map("${:.2f}".format)
st.dataframe(top_high_risk, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# 3. INDIVIDUAL SCORING
# ═══════════════════════════════════════════════════════════
st.header("3. Individual Customer Scoring")
st.caption("Simulate a real-time retention agent query via the scoring API")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=2)
        monthly_charges = st.number_input("Monthly Charges (AUD)", min_value=0.0, max_value=200.0, value=70.35)
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    with col2:
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

    with col3:
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    submitted = st.form_submit_button("Score Customer", type="primary")

if submitted:
    payload = {
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        result = response.json()

        prob = result["churn_probability"]
        segment = result["risk_segment"]
        factors = result["top_factors"]

        # Remaining lifetime correction
        remaining_lifetime = max(AVG_LIFETIME - tenure, 1)
        revenue_at_risk_individual = prob * ARPU * remaining_lifetime

        segment_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}

        col1, col2, col3 = st.columns(3)
        col1.metric("Churn Probability", f"{prob:.1%}")
        col2.metric("Risk Segment", f"{segment_color[segment]} {segment.upper()}")
        col3.metric("Est. Revenue at Risk", f"${revenue_at_risk_individual:,.0f} AUD")

        st.caption(
            f"Revenue at risk estimated over remaining lifetime "
            f"({remaining_lifetime} months = {AVG_LIFETIME} avg lifetime − {tenure} months tenure)"
        )

        st.subheader("Top Risk Factors")
        for factor in factors:
            impact_icon = "⬆️" if factor["impact"] == "increases risk" else "⬇️"
            st.markdown(f"{impact_icon} **{factor['feature']}**: {factor['value']} — *{factor['impact']}*")

    except Exception as e:
        st.error(f"API error: {e}. Make sure the API is running on {API_URL}")