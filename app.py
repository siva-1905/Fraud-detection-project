import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# =====================================================
# Load Model
# =====================================================
MODEL_PATH = Path("models/best_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# =====================================================
# Page Setup
# =====================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details and predict whether a transaction is fraudulent.")

# =====================================================
# User Inputs
# =====================================================
st.sidebar.header("Transaction Information")

amount = st.sidebar.number_input("Amount", min_value=0.0, value=100.0)

transaction_hour = st.sidebar.slider(
    "Transaction Hour",
    min_value=0,
    max_value=23,
    value=12
)

device_trust_score = st.sidebar.slider(
    "Device Trust Score",
    min_value=0.0,
    max_value=1.0,
    value=0.80
)

velocity_last_24h = st.sidebar.number_input(
    "Transactions in Last 24 Hours",
    min_value=0,
    value=2
)

cardholder_age = st.sidebar.number_input(
    "Cardholder Age",
    min_value=18,
    max_value=100,
    value=30
)

merchant_category = st.sidebar.selectbox(
    "Merchant Category",
    ["Electronics", "Food", "Grocery", "Travel"]
)

foreign_transaction = st.sidebar.checkbox("Foreign Transaction")
location_mismatch = st.sidebar.checkbox("Location Mismatch")

# =====================================================
# Engineered Features
# =====================================================
amount_velocity_ratio = amount / max(velocity_last_24h, 1)

is_night_transaction = int(transaction_hour <= 5)

is_high_amount = int(amount > 1000)

is_high_velocity = int(velocity_last_24h > 10)

is_low_device_trust = int(device_trust_score < 0.30)

is_foreign_mismatch = int(
    foreign_transaction and location_mismatch
)

# =====================================================
# One-Hot Encoding
# =====================================================
merchant_category_Electronics = int(
    merchant_category == "Electronics"
)

merchant_category_Food = int(
    merchant_category == "Food"
)

merchant_category_Grocery = int(
    merchant_category == "Grocery"
)

merchant_category_Travel = int(
    merchant_category == "Travel"
)

# =====================================================
# Create Input DataFrame
# =====================================================
input_df = pd.DataFrame({
    "amount": [amount],
    "transaction_hour": [transaction_hour],
    "device_trust_score": [device_trust_score],
    "velocity_last_24h": [velocity_last_24h],
    "cardholder_age": [cardholder_age],
    "amount_velocity_ratio": [amount_velocity_ratio],
    "merchant_category_Electronics": [merchant_category_Electronics],
    "merchant_category_Food": [merchant_category_Food],
    "merchant_category_Grocery": [merchant_category_Grocery],
    "merchant_category_Travel": [merchant_category_Travel],
    "foreign_transaction": [int(foreign_transaction)],
    "location_mismatch": [int(location_mismatch)],
    "is_night_transaction": [is_night_transaction],
    "is_high_amount": [is_high_amount],
    "is_high_velocity": [is_high_velocity],
    "is_low_device_trust": [is_low_device_trust],
    "is_foreign_mismatch": [is_foreign_mismatch],
})

# =====================================================
# Display Inputs
# =====================================================
st.subheader("Input Features")
st.dataframe(input_df)

# =====================================================
# Prediction
# =====================================================
if st.button("Predict"):

    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        fraud_probability = model.predict_proba(input_df)[0][1]
    else:
        fraud_probability = None

    st.markdown("---")

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

    if fraud_probability is not None:
        st.metric(
            "Fraud Probability",
            f"{fraud_probability:.2%}"
        )