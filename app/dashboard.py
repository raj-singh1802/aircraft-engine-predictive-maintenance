import streamlit as st
import pandas as pd
import joblib
import os

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Aircraft Engine Predictive Maintenance",
    layout="wide"
)

st.title("✈️ Aircraft Engine Predictive Maintenance System")

st.write(
"""
This AI system predicts the **Remaining Useful Life (RUL)** of aircraft engines
using sensor data and machine learning.
"""
)

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

rul_model = joblib.load(os.path.join(BASE_DIR, "../models/rul_model.pkl"))
risk_model = joblib.load(os.path.join(BASE_DIR, "../models/risk_model.pkl"))

st.success("Models loaded successfully")

# ---------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------

st.sidebar.header("Engine Sensor Inputs")

sensor_inputs = {}

# Sensor Inputs
for i in range(2, 22):
    sensor_inputs[f"sensor_{i}"] = st.sidebar.number_input(
        f"Sensor {i}", value=0.0
    )

# Operating Conditions
st.sidebar.header("Operating Conditions")

sensor_inputs["op1"] = st.sidebar.number_input(
    "Operating Condition 1", value=0.0
)

sensor_inputs["op2"] = st.sidebar.number_input(
    "Operating Condition 2", value=0.0
)

sensor_inputs["op3"] = st.sidebar.number_input(
    "Operating Condition 3", value=0.0
)

# ---------------------------------------------------
# Convert Inputs to DataFrame
# ---------------------------------------------------

input_df = pd.DataFrame([sensor_inputs])

# ---------------------------------------------------
# Create Missing Engineered Features
# ---------------------------------------------------

# Your model expects engineered features (rolling mean, std, delta)
# We create placeholders so the model receives the correct feature structure

for col in rul_model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match training order
input_df = input_df[rul_model.feature_names_in_]

# ---------------------------------------------------
# Prediction Button
# ---------------------------------------------------

if st.button("Predict Engine Health"):

    rul_pred = rul_model.predict(input_df)[0]
    risk_pred = risk_model.predict(input_df)[0]

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Remaining Useful Life",
            f"{round(rul_pred,2)} cycles"
        )

    with col2:
        st.metric(
            "Failure Risk Level",
            risk_pred
        )

    # Risk alerts

    if risk_pred == "High":
        st.error("⚠ Immediate maintenance recommended")

    elif risk_pred == "Medium":
        st.warning("Monitor engine condition")

    else:
        st.success("Engine operating normally")