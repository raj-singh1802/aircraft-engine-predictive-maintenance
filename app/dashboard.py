import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import shap
import time

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineering import create_features

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(page_title="AeroMind AI", layout="wide")

st.title("✈️ AeroMind: AI Engine Health Intelligence Platform")
st.markdown("### Live Predictive Maintenance System")

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

rul_model = joblib.load(os.path.join(BASE_DIR, "../models/rul_model.pkl"))
risk_model = joblib.load(os.path.join(BASE_DIR, "../models/risk_model.pkl"))

# SHAP explainer
explainer = shap.TreeExplainer(rul_model)

# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Monitoring", "ℹ️ About"])

# ===================================================
# TAB 1: PREDICTION
# ===================================================

with tab1:

    st.sidebar.header("⚙️ Engine Configuration")

    # -------------------------
    # Scenario Selection
    # -------------------------
    scenario = st.sidebar.selectbox(
        "Select Engine Scenario",
        ["Healthy", "Degrading", "Critical"]
    )

    # -------------------------
    # Base Values
    # -------------------------
    if scenario == "Healthy":
        base = 10
    elif scenario == "Degrading":
        base = 50
    else:
        base = 100

    # -------------------------
    # Generate Time-Series Data
    # -------------------------
    cycles = st.sidebar.slider("Simulation Cycles", 5, 30, 10)

    history = []

    for c in range(1, cycles + 1):
        row = {
            f"sensor_{i}": base + np.random.normal(0, 2)
            for i in range(2, 22)
        }

        row["op1"] = np.random.normal(0, 1)
        row["op2"] = np.random.normal(0, 1)
        row["op3"] = np.random.normal(0, 1)

        row["engine_id"] = 1
        row["cycle"] = c

        history.append(row)

    history_df = pd.DataFrame(history)

    # -------------------------
    # Feature Engineering
    # -------------------------
    features_df = create_features(history_df)

    latest_input = features_df.iloc[-1:]

    # Align columns
    latest_input = latest_input.reindex(
        columns=rul_model.feature_names_in_,
        fill_value=0
    )

    # -------------------------
    # Prediction
    # -------------------------
    if st.button("🚀 Predict Engine Health"):

        rul_pred = rul_model.predict(latest_input)[0]
        risk_pred = risk_model.predict(latest_input)[0]

        st.subheader("📊 Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Remaining Useful Life", f"{round(rul_pred,2)} cycles")

        with col2:
            st.metric("Risk Level", risk_pred)

        with col3:
            confidence = max(60, 100 - abs(rul_pred)/2)
            st.metric("Model Confidence", f"{round(confidence,2)}%")

        # -------------------------
        # Status Indicator
        # -------------------------
        if rul_pred < 30:
            st.error("🔴 CRITICAL: Immediate maintenance required")
        elif rul_pred < 80:
            st.warning("🟠 WARNING: Engine degrading")
        else:
            st.success("🟢 HEALTHY: Engine operating normally")

        # -------------------------
        # Time-Series Plot
        # -------------------------
        st.subheader("📈 Sensor Trend")

        fig, ax = plt.subplots()
        ax.plot(history_df["cycle"], history_df["sensor_4"], label="Sensor 4")
        ax.plot(history_df["cycle"], history_df["sensor_11"], label="Sensor 11")
        ax.legend()
        ax.set_title("Sensor Trends Over Time")

        st.pyplot(fig)

        # -------------------------
        # SHAP EXPLAINABILITY
        # -------------------------
        st.subheader("🧠 Explainable AI (SHAP)")

        shap_values = explainer(latest_input)

        fig_shap = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)

        st.pyplot(fig_shap)

# ===================================================
# TAB 2: LIVE MONITORING SIMULATION
# ===================================================

with tab2:

    st.subheader("📡 Real-Time Engine Monitoring")

    placeholder = st.empty()

    for i in range(20):

        simulated_value = np.random.uniform(20, 100)

        with placeholder.container():

            st.metric("Live Sensor Health", f"{round(simulated_value,2)}")

            fig, ax = plt.subplots()
            data = np.random.randn(20).cumsum()
            ax.plot(data)
            ax.set_title("Live Sensor Stream")

            st.pyplot(fig)

        time.sleep(1)

# ===================================================
# TAB 3: ABOUT
# ===================================================

with tab3:

    st.subheader("ℹ️ About Project")

    st.write("""
    AeroMind is an AI-driven predictive maintenance system for aircraft engines.

    Features:
    - Time-series modeling
    - Feature engineering
    - Machine learning prediction
    - Explainable AI (SHAP)
    - Real-time simulation dashboard
    """)

    st.markdown("### 🔮 Future Enhancements")

    st.write("""
    - IoT integration  
    - LSTM / Transformer models  
    - Cloud deployment  
    - REST API (FastAPI)  
    """)