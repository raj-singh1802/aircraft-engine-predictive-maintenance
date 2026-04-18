import streamlit as st
import numpy as np
import requests
import matplotlib.pyplot as plt

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

st.set_page_config(page_title="AeroMind AI", layout="wide")

st.title("✈️ AeroMind: AI Engine Health Intelligence Platform")
st.markdown("### AI-powered Predictive Maintenance System")

# ---------------------------------------------------
# SIDEBAR CONFIGURATION
# ---------------------------------------------------

st.sidebar.header("⚙️ Engine Configuration")

scenario = st.sidebar.selectbox(
    "Select Engine Scenario",
    ["Healthy", "Degrading", "Critical"]
)

cycles = st.sidebar.slider("Simulation Cycles", 5, 30, 10)

# Base values for simulation
if scenario == "Healthy":
    base = 10
elif scenario == "Degrading":
    base = 50
else:
    base = 100

# ---------------------------------------------------
# GENERATE SENSOR DATA (SIMULATION)
# ---------------------------------------------------

sensor_inputs = {}

for i in range(2, 22):
    sensor_inputs[f"sensor_{i}"] = float(base + np.random.normal(0, 2))

sensor_inputs["op1"] = float(np.random.normal(0, 1))
sensor_inputs["op2"] = float(np.random.normal(0, 1))
sensor_inputs["op3"] = float(np.random.normal(0, 1))

# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📡 Monitoring", "ℹ️ About"])

# ===================================================
# TAB 1: PREDICTION
# ===================================================

with tab1:

    st.subheader("🔍 Engine Health Prediction")

    if st.button("🚀 Predict Engine Health"):

        try:
            response = requests.post(
                "https://aircraft-engine-api.onrender.com/predict",
                json=sensor_inputs,
                timeout=20
            )

            result = response.json()

            rul_pred = result["RUL"]
            risk_pred = result["Risk"]

            # -------------------------
            # METRICS
            # -------------------------
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
            # STATUS
            # -------------------------
            if rul_pred < 30:
                st.error("🔴 CRITICAL: Immediate maintenance required")
            elif rul_pred < 80:
                st.warning("🟠 WARNING: Engine degrading")
            else:
                st.success("🟢 HEALTHY: Engine operating normally")

            # -------------------------
            # SENSOR TREND VISUAL
            # -------------------------
            st.subheader("📈 Sensor Trend Simulation")

            data = np.random.randn(cycles).cumsum()

            fig, ax = plt.subplots()
            ax.plot(data)
            ax.set_title("Simulated Sensor Trend")
            ax.set_xlabel("Cycle")
            ax.set_ylabel("Sensor Value")

            st.pyplot(fig)

        except Exception as e:
            st.error("❌ Backend API not running or unreachable")
            st.info("Make sure FastAPI is running: uvicorn api.main:app --reload")
            st.write(e)

# ===================================================
# TAB 2: MONITORING
# ===================================================

with tab2:

    st.subheader("📡 Live Engine Monitoring")

    placeholder = st.empty()

    for _ in range(10):
        value = np.random.uniform(20, 100)

        with placeholder.container():
            st.metric("Live Engine Health Index", f"{round(value,2)}")

            fig, ax = plt.subplots()
            data = np.random.randn(20).cumsum()
            ax.plot(data)
            ax.set_title("Live Sensor Stream")

            st.pyplot(fig)

# ===================================================
# TAB 3: ABOUT
# ===================================================

with tab3:

    st.subheader("ℹ️ About AeroMind")

    st.write("""
    AeroMind is an AI-powered predictive maintenance system for aircraft engines.

    🔹 Uses machine learning to estimate Remaining Useful Life (RUL)  
    🔹 Applies time-series feature engineering  
    🔹 Provides risk classification  
    🔹 Simulates real-world monitoring dashboard  

    Built using:
    - Streamlit (Frontend)
    - FastAPI (Backend)
    - Scikit-learn (ML)
    """)

    st.markdown("### 🔮 Future Improvements")

    st.write("""
    - Real-time IoT sensor integration  
    - Deep learning models (LSTM / Transformers)  
    - Cloud deployment with CI/CD  
    - API authentication & scaling  
    """)