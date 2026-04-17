from fastapi import FastAPI
import pandas as pd
import joblib
import os

from src.feature_engineering import create_features

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

rul_model = joblib.load(os.path.join(BASE_DIR, "../models/rul_model.pkl"))
risk_model = joblib.load(os.path.join(BASE_DIR, "../models/risk_model.pkl"))

@app.get("/")
def home():
    return {"message": "AI Engine API Running"}

@app.post("/predict")
def predict(data: dict):

    import numpy as np

    # -------------------------
    # CREATE TIME SERIES HISTORY
    # -------------------------
    history = []

    for c in range(1, 6):   # 5 cycles
        row = data.copy()

        # Add slight variation
        for key in row:
            if "sensor" in key:
                row[key] = row[key] + np.random.normal(0, 1)

        row["engine_id"] = 1
        row["cycle"] = c

        history.append(row)

    df = pd.DataFrame(history)

    # -------------------------
    # FEATURE ENGINEERING
    # -------------------------
    df = create_features(df)

    latest = df.iloc[-1:]

    latest = latest.reindex(
        columns=rul_model.feature_names_in_,
        fill_value=0
    )

    # -------------------------
    # PREDICTION
    # -------------------------
    rul = float(rul_model.predict(latest)[0])
    risk = str(risk_model.predict(latest)[0])

    return {
        "RUL": rul,
        "Risk": risk
    }