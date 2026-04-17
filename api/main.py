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

    df = pd.DataFrame([data])

    df["engine_id"] = 1
    df["cycle"] = 1

    df = create_features(df)

    df = df.reindex(columns=rul_model.feature_names_in_, fill_value=0)

    rul = float(rul_model.predict(df)[0])
    risk = str(risk_model.predict(df)[0])

    return {
        "RUL": rul,
        "Risk": risk
    }