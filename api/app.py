from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
from pathlib import Path
import numpy as np

# -----------------------------
# App & Model Load
# -----------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict churn risk and explain predictions using SHAP",
    version="1.0"
)

MODEL_PATH = Path("models/random_forest.joblib")
pipeline = joblib.load(MODEL_PATH)

preprocessor = pipeline.named_steps["preprocessing"]
model = pipeline.named_steps["model"]

# -----------------------------
# Request Schema
# -----------------------------
class CustomerInput(BaseModel):
    age: int
    tenure_months: int
    monthly_charges: float
    total_charges: float
    usage_minutes: int
    support_tickets: int
    last_login_days: int
    contract_type: str
    payment_method: str


# -----------------------------
# Helper Functions
# -----------------------------
def predict_proba(df: pd.DataFrame) -> float:
    return pipeline.predict_proba(df)[0, 1]


def explain_prediction(df: pd.DataFrame, top_k: int = 5):
    # Transform features
    X_transformed = preprocessor.transform(df)
    feature_names = preprocessor.get_feature_names_out()

    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    # Handle SHAP return format
    if isinstance(shap_values, list):
        values = shap_values[1][0]
        base_value = explainer.expected_value[1]
    else:
        values = shap_values[0]
        base_value = explainer.expected_value

    # Top drivers
    importance = np.abs(values)
    top_idx = np.argsort(importance)[::-1][:top_k]

    reasons = [
        {
            "feature": feature_names[i],
            "shap_value": float(values[i])
        }
        for i in top_idx
    ]

    return reasons, float(base_value)


# -----------------------------
# API Endpoints
# -----------------------------
@app.post("/predict")
def predict(input_data: CustomerInput):
    df = pd.DataFrame([input_data.dict()])
    prob = predict_proba(df)

    risk = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"

    return {
        "churn_probability": round(prob, 4),
        "risk_level": risk
    }


@app.post("/explain")
def explain(input_data: CustomerInput):
    df = pd.DataFrame([input_data.dict()])
    prob = predict_proba(df)

    reasons, base_value = explain_prediction(df)

    return {
        "churn_probability": round(prob, 4),
        "base_value": round(base_value, 4),
        "top_reasons": reasons
    }
