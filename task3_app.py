"""
================================================================================
CODTECH INTERNSHIP - TASK 3: FastAPI Deployment
================================================================================
Run AFTER task3_train.py has been executed.

Start server:
    uvicorn task3_app:app --reload

Open browser:
    http://127.0.0.1:8000/docs   ← Swagger UI to test the API
================================================================================
"""

import pickle
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

# ─────────────────────────────────────────────
# Load saved model and preprocessor
# ─────────────────────────────────────────────
OUTPUT_DIR = "task3_output"

with open(os.path.join(OUTPUT_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(OUTPUT_DIR, "preprocessor.pkl"), "rb") as f:
    preprocessor, numeric_cols, categorical_cols = pickle.load(f)

# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="CODTECH Internship Task 3 — Predict whether a telecom customer will churn.",
    version="1.0.0"
)

# ─────────────────────────────────────────────
# Input schema — validates request body
# ─────────────────────────────────────────────
class CustomerData(BaseModel):
    tenure_months    : int
    monthly_charges  : float
    total_charges    : float
    num_products     : int
    contract_type    : Literal["Month-to-month", "One year", "Two year"]
    payment_method   : Literal["Credit card", "Bank transfer", "Electronic check", "Mailed check"]
    internet_service : Literal["DSL", "Fiber optic", "No"]
    tech_support     : Literal["Yes", "No"]
    online_security  : Literal["Yes", "No"]
    senior_citizen   : Literal[0, 1]
    dependents       : Literal["Yes", "No"]
    paperless_billing: Literal["Yes", "No"]

    class Config:
        json_schema_extra = {
            "example": {
                "tenure_months"    : 5,
                "monthly_charges"  : 95.5,
                "total_charges"    : 450.0,
                "num_products"     : 2,
                "contract_type"    : "Month-to-month",
                "payment_method"   : "Electronic check",
                "internet_service" : "Fiber optic",
                "tech_support"     : "No",
                "online_security"  : "No",
                "senior_citizen"   : 1,
                "dependents"       : "No",
                "paperless_billing": "Yes"
            }
        }


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running!",
        "docs"   : "Visit /docs for interactive Swagger UI"
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model": type(model).__name__}


@app.post("/predict")
def predict(data: CustomerData):
    """
    Predict churn for a single customer.
    Returns: churn prediction (0 or 1) + probability + risk level.
    """
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Preprocess using saved pipeline
    input_processed = preprocessor.transform(input_df)

    # Predict
    prediction  = int(model.predict(input_processed)[0])
    probability = float(model.predict_proba(input_processed)[0][1])

    # Risk level
    if probability >= 0.7:
        risk = "HIGH"
    elif probability >= 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "churn_prediction" : prediction,
        "churn_label"      : "Will Churn" if prediction == 1 else "Will NOT Churn",
        "churn_probability": round(probability, 4),
        "risk_level"       : risk,
        "recommendation"   : (
            "Immediate retention action needed!"
            if risk == "HIGH" else
            "Monitor closely and offer loyalty benefits."
            if risk == "MEDIUM" else
            "Customer is stable. No action needed."
        )
    }


@app.post("/predict_batch")
def predict_batch(customers: list[CustomerData]):
    """
    Predict churn for multiple customers at once.
    """
    results = []
    for customer in customers:
        input_df        = pd.DataFrame([customer.dict()])
        input_processed = preprocessor.transform(input_df)
        prediction      = int(model.predict(input_processed)[0])
        probability     = float(model.predict_proba(input_processed)[0][1])
        results.append({
            "churn_prediction" : prediction,
            "churn_probability": round(probability, 4),
            "churn_label"      : "Will Churn" if prediction == 1 else "Will NOT Churn"
        })
    return {"predictions": results, "total": len(results)}
