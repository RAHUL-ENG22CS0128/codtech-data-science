# Task 3: End-to-End Data Science Project — Customer Churn Prediction API

## Overview
A full data science project: data collection → preprocessing → model training → **deployed as a REST API using FastAPI**.  
Predicts whether a telecom customer will churn (leave) or stay.

---

## Files

| File | Description |
|------|-------------|
| `task3_train.py` | Data generation, preprocessing, model training & evaluation |
| `task3_app.py` | FastAPI web app — serves predictions via REST API |
| `task3_output/model.pkl` | Saved trained model (generated on run) |
| `task3_output/preprocessor.pkl` | Saved preprocessor (generated on run) |
| `task3_output/churn_raw_dataset.csv` | Raw dataset (generated on run) |
| `task3_output/results_visualization.png` | Evaluation charts (generated on run) |
| `README.md` | Project documentation |

---

## How to Run

### Step 1 — Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib fastapi uvicorn
```

### Step 2 — Train the model
```bash
python task3_train.py
```

### Step 3 — Start the API
```bash
uvicorn task3_app:app --reload
```

### Step 4 — Test in browser
Open: **http://127.0.0.1:8000/docs**  
This opens the interactive Swagger UI where you can test all endpoints.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | API health check |
| POST | `/predict` | Predict churn for one customer |
| POST | `/predict_batch` | Predict churn for multiple customers |

### Example Request (`/predict`)
```json
{
  "tenure_months": 5,
  "monthly_charges": 95.5,
  "total_charges": 450.0,
  "num_products": 2,
  "contract_type": "Month-to-month",
  "payment_method": "Electronic check",
  "internet_service": "Fiber optic",
  "tech_support": "No",
  "online_security": "No",
  "senior_citizen": 1,
  "dependents": "No",
  "paperless_billing": "Yes"
}
```

### Example Response
```json
{
  "churn_prediction": 1,
  "churn_label": "Will Churn",
  "churn_probability": 0.8123,
  "risk_level": "HIGH",
  "recommendation": "Immediate retention action needed!"
}
```

---

## Model Pipeline
- **3 models compared**: Logistic Regression, Random Forest, Gradient Boosting
- **Best model selected** automatically by ROC-AUC score
- **Expected accuracy**: ~85%

---

## Internship Details
- **Organization**: CODTECH IT Solutions
- **Task**: Task 3 — End-to-End Data Science Project
- **Domain**: Data Science
