"""
================================================================================
CODTECH INTERNSHIP - TASK 3: END-TO-END DATA SCIENCE PROJECT
================================================================================
Author      : Data Science Intern
Description : Full Data Science pipeline — Data Collection, Preprocessing,
              Model Training, and Deployment as a REST API using FastAPI.
Problem     : Predict whether a customer will churn (leave) or not.
Dataset     : Synthetic Telecom Customer Churn Dataset
================================================================================

HOW TO RUN:
    Step 1 — Train & save the model:
        python task3_train.py

    Step 2 — Start the API:
        uvicorn task3_app:app --reload

    Step 3 — Test in browser:
        http://127.0.0.1:8000/docs   ← Interactive Swagger UI
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# FILE 1: task3_train.py  — Data + Preprocessing + Model Training
# ─────────────────────────────────────────────────────────────────────────────
# Run this file first to generate:
#   • model.pkl         (trained model)
#   • preprocessor.pkl  (fitted scaler/encoder)
#   • label_encoder.pkl (target encoder)
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "task3_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================================
# STEP 1: DATA COLLECTION — Generate Synthetic Telecom Churn Dataset
# ================================================================================

def collect_data(n=1000, seed=42):
    print("\n" + "="*60)
    print("  STEP 1: DATA COLLECTION")
    print("="*60)

    np.random.seed(seed)

    data = {
        "customer_id"       : [f"CUST{i:04d}" for i in range(1, n+1)],
        "tenure_months"     : np.random.randint(1, 72, n),
        "monthly_charges"   : np.round(np.random.uniform(20, 120, n), 2),
        "total_charges"     : np.round(np.random.uniform(100, 8000, n), 2),
        "num_products"      : np.random.randint(1, 6, n),
        "contract_type"     : np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "payment_method"    : np.random.choice(["Credit card", "Bank transfer", "Electronic check", "Mailed check"], n),
        "internet_service"  : np.random.choice(["DSL", "Fiber optic", "No"], n),
        "tech_support"      : np.random.choice(["Yes", "No"], n),
        "online_security"   : np.random.choice(["Yes", "No"], n),
        "senior_citizen"    : np.random.choice([0, 1], n, p=[0.84, 0.16]),
        "dependents"        : np.random.choice(["Yes", "No"], n),
        "paperless_billing" : np.random.choice(["Yes", "No"], n),
    }

    df = pd.DataFrame(data)

    # Churn logic — realistic rules
    churn_prob = (
        0.3
        + (df["contract_type"] == "Month-to-month").astype(float) * 0.25
        + (df["tenure_months"] < 12).astype(float) * 0.15
        + (df["monthly_charges"] > 80).astype(float) * 0.10
        - (df["tech_support"] == "Yes").astype(float) * 0.10
        - (df["online_security"] == "Yes").astype(float) * 0.08
    ).clip(0.05, 0.95)

    df["churn"] = (np.random.uniform(0, 1, n) < churn_prob).astype(int)

    # Introduce missing values
    df.loc[np.random.choice(df.index, 30, replace=False), "total_charges"] = np.nan
    df.loc[np.random.choice(df.index, 15, replace=False), "monthly_charges"] = np.nan

    # Save raw dataset
    raw_path = os.path.join(OUTPUT_DIR, "churn_raw_dataset.csv")
    df.to_csv(raw_path, index=False)

    print(f"  ✔ Dataset created  : {df.shape}")
    print(f"  ✔ Churn rate       : {df['churn'].mean()*100:.1f}%")
    print(f"  ✔ Raw data saved   : {raw_path}")
    return df


# ================================================================================
# STEP 2: PREPROCESSING
# ================================================================================

def preprocess_data(df):
    print("\n" + "="*60)
    print("  STEP 2: PREPROCESSING")
    print("="*60)

    df = df.drop(columns=["customer_id"])

    TARGET = "churn"
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_cols     = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"  ✔ Numeric features    : {numeric_cols}")
    print(f"  ✔ Categorical features: {categorical_cols}")

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline,   numeric_cols),
        ("cat", cat_pipeline,   categorical_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    print(f"  ✔ Train shape: {X_train_proc.shape}")
    print(f"  ✔ Test  shape: {X_test_proc.shape}")

    # Save preprocessor
    with open(os.path.join(OUTPUT_DIR, "preprocessor.pkl"), "wb") as f:
        pickle.dump((preprocessor, numeric_cols, categorical_cols), f)
    print("  ✔ Preprocessor saved")

    return X_train_proc, X_test_proc, y_train, y_test, X_train, X_test


# ================================================================================
# STEP 3: MODEL TRAINING & SELECTION
# ================================================================================

def train_models(X_train, y_train, X_test, y_test):
    print("\n" + "="*60)
    print("  STEP 3: MODEL TRAINING & SELECTION")
    print("="*60)

    models = {
        "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    best_model, best_auc, best_name = None, 0, ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred   = model.predict(X_test)
        y_prob   = model.predict_proba(X_test)[:, 1]
        acc      = accuracy_score(y_test, y_pred)
        roc      = roc_auc_score(y_test, y_prob)
        results[name] = {"accuracy": acc, "roc_auc": roc, "model": model}
        print(f"  {name:<25} Accuracy={acc*100:.2f}%  ROC-AUC={roc:.4f}")

        if roc > best_auc:
            best_auc, best_model, best_name = roc, model, name

    print(f"\n  ✔ Best model: {best_name} (ROC-AUC = {best_auc:.4f})")

    # Save best model
    with open(os.path.join(OUTPUT_DIR, "model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    print("  ✔ Best model saved → model.pkl")

    return best_model, best_name, results


# ================================================================================
# STEP 4: EVALUATION & VISUALIZATION
# ================================================================================

def evaluate_and_visualize(model, X_test, y_test, model_name, results):
    print("\n" + "="*60)
    print("  STEP 4: EVALUATION & VISUALIZATION")
    print("="*60)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n  Classification Report ({model_name}):")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Task 3 — Churn Prediction Results ({model_name})", fontsize=14, fontweight="bold")

    # Model comparison bar chart
    ax = axes[0]
    names = list(results.keys())
    aucs  = [results[n]["roc_auc"] for n in names]
    bars  = ax.bar(names, aucs, color=["steelblue","darkorange","green"])
    ax.set_title("Model Comparison (ROC-AUC)")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.5, 1.0)
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)

    # Confusion Matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn","Churn"])
    disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
    axes[1].set_title("Confusion Matrix")

    # ROC Curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc     = auc(fpr, tpr)
    axes[2].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[2].plot([0,1],[0,1], "k--")
    axes[2].set_title("ROC Curve")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "results_visualization.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  ✔ Visualization saved → {save_path}")


# ================================================================================
# MAIN
# ================================================================================

if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  CODTECH TASK 3 — TRAINING PIPELINE")
    print("█"*60)

    df                                          = collect_data()
    X_tr, X_te, y_tr, y_te, X_tr_raw, X_te_raw = preprocess_data(df)
    best_model, best_name, results              = train_models(X_tr, y_tr, X_te, y_te)
    evaluate_and_visualize(best_model, X_te, y_te, best_name, results)

    print("\n  ✅ Training complete! Now run: uvicorn task3_app:app --reload")
