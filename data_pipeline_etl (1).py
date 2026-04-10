"""
================================================================================
CODTECH INTERNSHIP - TASK 1: DATA PIPELINE DEVELOPMENT
================================================================================
Author      : Data Science Intern
Description : A complete ETL (Extract, Transform, Load) pipeline using
              Pandas and Scikit-learn for data preprocessing and transformation.
Dataset     : Synthetic Employee/HR Dataset (generated programmatically)
================================================================================
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler,
    LabelEncoder, OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# ================================================================================
# STEP 1: EXTRACT — Generate / Load Raw Data
# ================================================================================

def extract_data(n_samples: int = 500, random_seed: int = 42) -> pd.DataFrame:
    """
    Simulates extracting raw data from a source (CSV / database / API).
    Here we generate a synthetic HR dataset with intentional messiness:
    missing values, inconsistent categories, and mixed dtypes.
    """
    print("\n" + "="*60)
    print("  STEP 1: EXTRACT — Loading Raw Data")
    print("="*60)

    np.random.seed(random_seed)

    departments = ["Engineering", "Marketing", "HR", "Finance", "Sales", None]
    genders      = ["Male", "Female", "male", "FEMALE", None]          # inconsistent casing

    data = {
        "employee_id"   : range(1, n_samples + 1),
        "age"           : np.random.randint(22, 60, n_samples).astype(float),
        "gender"        : np.random.choice(genders, n_samples),
        "department"    : np.random.choice(departments, n_samples),
        "years_exp"     : np.random.uniform(0, 35, n_samples),
        "salary"        : np.random.normal(60000, 15000, n_samples),
        "performance"   : np.random.choice([1, 2, 3, 4, 5, None], n_samples),
        "satisfaction"  : np.random.uniform(1, 10, n_samples),
        "left_company"  : np.random.choice([0, 1], n_samples),         # target variable
    }

    df = pd.DataFrame(data)

    # ── Introduce artificial missing values ──
    missing_idx_age  = np.random.choice(df.index, size=30, replace=False)
    missing_idx_sal  = np.random.choice(df.index, size=20, replace=False)
    df.loc[missing_idx_age, "age"]    = np.nan
    df.loc[missing_idx_sal, "salary"] = np.nan

    # ── Introduce some outliers ──
    df.loc[np.random.choice(df.index, 5), "salary"] = 500000   # extremely high salaries

    print(f"  ✔ Data extracted successfully — Shape: {df.shape}")
    print(f"  ✔ Columns: {list(df.columns)}")
    print(f"\n  Sample (first 5 rows):\n")
    print(df.head(5).to_string(index=False))
    return df


# ================================================================================
# STEP 2: TRANSFORM — Clean, Preprocess, and Engineer Features
# ================================================================================

# ── 2a. Basic Cleaning ──────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs foundational cleaning:
      - Drops exact duplicate rows
      - Standardises categorical casing
      - Clips extreme outliers using IQR
    """
    print("\n" + "="*60)
    print("  STEP 2a: CLEAN — Basic Data Cleaning")
    print("="*60)

    original_shape = df.shape

    # Drop duplicates
    df = df.drop_duplicates()
    print(f"  ✔ Duplicates removed : {original_shape[0] - df.shape[0]} rows dropped")

    # Standardise string columns to Title Case
    for col in ["gender", "department"]:
        df[col] = df[col].str.strip().str.title()

    # Clip salary outliers using IQR fence
    Q1, Q3 = df["salary"].quantile(0.25), df["salary"].quantile(0.75)
    IQR    = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    before_clip  = (df["salary"] > upper).sum()
    df["salary"]  = df["salary"].clip(lower, upper)
    print(f"  ✔ Salary outliers clipped : {before_clip} values capped")

    # Missing value summary
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(f"\n  Missing values after cleaning:")
    for col, cnt in missing.items():
        print(f"    - {col}: {cnt} missing ({cnt/len(df)*100:.1f}%)")

    return df


# ── 2b. Feature Engineering ─────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new informative features from existing columns.
    """
    print("\n" + "="*60)
    print("  STEP 2b: FEATURE ENGINEERING")
    print("="*60)

    # Salary-to-experience ratio (handle zero experience)
    df["salary_per_year_exp"] = df["salary"] / (df["years_exp"] + 1)

    # Age group bins
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 40, 50, 100],
        labels=["<30", "30-40", "40-50", "50+"]
    )

    # High performer flag
    df["high_performer"] = (df["performance"] >= 4).astype(float)

    # Seniority label based on years of experience
    df["seniority"] = pd.cut(
        df["years_exp"],
        bins=[-1, 2, 7, 15, 100],
        labels=["Junior", "Mid", "Senior", "Principal"]
    )

    print("  ✔ New features created:")
    print("    - salary_per_year_exp  (numeric)")
    print("    - age_group            (categorical bin)")
    print("    - high_performer       (binary flag)")
    print("    - seniority            (ordinal bin)")

    return df


# ── 2c. Scikit-learn Preprocessing Pipeline ─────────────────────────────────────

def build_preprocessing_pipeline(
    numeric_cols   : list,
    categorical_cols: list
) -> ColumnTransformer:
    """
    Builds a ColumnTransformer that applies:
      • Numeric  : Impute (median) → StandardScaler
      • Categorical: Impute (most_frequent) → OneHotEncoder
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline,   numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    return preprocessor


def transform_data(df: pd.DataFrame):
    """
    Applies the full Scikit-learn preprocessing pipeline and returns
    train/test arrays ready for modelling.
    """
    print("\n" + "="*60)
    print("  STEP 2c: TRANSFORM — Scikit-learn Pipeline")
    print("="*60)

    # Define target and features
    TARGET = "left_company"
    DROP_COLS = ["employee_id", TARGET]

    # Separate features and target
    X = df.drop(columns=DROP_COLS, errors="ignore")
    y = df[TARGET]

    # Identify column types for the pipeline
    numeric_cols    = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"  ✔ Numeric features    ({len(numeric_cols)}): {numeric_cols}")
    print(f"  ✔ Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  ✔ Train set : {X_train.shape[0]} rows")
    print(f"  ✔ Test  set : {X_test.shape[0]} rows")

    # Build and fit the preprocessor
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed  = preprocessor.transform(X_test)

    print(f"\n  ✔ Pipeline fitted successfully")
    print(f"  ✔ Processed train shape : {X_train_processed.shape}")
    print(f"  ✔ Processed test  shape : {X_test_processed.shape}")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


# ================================================================================
# STEP 3: LOAD — Save Processed Data to Disk
# ================================================================================

def load_data(
    X_train, X_test, y_train, y_test,
    output_dir: str = "pipeline_output"
) -> None:
    """
    Persists the processed train/test arrays as CSV files,
    simulating a 'load' step into a data warehouse or file store.
    """
    print("\n" + "="*60)
    print("  STEP 3: LOAD — Saving Processed Data")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    # Convert arrays to DataFrames for easy saving
    train_df = pd.DataFrame(X_train)
    train_df.insert(0, "target", y_train.values)

    test_df  = pd.DataFrame(X_test)
    test_df.insert(0, "target", y_test.values)

    train_path = os.path.join(output_dir, "train_processed.csv")
    test_path  = os.path.join(output_dir, "test_processed.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

    print(f"  ✔ Train data saved → {train_path}  ({train_df.shape})")
    print(f"  ✔ Test  data saved → {test_path}   ({test_df.shape})")


# ================================================================================
# STEP 4: PIPELINE SUMMARY REPORT
# ================================================================================

def print_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame,
                  X_train, X_test) -> None:
    print("\n" + "="*60)
    print("  PIPELINE SUMMARY REPORT")
    print("="*60)
    print(f"  Raw data shape            : {df_raw.shape}")
    print(f"  After cleaning shape      : {df_clean.shape}")
    print(f"  Final train features shape: {X_train.shape}")
    print(f"  Final test  features shape: {X_test.shape}")
    print(f"  Missing values remaining  : {pd.DataFrame(X_train).isnull().sum().sum()}")
    print("\n  ✅ ETL Pipeline completed successfully!")
    print("="*60 + "\n")


# ================================================================================
# MAIN — Orchestrate the Full ETL Pipeline
# ================================================================================

if __name__ == "__main__":

    print("\n" + "█"*60)
    print("  CODTECH INTERNSHIP — TASK 1: DATA PIPELINE DEVELOPMENT")
    print("█"*60)

    # ── EXTRACT ──────────────────────────────────────────────────
    df_raw = extract_data(n_samples=500)

    # ── TRANSFORM ────────────────────────────────────────────────
    df_clean      = clean_data(df_raw.copy())
    df_engineered = engineer_features(df_clean)

    X_train, X_test, y_train, y_test, pipeline = transform_data(df_engineered)

    # ── LOAD ─────────────────────────────────────────────────────
    load_data(X_train, X_test, y_train, y_test, output_dir="pipeline_output")

    # ── SUMMARY ──────────────────────────────────────────────────
    print_summary(df_raw, df_clean, X_train, X_test)
