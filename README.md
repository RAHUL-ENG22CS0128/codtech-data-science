# CODTECH Data Science Internship — Task 1: Data Pipeline Development

## Overview
This project implements a complete **ETL (Extract, Transform, Load)** pipeline for data preprocessing and transformation using **Pandas** and **Scikit-learn**.

---

## Files in This Repository

| File | Description |
|------|-------------|
| `data_pipeline_etl.py` | Main Python script — full ETL pipeline |
| `hr_raw_dataset.csv` | Raw synthetic HR dataset (input) |
| `pipeline_output/train_processed.csv` | Processed training data (generated on run) |
| `pipeline_output/test_processed.csv` | Processed test data (generated on run) |
| `README.md` | Project documentation |

---

## Dataset — `hr_raw_dataset.csv`

A synthetic HR/Employee dataset with **500 rows** and **9 columns**, designed to mimic real-world data quality issues.

| Column | Type | Description |
|--------|------|-------------|
| `employee_id` | int | Unique employee identifier |
| `age` | float | Employee age (has ~30 missing values) |
| `gender` | string | Gender (inconsistent casing: Male/male/FEMALE) |
| `department` | string | Department (has ~80 missing values) |
| `years_exp` | float | Years of experience |
| `salary` | float | Annual salary in USD (has outliers + ~20 missing) |
| `performance` | float | Performance rating 1–5 (has missing values) |
| `satisfaction` | float | Job satisfaction score 1–10 |
| `left_company` | int | **Target variable** — 1 = left, 0 = stayed |

### Intentional Data Issues (for pipeline to handle)
- ~30 missing values in `age`
- ~20 missing values in `salary`
- Missing values in `gender` and `department`
- Inconsistent casing in `gender` (Male / male / FEMALE)
- 5 extreme salary outliers (500,000)

---

## Pipeline Steps

### 1. Extract
- Loads the raw CSV dataset into a Pandas DataFrame

### 2. Transform
**2a. Clean**
- Removes duplicate rows
- Standardises string casing (Title Case)
- Clips salary outliers using the IQR method

**2b. Feature Engineering**
- `salary_per_year_exp` — salary efficiency ratio
- `age_group` — binned age categories (<30, 30-40, 40-50, 50+)
- `high_performer` — binary flag for performance ≥ 4
- `seniority` — experience-based label (Junior / Mid / Senior / Principal)

**2c. Scikit-learn Pipeline**
- `ColumnTransformer` with separate pipelines for numeric and categorical columns
- Numeric: `SimpleImputer(median)` → `StandardScaler`
- Categorical: `SimpleImputer(most_frequent)` → `OneHotEncoder`
- 80/20 train-test split with stratification

### 3. Load
- Saves processed train and test sets as CSV files in `pipeline_output/`

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/codtech-data-science.git
cd codtech-data-science

# 2. Install dependencies
pip install pandas numpy scikit-learn

# 3. Run the pipeline
python data_pipeline_etl.py
```

**Output:**
```
pipeline_output/
├── train_processed.csv   ← 400 rows, processed features
└── test_processed.csv    ← 100 rows, processed features
```

---

## Dependencies

```
pandas
numpy
scikit-learn
```

---

## Internship Details
- **Organization**: CODTECH IT Solutions
- **Task**: Task 1 — Data Pipeline Development
- **Domain**: Data Science
