"""
Student Performance Predictor - ML Training and Evaluation Pipeline
Demonstrating the Fundamental Applied Machine Learning Workflow (SDAIA Academy).
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# ==============================================================================
# 1. Dataset Configuration & Loading
# ==============================================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "StudentPerformanceFactors.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Selected Features & Target as defined in project requirements
NUMERICAL_FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Previous_Scores",
    "Sleep_Hours",
    "Tutoring_Sessions"
]

CATEGORICAL_FEATURES = [
    "Motivation_Level",
    "Extracurricular_Activities"
]

SELECTED_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "Exam_Score"


def load_raw_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Loads the original raw dataset from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at: {path}")
    return pd.read_csv(path)


# ==============================================================================
# 2. Understand the Data
# ==============================================================================
def inspect_dataset(df: pd.DataFrame):
    """Prints clear inspection diagnostics for the dataset."""
    print("=" * 70)
    print("STEP 2: UNDERSTAND THE DATA")
    print("=" * 70)
    print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nFirst 5 rows (Selected Features + Target):")
    print(df[SELECTED_FEATURES + [TARGET_COLUMN]].head())

    print("\nFeature Columns & Data Types:")
    for col in SELECTED_FEATURES + [TARGET_COLUMN]:
        print(f" - {col:28s} : {df[col].dtype}")

    print("\nDescriptive Statistics (Numerical Selected Features + Target):")
    print(df[NUMERICAL_FEATURES + [TARGET_COLUMN]].describe().round(2))

    print("\nCategorical Distributions:")
    for col in CATEGORICAL_FEATURES:
        print(f"\nDistribution for {col}:")
        print(df[col].value_counts(dropna=False))


# ==============================================================================
# 3. Data Quality & Cleaning
# ==============================================================================
def clean_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Checks data quality and returns a cleaned copy of the dataset:
    - Verifies missing values
    - Checks duplicate rows
    - Validates target range [0, 100] and handles any unrealistic values
    """
    print("\n" + "=" * 70)
    print("STEP 3: DATA QUALITY & CLEANING")
    print("=" * 70)

    # Work on a dedicated copy to preserve original raw data
    df_clean = df_raw.copy()

    # Missing values check on selected columns
    missing_counts = df_clean[SELECTED_FEATURES + [TARGET_COLUMN]].isnull().sum()
    missing_pct = (missing_counts / len(df_clean)) * 100

    print("Missing Values Check:")
    quality_summary = pd.DataFrame({
        "Missing Count": missing_counts,
        "Percentage (%)": missing_pct.round(2)
    })
    print(quality_summary)

    # Duplicate rows check
    exact_duplicates = df_clean.duplicated().sum()
    selected_duplicates = df_clean.duplicated(subset=SELECTED_FEATURES).sum()
    print(f"\nExact duplicate rows in full dataset: {exact_duplicates}")
    print(f"Duplicate student feature profiles: {selected_duplicates}")

    # Inspect invalid or unrealistic values in Target (Exam Score expected: 0 - 100)
    invalid_target_high = df_clean[df_clean[TARGET_COLUMN] > 100]
    invalid_target_low = df_clean[df_clean[TARGET_COLUMN] < 0]
    print(f"\nOut-of-range target values (Exam_Score > 100): {len(invalid_target_high)}")
    print(f"Out-of-range target values (Exam_Score < 0): {len(invalid_target_low)}")

    if len(invalid_target_high) > 0:
        print("Note: Detected score(s) exceeding 100. Clipping target to maximum valid boundary of 100.")
        df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].clip(0, 100)

    print("Data cleaning completed successfully on independent cleaned copy.")
    return df_clean


# ==============================================================================
# 4. Define Features and Target
# ==============================================================================
def split_features_target(df: pd.DataFrame):
    """Clearly separates X (features) and y (target)."""
    X = df[SELECTED_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


# ==============================================================================
# 5. Split the Data (Train / Validation / Test)
# ==============================================================================
def create_data_splits(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """
    Splits the data into Training (70%), Validation (15%), and Test (15%) sets.
    Uses a fixed random_state for reproducible results.
    """
    print("\n" + "=" * 70)
    print("STEP 5: SPLIT THE DATA (TRAIN / VALIDATION / TEST)")
    print("=" * 70)

    # First split: 70% Train, 30% Temporary (Validation + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state
    )

    # Second split: Split the 30% equally into Validation (15%) and Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state
    )

    print(f"Training set:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X):.1%})")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X):.1%})")
    print(f"Test set:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X):.1%})")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ==============================================================================
# 6. Preprocessing Pipeline
# ==============================================================================
def build_preprocessor() -> ColumnTransformer:
    """
    Builds the ColumnTransformer for numerical and categorical features:
    - Numerical: SimpleImputer(strategy='median') + StandardScaler()
    - Categorical: SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore')
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor


# ==============================================================================
# 7, 8, 9. Baseline, Model Training & Validation Evaluation
# ==============================================================================
def train_and_evaluate_models(X_train, y_train, X_val, y_val, random_state: int = 42):
    """
    Trains Baseline DummyRegressor, Linear Regression, and Random Forest Regressor
    using the shared preprocessing pipeline, and evaluates them on the Validation set.
    """
    print("\n" + "=" * 70)
    print("STEP 7, 8 & 9: BASELINE, MODEL TRAINING & VALIDATION EVALUATION")
    print("=" * 70)

    preprocessor = build_preprocessor()

    # Define candidate models
    candidate_models = {
        "Dummy Regressor (Baseline)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=random_state)
    }

    results = []
    fitted_pipelines = {}

    for name, model in candidate_models.items():
        # Encapsulate preprocessing and estimator in a single clean Pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # Train on Training set only
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Predict on Validation set
        y_val_pred = pipeline.predict(X_val)

        mae = mean_absolute_error(y_val, y_val_pred)
        rmse = root_mean_squared_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)

        results.append({
            "Model Name": name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2 Score": round(r2, 4)
        })

    results_df = pd.DataFrame(results)
    print("\n--- Validation Set Comparison Table ---")
    print(results_df.to_string(index=False))

    return results_df, fitted_pipelines


# ==============================================================================
# 10. Model Selection & Final Test Set Evaluation
# ==============================================================================
def evaluate_best_model_on_test(fitted_pipelines, results_df, X_test, y_test):
    """
    Selects the best performing model (excluding baseline) on the validation set,
    and performs the final unbiased evaluation on the unseen Test set.
    """
    print("\n" + "=" * 70)
    print("STEP 10: MODEL SELECTION & FINAL TEST EVALUATION")
    print("=" * 70)

    # Exclude baseline from selection, pick lowest RMSE on validation set
    valid_models = results_df[~results_df["Model Name"].str.contains("Dummy")].copy()
    best_row = valid_models.sort_values(by="RMSE", ascending=True).iloc[0]
    best_name = best_row["Model Name"]

    print(f"Selected Champion Model based on Validation RMSE: '{best_name}'")
    best_pipeline = fitted_pipelines[best_name]

    # Evaluate once on unseen Test set
    y_test_pred = best_pipeline.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    test_metrics = {
        "Model Name": best_name,
        "Test MAE": round(test_mae, 4),
        "Test RMSE": round(test_rmse, 4),
        "Test R2 Score": round(test_r2, 4)
    }

    test_metrics_df = pd.DataFrame([test_metrics])
    print("\n--- Final Test Set Evaluation (Held-out Unbiased Evaluation) ---")
    print(test_metrics_df.to_string(index=False))

    # Save the champion pipeline for deployment/Streamlit usage
    model_save_path = os.path.join(MODELS_DIR, "best_student_predictor.joblib")
    joblib.dump(best_pipeline, model_save_path)
    print(f"\nChampion model pipeline persisted to: {model_save_path}")

    return best_name, best_pipeline, test_metrics


# ==============================================================================
# Main Runner
# ==============================================================================
def run_full_pipeline():
    """Executes the complete Applied ML Workflow from end to end."""
    print("Starting Student Performance Predictor ML Workflow...")
    df_raw = load_raw_dataset()
    inspect_dataset(df_raw)

    df_clean = clean_dataset(df_raw)
    X, y = split_features_target(df_clean)

    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(X, y, random_state=42)

    val_results_df, fitted_pipelines = train_and_evaluate_models(
        X_train, y_train, X_val, y_val, random_state=42
    )

    best_name, best_pipeline, test_metrics = evaluate_best_model_on_test(
        fitted_pipelines, val_results_df, X_test, y_test
    )

    print("\n" + "=" * 70)
    print("APPLIED ML WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return {
        "val_results": val_results_df,
        "best_name": best_name,
        "test_metrics": test_metrics,
        "best_pipeline": best_pipeline
    }


if __name__ == "__main__":
    run_full_pipeline()
