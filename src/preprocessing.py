"""
preprocessing.py
================
Reusable preprocessing pipeline for the Credit Card Fraud Detection project.

Responsibilities
----------------
1. Load raw CSV data
2. Validate expected columns and data types
3. Remove duplicate rows
4. Handle missing values (imputation strategy per column type)
5. Cap outliers on skewed continuous features (Winsorization)
6. Encode categorical features
   - One-Hot Encoding  : merchant_category
   - Binary flags      : foreign_transaction, location_mismatch (already 0/1)
7. Scale continuous features with StandardScaler
8. Perform stratified train / test split (preserves fraud ratio)
9. Build and return a reusable sklearn Pipeline + preprocessor
10. Save the fitted preprocessor to disk (Joblib) for inference reuse

Usage
-----
Run standalone to preprocess and persist artifacts:
    python src/preprocessing.py

Import in other modules:
    from src.preprocessing import load_data, build_preprocessor, run_preprocessing
"""

import os
import sys
import logging
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Logging configuration
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Project-level path constants
# ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "credit_card_fraud_10k.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")

# ──────────────────────────────────────────────
# Column definitions
# ──────────────────────────────────────────────
TARGET_COL    = "is_fraud"
DROP_COLS     = ["transaction_id"]           # ID — no predictive value

# Continuous features that will be scaled
NUMERIC_COLS  = [
    "amount",
    "transaction_hour",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age",
]

# Binary flags — already 0/1, no encoding needed but kept separate for clarity
BINARY_COLS   = ["foreign_transaction", "location_mismatch"]

# Categorical features to One-Hot Encode
CATEGORICAL_COLS = ["merchant_category"]

# Outlier capping: only skewed continuous columns (not hour / age / scores)
OUTLIER_COLS  = ["amount", "velocity_last_24h"]
OUTLIER_CAP_PERCENTILE = 99   # Cap at 99th percentile (Winsorization)

# Train / test split ratio
TEST_SIZE   = 0.20
RANDOM_STATE = 42


# ══════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════

def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw CSV dataset and perform basic sanity checks.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with all original columns intact.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    ValueError
        If expected columns are missing from the dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            "Please place 'credit_card_fraud_10k.csv' in the data/ directory."
        )

    logger.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)

    # ── Validate expected columns ──────────────────────────────────────────
    expected_cols = DROP_COLS + NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS + [TARGET_COL]
    missing_cols  = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing expected columns: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )

    logger.info(f"Dataset loaded  →  rows: {len(df):,}  |  columns: {df.shape[1]}")
    return df


# ══════════════════════════════════════════════
# 2. DATA QUALITY CHECKS
# ══════════════════════════════════════════════

def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Run a suite of data-quality checks and log results.

    Checks performed
    ----------------
    - Shape
    - Missing values per column
    - Duplicate rows
    - Class distribution of target
    - Basic descriptive statistics

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.

    Returns
    -------
    dict
        Summary dictionary with quality metrics.
    """
    logger.info("Running data quality checks ...")

    summary = {}

    # Shape
    summary["rows"], summary["cols"] = df.shape
    logger.info(f"  Shape           : {summary['rows']:,} rows × {summary['cols']} cols")

    # Missing values
    missing = df.isnull().sum()
    summary["missing_values"] = missing[missing > 0].to_dict()
    if summary["missing_values"]:
        logger.warning(f"  Missing values  : {summary['missing_values']}")
    else:
        logger.info("  Missing values  : None ✓")

    # Duplicates
    summary["duplicate_rows"] = int(df.duplicated().sum())
    if summary["duplicate_rows"] > 0:
        logger.warning(f"  Duplicate rows  : {summary['duplicate_rows']}")
    else:
        logger.info("  Duplicate rows  : None ✓")

    # Class distribution
    counts = df[TARGET_COL].value_counts()
    fraud_rate = df[TARGET_COL].mean() * 100
    summary["class_distribution"] = counts.to_dict()
    summary["fraud_rate_pct"]     = round(fraud_rate, 4)
    logger.info(
        f"  Class dist      : Non-Fraud={counts.get(0, 0):,}  "
        f"Fraud={counts.get(1, 0):,}  "
        f"(fraud rate = {fraud_rate:.2f}%)"
    )

    return summary


# ══════════════════════════════════════════════
# 3. CLEANING
# ══════════════════════════════════════════════

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data cleaning:
      - Drop duplicate rows
      - Fill any missing values (numeric → median, categorical → mode)
      - Enforce correct data types

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    original_len = len(df)
    df = df.copy()

    # ── 3a. Remove duplicates ──────────────────────────────────────────────
    df.drop_duplicates(inplace=True)
    removed = original_len - len(df)
    if removed:
        logger.info(f"  Removed {removed} duplicate rows.")
    else:
        logger.info("  No duplicate rows found ✓")

    # ── 3b. Fill missing values ────────────────────────────────────────────
    # Numeric: fill with column median (robust to skew)
    for col in NUMERIC_COLS + BINARY_COLS:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"  Imputed missing values in '{col}' with median={median_val:.2f}")

    # Categorical: fill with column mode
    for col in CATEGORICAL_COLS:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            logger.info(f"  Imputed missing values in '{col}' with mode='{mode_val}'")

    # ── 3c. Type enforcement ───────────────────────────────────────────────
    for col in BINARY_COLS:
        df[col] = df[col].astype(int)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    logger.info(f"  Cleaning complete  →  {len(df):,} rows remaining")
    return df


# ══════════════════════════════════════════════
# 4. OUTLIER TREATMENT (Winsorization)
# ══════════════════════════════════════════════

def cap_outliers(df: pd.DataFrame, cols: list = OUTLIER_COLS,
                 percentile: int = OUTLIER_CAP_PERCENTILE) -> pd.DataFrame:
    """
    Cap extreme values in specified columns at the given upper percentile
    (Winsorization). Lower values are floored at the 1st percentile.

    Why not remove outliers?
    ------------------------
    In fraud detection, high-value or high-velocity transactions can be
    legitimate. Removing them would lose real signal. Capping preserves
    the distribution shape while dampening extreme skew.

    Parameters
    ----------
    df         : pd.DataFrame  — input data
    cols       : list          — columns to apply capping to
    percentile : int           — upper cap percentile (default 99)

    Returns
    -------
    pd.DataFrame with capped columns.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            logger.warning(f"  Outlier cap: column '{col}' not found, skipping.")
            continue

        lower = df[col].quantile(0.01)
        upper = df[col].quantile(percentile / 100)
        before_max = df[col].max()

        df[col] = df[col].clip(lower=lower, upper=upper)

        logger.info(
            f"  Winsorized '{col}'  →  "
            f"capped [{lower:.2f}, {upper:.2f}]  "
            f"(was max={before_max:.2f})"
        )
    return df


# ══════════════════════════════════════════════
# 5. FEATURE / TARGET SPLIT
# ══════════════════════════════════════════════

def split_features_target(df: pd.DataFrame):
    """
    Separate feature matrix X from target vector y.
    Drops non-predictive columns (transaction_id).

    Returns
    -------
    X : pd.DataFrame  — feature matrix
    y : pd.Series     — binary target (0 = legit, 1 = fraud)
    """
    df = df.copy()
    df.drop(columns=DROP_COLS, errors="ignore", inplace=True)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    logger.info(
        f"  Features: {list(X.columns)}\n"
        f"  Feature matrix shape : {X.shape}\n"
        f"  Target distribution  : {y.value_counts().to_dict()}"
    )
    return X, y


# ══════════════════════════════════════════════
# 6. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════

def split_train_test(X: pd.DataFrame, y: pd.Series,
                     test_size: float = TEST_SIZE,
                     random_state: int = RANDOM_STATE):
    """
    Stratified train/test split.

    Stratification ensures both splits preserve the 1.51% fraud ratio —
    critical for imbalanced datasets. Without stratification, the test set
    could end up with zero or very few fraud cases.

    Parameters
    ----------
    X            : pd.DataFrame  — feature matrix
    y            : pd.Series     — target vector
    test_size    : float         — fraction of data reserved for testing
    random_state : int           — reproducibility seed

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y           # Preserve class ratio in both splits
    )

    logger.info(
        f"  Train split  →  {len(X_train):,} rows  "
        f"| fraud={y_train.sum()} ({y_train.mean()*100:.2f}%)"
    )
    logger.info(
        f"  Test split   →  {len(X_test):,} rows  "
        f"| fraud={y_test.sum()} ({y_test.mean()*100:.2f}%)"
    )
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════
# 7. SKLEARN PIPELINE / COLUMN TRANSFORMER
# ══════════════════════════════════════════════

def build_preprocessor() -> ColumnTransformer:
    """
    Build a reusable sklearn ColumnTransformer that handles:

    Numeric pipeline
    ----------------
    Step 1 – SimpleImputer(strategy='median')
        Safety net: handles any residual NaN values that slipped through
        the manual cleaning step (unlikely but defensive).
    Step 2 – StandardScaler()
        Zero-mean, unit-variance scaling.
        Important for Logistic Regression and SVM which are sensitive to
        feature magnitude. Tree-based models (RF, XGBoost) don't strictly
        need scaling but it doesn't hurt them.

    Categorical pipeline
    --------------------
    Step 1 – SimpleImputer(strategy='most_frequent')
        Safety net for NaN in categorical columns.
    Step 2 – OneHotEncoder(drop='first', handle_unknown='ignore')
        drop='first' avoids the dummy variable trap (multicollinearity).
        handle_unknown='ignore' ensures unseen categories at inference time
        don't crash the pipeline — they simply map to the all-zeros vector.

    Binary columns (foreign_transaction, location_mismatch)
    --------------------------------------------------------
    Passed through as-is ('passthrough') — they are already 0/1 integers
    and require no further transformation. Imputer is still applied as a
    safety net.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor ready to be used inside a Pipeline.
    """

    # ── Numeric pipeline ───────────────────────────────────────────────────
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # ── Categorical pipeline ───────────────────────────────────────────────
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    # ── Binary passthrough pipeline ────────────────────────────────────────
    binary_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    # ── Combine into ColumnTransformer ─────────────────────────────────────
    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  numeric_pipeline,     NUMERIC_COLS),
            ("cat",  categorical_pipeline, CATEGORICAL_COLS),
            ("bin",  binary_pipeline,      BINARY_COLS),
        ],
        remainder="drop",    # Drop any columns not explicitly listed
        verbose_feature_names_out=False,
    )

    logger.info("  Preprocessor (ColumnTransformer) built successfully.")
    return preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer) -> list:
    """
    Extract human-readable output feature names after fitting the preprocessor.

    One-Hot Encoding expands merchant_category into multiple columns, so the
    final feature list differs from the input list. This helper provides the
    correct ordered names for downstream analysis (feature importance, etc.).

    Parameters
    ----------
    preprocessor : fitted ColumnTransformer

    Returns
    -------
    list of str — ordered output feature names
    """
    try:
        names = list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback: manually reconstruct names
        ohe = preprocessor.named_transformers_["cat"]["onehot"]
        cat_names = [
            f"{CATEGORICAL_COLS[0]}_{c}"
            for c in ohe.categories_[0][1:]   # drop='first' removes the first category
        ]
        names = NUMERIC_COLS + cat_names + BINARY_COLS

    return names


# ══════════════════════════════════════════════
# 8. SAVE / LOAD PREPROCESSOR
# ══════════════════════════════════════════════

def save_preprocessor(preprocessor: ColumnTransformer,
                      path: str = PREPROCESSOR_PATH) -> None:
    """
    Persist the fitted ColumnTransformer to disk using Joblib.

    The preprocessor must be saved after fitting on the training data so
    that inference (predict.py / Streamlit app) applies exactly the same
    transformations as training — using training-set statistics (mean, std,
    OHE categories) rather than recomputing on new data.

    Parameters
    ----------
    preprocessor : fitted ColumnTransformer
    path         : str — destination .pkl file path
    """
    joblib.dump(preprocessor, path)
    logger.info(f"  Preprocessor saved  →  {path}")


def load_preprocessor(path: str = PREPROCESSOR_PATH) -> ColumnTransformer:
    """
    Load a previously saved fitted ColumnTransformer from disk.

    Parameters
    ----------
    path : str — path to the saved .pkl file

    Returns
    -------
    ColumnTransformer (fitted)

    Raises
    ------
    FileNotFoundError if the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Preprocessor not found at: {path}\n"
            "Run preprocessing.py first to generate it."
        )
    preprocessor = joblib.load(path)
    logger.info(f"  Preprocessor loaded  ←  {path}")
    return preprocessor


# ══════════════════════════════════════════════
# 9. MASTER PIPELINE RUNNER
# ══════════════════════════════════════════════

def run_preprocessing(data_path: str = DATA_PATH,
                      save_artifacts: bool = True):
    """
    End-to-end preprocessing pipeline.

    Executes all steps in order:
      load → quality check → clean → cap outliers →
      split features/target → train/test split →
      build & fit preprocessor → transform data →
      (optionally) save preprocessor

    Parameters
    ----------
    data_path       : str  — path to raw CSV
    save_artifacts  : bool — whether to save the fitted preprocessor

    Returns
    -------
    dict with keys:
        X_train_proc  : np.ndarray — preprocessed training features
        X_test_proc   : np.ndarray — preprocessed test features
        y_train       : pd.Series  — training labels
        y_test        : pd.Series  — test labels
        X_train_raw   : pd.DataFrame — unprocessed training features (for EDA)
        X_test_raw    : pd.DataFrame — unprocessed test features
        preprocessor  : fitted ColumnTransformer
        feature_names : list of str — output feature names after OHE expansion
        quality_summary : dict     — data quality check results
    """
    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE — START")
    logger.info("=" * 60)

    # Step 1 — Load
    df = load_data(data_path)

    # Step 2 — Quality checks
    quality_summary = check_data_quality(df)

    # Step 3 — Clean
    logger.info("\n[Step 3] Cleaning data ...")
    df = clean_data(df)

    # Step 4 — Cap outliers
    logger.info("\n[Step 4] Capping outliers ...")
    df = cap_outliers(df)

    # Step 5 — Feature / target split
    logger.info("\n[Step 5] Splitting features and target ...")
    X, y = split_features_target(df)

    # Step 6 — Train / test split
    logger.info("\n[Step 6] Train / test split ...")
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Step 7 — Build preprocessor
    logger.info("\n[Step 7] Building preprocessing pipeline ...")
    preprocessor = build_preprocessor()

    # Step 8 — Fit on training data ONLY (prevent data leakage)
    logger.info("\n[Step 8] Fitting preprocessor on training data ...")
    X_train_proc = preprocessor.fit_transform(X_train)
    logger.info(
        f"  X_train_proc shape : {X_train_proc.shape}  "
        f"(expanded from {X_train.shape[1]} raw features)"
    )

    # Step 9 — Transform test data using training statistics
    logger.info("\n[Step 9] Transforming test data ...")
    X_test_proc = preprocessor.transform(X_test)
    logger.info(f"  X_test_proc shape  : {X_test_proc.shape}")

    # Step 10 — Extract feature names
    feature_names = get_feature_names_out(preprocessor)
    logger.info(f"\n  Final feature names ({len(feature_names)}):")
    for i, name in enumerate(feature_names, 1):
        logger.info(f"    {i:>2}. {name}")

    # Step 11 — Save artifacts
    if save_artifacts:
        logger.info("\n[Step 10] Saving preprocessor ...")
        save_preprocessor(preprocessor)

    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING PIPELINE — COMPLETE")
    logger.info("=" * 60)

    return {
        "X_train_proc"    : X_train_proc,
        "X_test_proc"     : X_test_proc,
        "y_train"         : y_train,
        "y_test"          : y_test,
        "X_train_raw"     : X_train,
        "X_test_raw"      : X_test,
        "preprocessor"    : preprocessor,
        "feature_names"   : feature_names,
        "quality_summary" : quality_summary,
    }


# ══════════════════════════════════════════════
# MAIN — standalone execution
# ══════════════════════════════════════════════

if __name__ == "__main__":
    results = run_preprocessing()

    print("\n" + "─" * 50)
    print("PREPROCESSING RESULTS SUMMARY")
    print("─" * 50)
    print(f"  X_train shape  : {results['X_train_proc'].shape}")
    print(f"  X_test  shape  : {results['X_test_proc'].shape}")
    print(f"  y_train fraud  : {results['y_train'].sum()} / {len(results['y_train'])}")
    print(f"  y_test  fraud  : {results['y_test'].sum()} / {len(results['y_test'])}")
    print(f"  Feature names  : {results['feature_names']}")
    print(f"  Preprocessor   : saved to models/preprocessor.pkl")
    print("─" * 50)
    print("Ready for Phase 5: Feature Engineering")
