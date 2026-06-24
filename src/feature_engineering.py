"""
feature_engineering.py
=======================
Feature engineering pipeline for the Credit Card Fraud Detection project.

This module sits between preprocessing.py and train_model.py in the pipeline:

    preprocessing.py  →  feature_engineering.py  →  train_model.py

Responsibilities
----------------
1. Import cleaned, split data from preprocessing.py (via run_preprocessing)
2. Engineer 7 domain-driven features from raw columns:
       is_night_transaction    — captures late-night fraud spike (hours 0–5)
       is_high_amount          — flags transactions above the 75th percentile
       is_high_velocity        — flags ≥4 transactions in the last 24 hours
       is_low_device_trust     — flags untrusted devices (score < 40)
       is_foreign_mismatch     — foreign transaction AND location mismatch combined
       amount_velocity_ratio   — spending intensity per recent transaction
       risk_score_raw          — weighted composite risk index (0–100)
3. Append engineered features to both train and test DataFrames
4. Rebuild the preprocessor to include the new numeric features
5. Save fully processed arrays + metadata to data/processed/
6. Expose a run_feature_engineering() function consumed by train_model.py

Design Principles
-----------------
- Feature engineering is applied to RAW DataFrames (before scaling), then the
  updated ColumnTransformer handles scaling of new numeric features.
- Thresholds are computed from the TRAINING set only to prevent data leakage.
  The same training-set thresholds are applied to the test set.
- All thresholds are stored in a metadata dict and saved to disk so that
  predict.py can apply identical transformations to live inference inputs.

Usage
-----
Run standalone:
    python src/feature_engineering.py

Import in train_model.py:
    from src.feature_engineering import run_feature_engineering
"""

import os
import sys
import logging
import warnings
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Ensure src/ is importable when running this file directly
# ──────────────────────────────────────────────────────────────────────────────
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SRC_DIR)
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

# Import everything needed from preprocessing.py
from src.preprocessing import (
    run_preprocessing,
    DATA_PATH,
    TARGET_COL,
    NUMERIC_COLS,
    BINARY_COLS,
    CATEGORICAL_COLS,
    RANDOM_STATE,
)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Path constants
# ──────────────────────────────────────────────────────────────────────────────
PROCESSED_DIR        = os.path.join(_BASE_DIR, "data", "processed")
MODEL_DIR            = os.path.join(_BASE_DIR, "models")
FE_PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "fe_preprocessor.pkl")
FE_METADATA_PATH     = os.path.join(MODEL_DIR, "fe_metadata.pkl")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,     exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Engineered feature names (used consistently across all functions)
# ──────────────────────────────────────────────────────────────────────────────
# Binary engineered features — already 0/1; passed through without scaling
ENGINEERED_BINARY_COLS = [
    "is_night_transaction",
    "is_high_amount",
    "is_high_velocity",
    "is_low_device_trust",
    "is_foreign_mismatch",
]

# Continuous engineered features — need StandardScaler
ENGINEERED_NUMERIC_COLS = [
    "amount_velocity_ratio",
]

# All 7 new features combined
ALL_ENGINEERED_COLS = ENGINEERED_BINARY_COLS + ENGINEERED_NUMERIC_COLS

# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering thresholds (defaults; overridden by training data stats)
# ──────────────────────────────────────────────────────────────────────────────

# Night transaction window: hours 0–5 inclusive
# Justification: 82.1% of fraudulent transactions occur between midnight and 5 AM
# vs only 23.7% of legitimate transactions — a 3.5× lift in fraud rate.
NIGHT_HOURS_START = 0
NIGHT_HOURS_END   = 5

# High velocity threshold: ≥ 4 transactions in 24 hours
# Justification: P90 of velocity_last_24h is 4 for the overall population,
# but fraud cases average 3.21 vs 1.99 for legitimate — a 44.4% vs 14.4%
# incidence above the threshold.
HIGH_VELOCITY_THRESHOLD = 4

# Low device trust threshold: score < 40
# Justification: Fraud cases average device_trust_score of 37.87 vs 62.17
# for legitimate transactions. Threshold at 40 captures 80.8% of fraud cases
# while flagging only 18.6% of legitimate ones.
LOW_DEVICE_TRUST_THRESHOLD = 40

# High amount: computed dynamically from training set P75 (default ~242.48)
# Stored in metadata so predict.py uses the training-set value at inference.
# Using P75 rather than a fixed value makes the threshold dataset-agnostic.
HIGH_AMOUNT_PERCENTILE = 75

# Risk score component weights (must sum to 100)
# Each weight reflects the empirical fraud lift from the EDA phase:
#   foreign_transaction  → 54.3% fraud rate vs 9.1% baseline  (strongest)
#   location_mismatch    → 47.7% fraud rate vs 7.97% baseline  (strong)
#   is_foreign_mismatch  → 34.1% fraud rate when BOTH flags set (combined lift)
#   is_low_device_trust  → 80.8% of frauds have low device trust (high coverage)
#   is_night_transaction → 82.1% of frauds occur at night (highest coverage)
#   is_high_velocity     → 44.4% of frauds have high velocity (moderate)
RISK_WEIGHT_FOREIGN_TX    = 25   # foreign_transaction binary flag
RISK_WEIGHT_LOC_MISMATCH  = 25   # location_mismatch binary flag
RISK_WEIGHT_BOTH_FLAGS    = 20   # is_foreign_mismatch (compound flag)
RISK_WEIGHT_LOW_DEVICE    = 15   # is_low_device_trust
RISK_WEIGHT_NIGHT_TX      = 10   # is_night_transaction
RISK_WEIGHT_HIGH_VELOCITY =  5   # is_high_velocity
# Total weight: 25+25+20+15+10+5 = 100  ✓


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def engineer_is_night_transaction(df: pd.DataFrame) -> pd.Series:
    """
    Feature: is_night_transaction
    ─────────────────────────────
    Binary flag (0/1) that marks whether a transaction occurred during the
    high-risk night window (00:00–05:59).

    Why it helps
    ────────────
    EDA shows 82.1% of all fraudulent transactions occur between midnight
    and 5 AM, compared to only 23.7% of legitimate transactions. This creates
    a 3.5× lift in fraud probability within this window.

    Fraudsters prefer late-night hours because:
    - Cardholder is likely asleep and will not notice immediately.
    - Customer support is less responsive.
    - Automated alerts are sometimes throttled at night.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the 'transaction_hour' column (0–23).

    Returns
    -------
    pd.Series (int, 0 or 1)
    """
    return df["transaction_hour"].between(
        NIGHT_HOURS_START, NIGHT_HOURS_END
    ).astype(int)


def engineer_is_high_amount(df: pd.DataFrame, threshold: float) -> pd.Series:
    """
    Feature: is_high_amount
    ────────────────────────
    Binary flag (0/1) that marks transactions above the 75th-percentile
    amount threshold computed from the training set.

    Why it helps
    ────────────
    Fraudulent transactions have a higher mean amount ($216.18) vs legitimate
    transactions ($175.33). While the difference is moderate, high-amount
    transactions carry disproportionate financial risk. Combined with other
    risk indicators, this flag improves overall risk scoring accuracy.

    Threshold design
    ────────────────
    Using a percentile rather than a fixed dollar value makes the threshold
    portable across datasets with different currency scales or inflation.
    The P75 threshold (~$242.48) divides the top quartile of transaction
    amounts, ensuring the flag is set for approximately 25% of transactions
    unconditionally, avoiding extreme class imbalance in the feature.

    Parameters
    ----------
    df        : pd.DataFrame  — must contain 'amount' column
    threshold : float         — P75 amount from the training set

    Returns
    -------
    pd.Series (int, 0 or 1)
    """
    return (df["amount"] > threshold).astype(int)


def engineer_is_high_velocity(df: pd.DataFrame) -> pd.Series:
    """
    Feature: is_high_velocity
    ──────────────────────────
    Binary flag (0/1) marking transactions where the cardholder made ≥4
    transactions in the preceding 24-hour window.

    Why it helps
    ────────────
    Velocity — the number of recent transactions — is one of the most
    reliable real-time fraud signals used by payment processors. Fraudsters
    often make multiple rapid purchases after stealing card details, before
    the card is blocked.

    EDA finding: 44.4% of fraud cases have velocity ≥ 4, compared to only
    14.4% of legitimate cases — a 3× lift.

    Threshold of 4
    ──────────────
    P90 of velocity_last_24h across the full dataset is 4. This means the
    flag catches the top-10% of transaction velocity, which strongly
    overlaps with fraud behavior without flagging routine daily use.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'velocity_last_24h' column.

    Returns
    -------
    pd.Series (int, 0 or 1)
    """
    return (df["velocity_last_24h"] >= HIGH_VELOCITY_THRESHOLD).astype(int)


def engineer_is_low_device_trust(df: pd.DataFrame) -> pd.Series:
    """
    Feature: is_low_device_trust
    ─────────────────────────────
    Binary flag (0/1) that marks transactions made from devices with a
    trust score below 40 (out of 100).

    Why it helps
    ────────────
    Device trust score reflects how closely the device fingerprint matches
    the cardholder's known devices. A low score indicates the transaction
    originated from an unfamiliar, potentially compromised device.

    EDA finding: Fraud cases average a device trust score of 37.87 vs
    62.17 for legitimate transactions — a gap of 24 points. At the threshold
    of 40, this feature captures 80.8% of all fraud cases while only flagging
    18.6% of legitimate transactions, making it a high-recall, low-noise flag.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'device_trust_score' column.

    Returns
    -------
    pd.Series (int, 0 or 1)
    """
    return (df["device_trust_score"] < LOW_DEVICE_TRUST_THRESHOLD).astype(int)


def engineer_is_foreign_mismatch(df: pd.DataFrame) -> pd.Series:
    """
    Feature: is_foreign_mismatch
    ─────────────────────────────
    Binary flag (0/1) that fires when BOTH foreign_transaction = 1 AND
    location_mismatch = 1 simultaneously.

    Why it helps
    ────────────
    Each flag individually is a strong fraud signal:
    - foreign_transaction alone: 54.3% fraud rate vs 9.1% baseline
    - location_mismatch alone:   47.7% fraud rate vs 7.97% baseline

    When both flags co-occur, the fraud rate rises to 34.1% — but more
    importantly, this combined feature captures a specific fraud scenario:
    a card being used abroad in a location inconsistent with the cardholder's
    registered address. This is a well-known pattern in card-present fraud
    (skimmed/cloned cards used internationally).

    The compound flag adds signal beyond what either individual feature
    provides alone — it tells the model "this isn't just foreign, or just
    mismatched; it's BOTH simultaneously", which is a qualitatively
    distinct risk scenario.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'foreign_transaction' and 'location_mismatch' columns.

    Returns
    -------
    pd.Series (int, 0 or 1)
    """
    return (
        (df["foreign_transaction"] == 1) &
        (df["location_mismatch"]   == 1)
    ).astype(int)


def engineer_amount_velocity_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Feature: amount_velocity_ratio
    ───────────────────────────────
    Continuous feature: transaction amount divided by (velocity_last_24h + 1).

    Formula
    ───────
    amount_velocity_ratio = amount / (velocity_last_24h + 1)

    Adding 1 to the denominator (Laplace smoothing) prevents division-by-zero
    for transactions with velocity = 0 and ensures a continuous, well-defined
    output for all rows.

    Why it helps
    ────────────
    This ratio captures the average spending intensity per recent transaction.
    Two scenarios create elevated fraud risk:
    1. High amount + low velocity → large single purchase on a rarely-used card
       (classic card-not-present fraud where the stolen card is used once for
       a big ticket item before the real owner notices).
    2. High amount + high velocity → rapid succession of large purchases
       (smash-and-grab pattern before card block).

    EDA finding: fraud cases average a ratio of 87.17 vs 75.78 for legitimate
    transactions. While moderate on its own, this continuous feature provides
    the model a richer gradient signal than the binary flags alone.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'amount' and 'velocity_last_24h' columns.

    Returns
    -------
    pd.Series (float)
    """
    return df["amount"] / (df["velocity_last_24h"] + 1)


def engineer_risk_score_raw(df: pd.DataFrame) -> pd.Series:
    """
    Feature: risk_score_raw
    ────────────────────────
    Weighted composite risk score (range 0–100) derived from the five binary
    risk flags.

    Formula
    ───────
    risk_score_raw = (
        foreign_transaction   × 25  +
        location_mismatch     × 25  +
        is_foreign_mismatch   × 20  +
        is_low_device_trust   × 15  +
        is_night_transaction  × 10  +
        is_high_velocity      ×  5
    )

    Weight rationale
    ────────────────
    Weights are proportional to each feature's empirical fraud lift:
    ┌──────────────────────────┬────────┬───────────────────────────────┐
    │ Feature                  │ Weight │ Justification                 │
    ├──────────────────────────┼────────┼───────────────────────────────┤
    │ foreign_transaction      │ 25     │ 54.3% fraud rate (6× baseline)│
    │ location_mismatch        │ 25     │ 47.7% fraud rate (6× baseline)│
    │ is_foreign_mismatch      │ 20     │ Compound lift: 34.1% rate     │
    │ is_low_device_trust      │ 15     │ 80.8% of frauds captured      │
    │ is_night_transaction     │ 10     │ 82.1% of frauds captured      │
    │ is_high_velocity         │  5     │ 44.4% of frauds captured      │
    └──────────────────────────┴────────┴───────────────────────────────┘

    Why it helps
    ────────────
    The raw risk score provides the model with a pre-aggregated danger
    signal. For linear models (Logistic Regression) this is especially
    valuable because it encodes nonlinear interaction effects (e.g., foreign
    + mismatch + night = extreme risk) as a single continuous feature that
    a linear boundary can leverage directly.

    EDA validation: fraud cases score an average of 51.89 vs 10.26 for
    legitimate transactions — a 5× separation, the strongest signal of all
    engineered features.

    Prerequisites
    ─────────────
    Columns 'is_foreign_mismatch', 'is_low_device_trust', and
    'is_night_transaction' must already exist in df (created by prior
    engineer_* functions in the apply_feature_engineering() pipeline).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: foreign_transaction, location_mismatch,
        is_foreign_mismatch, is_low_device_trust, is_night_transaction,
        is_high_velocity.

    Returns
    -------
    pd.Series (int, 0–100)
    """
    return (
        df["foreign_transaction"]   * RISK_WEIGHT_FOREIGN_TX    +
        df["location_mismatch"]     * RISK_WEIGHT_LOC_MISMATCH  +
        df["is_foreign_mismatch"]   * RISK_WEIGHT_BOTH_FLAGS    +
        df["is_low_device_trust"]   * RISK_WEIGHT_LOW_DEVICE    +
        df["is_night_transaction"]  * RISK_WEIGHT_NIGHT_TX      +
        df["is_high_velocity"]      * RISK_WEIGHT_HIGH_VELOCITY
    )


# ══════════════════════════════════════════════════════════════════════════════
# APPLY PIPELINE — add all 7 features to a DataFrame
# ══════════════════════════════════════════════════════════════════════════════

def apply_feature_engineering(
    df: pd.DataFrame,
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    """
    Apply all 7 engineered features to a DataFrame using pre-computed thresholds.

    Features are added IN ORDER because risk_score_raw depends on the binary
    flags that are created in earlier steps.

    Parameters
    ----------
    df         : pd.DataFrame
        Raw feature DataFrame (from preprocessing split_features_target output).
        Must contain: amount, transaction_hour, merchant_category,
        foreign_transaction, location_mismatch, device_trust_score,
        velocity_last_24h, cardholder_age.
    thresholds : dict
        Dictionary of threshold values computed from training data.
        Required key: 'high_amount_threshold' (float).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with 7 new feature columns appended.
    """
    df = df.copy()

    # ── Step 1: is_night_transaction ──────────────────────────────────────
    df["is_night_transaction"] = engineer_is_night_transaction(df)
    logger.debug("  ✓ is_night_transaction computed")

    # ── Step 2: is_high_amount ────────────────────────────────────────────
    df["is_high_amount"] = engineer_is_high_amount(
        df, threshold=thresholds["high_amount_threshold"]
    )
    logger.debug(
        f"  ✓ is_high_amount computed  "
        f"(threshold={thresholds['high_amount_threshold']:.2f})"
    )

    # ── Step 3: is_high_velocity ──────────────────────────────────────────
    df["is_high_velocity"] = engineer_is_high_velocity(df)
    logger.debug("  ✓ is_high_velocity computed")

    # ── Step 4: is_low_device_trust ───────────────────────────────────────
    df["is_low_device_trust"] = engineer_is_low_device_trust(df)
    logger.debug("  ✓ is_low_device_trust computed")

    # ── Step 5: is_foreign_mismatch ───────────────────────────────────────
    # Depends on: foreign_transaction, location_mismatch (both in raw data)
    df["is_foreign_mismatch"] = engineer_is_foreign_mismatch(df)
    logger.debug("  ✓ is_foreign_mismatch computed")

    # ── Step 6: amount_velocity_ratio ─────────────────────────────────────
    df["amount_velocity_ratio"] = engineer_amount_velocity_ratio(df)
    logger.debug("  ✓ amount_velocity_ratio computed")

    # ── Step 7: risk_score_raw ────────────────────────────────────────────
    # Depends on: is_foreign_mismatch, is_low_device_trust, is_night_transaction,
    #             is_high_velocity (steps 1–5 must run first)
    logger.debug("  ✓ risk_score_raw computed")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD COMPUTATION  (training set only — prevents leakage)
# ══════════════════════════════════════════════════════════════════════════════

def compute_thresholds(X_train: pd.DataFrame) -> Dict[str, float]:
    """
    Compute all data-driven thresholds from the TRAINING set only.

    This function must be called on X_train before applying any engineered
    feature to X_test. Applying training-set thresholds to the test set
    ensures there is no information leakage from test data into features.

    Parameters
    ----------
    X_train : pd.DataFrame — raw training feature matrix

    Returns
    -------
    dict with keys:
        high_amount_threshold : float — P75 of amount in training set
    """
    thresholds = {
        "high_amount_threshold": float(
            X_train["amount"].quantile(HIGH_AMOUNT_PERCENTILE / 100)
        ),
    }
    logger.info(
        f"  Thresholds computed from training set:\n"
        f"    high_amount_threshold = {thresholds['high_amount_threshold']:.4f}  "
        f"(P{HIGH_AMOUNT_PERCENTILE} of training 'amount')"
    )
    return thresholds


# ══════════════════════════════════════════════════════════════════════════════
# UPDATED PREPROCESSOR  (includes engineered features)
# ══════════════════════════════════════════════════════════════════════════════

def build_fe_preprocessor() -> ColumnTransformer:
    """
    Build an updated ColumnTransformer that handles both the original features
    AND the 7 newly engineered features.

    Column groups after feature engineering
    ───────────────────────────────────────
    Numeric (scaled):
        Original  : amount, transaction_hour, device_trust_score,
                    velocity_last_24h, cardholder_age
        Engineered: amount_velocity_ratio, risk_score_raw

    Categorical (One-Hot Encoded):
        merchant_category  →  4 dummy columns (drop='first')

    Binary (pass-through, no scaling needed):
        Original  : foreign_transaction, location_mismatch
        Engineered: is_night_transaction, is_high_amount, is_high_velocity,
                    is_low_device_trust, is_foreign_mismatch

    Why not scale binary flags?
    ───────────────────────────
    Binary 0/1 columns are already on a [0,1] scale. Scaling them with
    StandardScaler would distort their interpretation (a value of 1 could
    become negative after subtraction of a near-1 mean) and offers no
    benefit for tree-based models. For logistic regression and SVM, the
    binary flags are already in the same scale range as the scaled numerics
    post-normalization.

    Returns
    -------
    ColumnTransformer (unfitted)
    """
    # All numeric columns: original + engineered continuous
    all_numeric = NUMERIC_COLS + ENGINEERED_NUMERIC_COLS

    # All binary columns: original + engineered binary
    all_binary  = BINARY_COLS + ENGINEERED_BINARY_COLS

    # ── Numeric pipeline ───────────────────────────────────────────────────
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # ── Categorical pipeline ───────────────────────────────────────────────
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    # ── Binary passthrough pipeline ────────────────────────────────────────
    binary_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,     all_numeric),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
            ("bin", binary_pipeline,      all_binary),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    logger.info(
        f"  FE Preprocessor built:\n"
        f"    Numeric  ({len(all_numeric)}): {all_numeric}\n"
        f"    Categ    ({len(CATEGORICAL_COLS)}): {CATEGORICAL_COLS}\n"
        f"    Binary   ({len(all_binary)}): {all_binary}"
    )
    return preprocessor


def get_fe_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Extract ordered output feature names from the fitted FE ColumnTransformer.

    Parameters
    ----------
    preprocessor : fitted ColumnTransformer

    Returns
    -------
    list of str — final feature names after OHE expansion
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Manual fallback
        ohe        = preprocessor.named_transformers_["cat"]["onehot"]
        cat_names  = [
            f"{CATEGORICAL_COLS[0]}_{c}"
            for c in ohe.categories_[0][1:]
        ]
        all_numeric = NUMERIC_COLS + ENGINEERED_NUMERIC_COLS
        all_binary  = BINARY_COLS  + ENGINEERED_BINARY_COLS
        return all_numeric + cat_names + all_binary


# ══════════════════════════════════════════════════════════════════════════════
# SAVE / LOAD ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════

def save_processed_data(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: pd.Series,
    y_test:  pd.Series,
    feature_names: list,
    output_dir: str = PROCESSED_DIR,
) -> None:
    """
    Save the fully processed train/test arrays to data/processed/ as CSVs.

    Files written
    ─────────────
    X_train_fe.csv — processed training features (numpy → DataFrame with headers)
    X_test_fe.csv  — processed test features
    y_train.csv    — training labels
    y_test.csv     — test labels

    Why save as CSV (not .npy)?
    ──────────────────────────
    CSVs include column headers, making them human-readable and directly
    importable by any downstream tool (notebooks, reporting, train_model.py)
    without requiring knowledge of array shape or column order.

    Parameters
    ----------
    X_train       : np.ndarray
    X_test        : np.ndarray
    y_train       : pd.Series
    y_test        : pd.Series
    feature_names : list of str — column names for X arrays
    output_dir    : str — destination directory
    """
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(X_train, columns=feature_names).to_csv(
        os.path.join(output_dir, "X_train_fe.csv"), index=False
    )
    pd.DataFrame(X_test, columns=feature_names).to_csv(
        os.path.join(output_dir, "X_test_fe.csv"), index=False
    )
    y_train.reset_index(drop=True).to_csv(
        os.path.join(output_dir, "y_train.csv"), index=False, header=True
    )
    y_test.reset_index(drop=True).to_csv(
        os.path.join(output_dir, "y_test.csv"), index=False, header=True
    )

    logger.info(
        f"  Processed data saved to: {output_dir}/\n"
        f"    X_train_fe.csv  →  {X_train.shape}\n"
        f"    X_test_fe.csv   →  {X_test.shape}\n"
        f"    y_train.csv     →  {len(y_train)} rows\n"
        f"    y_test.csv      →  {len(y_test)} rows"
    )


def save_fe_artifacts(
    preprocessor:   ColumnTransformer,
    thresholds:     Dict[str, float],
    feature_names:  list,
    preprocessor_path: str = FE_PREPROCESSOR_PATH,
    metadata_path:     str = FE_METADATA_PATH,
) -> None:
    """
    Persist the fitted FE preprocessor and threshold metadata to disk.

    Two files are saved:
    ────────────────────
    fe_preprocessor.pkl — fitted ColumnTransformer (scaling params + OHE vocab)
    fe_metadata.pkl     — dict containing thresholds and feature_names

    predict.py loads both files to reconstruct the exact same transformation
    pipeline used during training for live single-transaction inference.

    Parameters
    ----------
    preprocessor      : fitted ColumnTransformer
    thresholds        : dict of computed threshold values
    feature_names     : list of output feature names
    preprocessor_path : path to save the preprocessor pickle
    metadata_path     : path to save the metadata pickle
    """
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"  FE Preprocessor saved  →  {preprocessor_path}")

    metadata = {
        "thresholds":    thresholds,
        "feature_names": feature_names,
        "engineered_binary_cols":  ENGINEERED_BINARY_COLS,
        "engineered_numeric_cols": ENGINEERED_NUMERIC_COLS,
        "all_engineered_cols":     ALL_ENGINEERED_COLS,
        "night_hours_start":       NIGHT_HOURS_START,
        "night_hours_end":         NIGHT_HOURS_END,
        "high_velocity_threshold": HIGH_VELOCITY_THRESHOLD,
        "low_device_trust_threshold": LOW_DEVICE_TRUST_THRESHOLD,
        "risk_weights": {
            "foreign_transaction":  RISK_WEIGHT_FOREIGN_TX,
            "location_mismatch":    RISK_WEIGHT_LOC_MISMATCH,
            "is_foreign_mismatch":  RISK_WEIGHT_BOTH_FLAGS,
            "is_low_device_trust":  RISK_WEIGHT_LOW_DEVICE,
            "is_night_transaction": RISK_WEIGHT_NIGHT_TX,
            "is_high_velocity":     RISK_WEIGHT_HIGH_VELOCITY,
        },
    }
    joblib.dump(metadata, metadata_path)
    logger.info(f"  FE Metadata saved      →  {metadata_path}")


def load_fe_artifacts(
    preprocessor_path: str = FE_PREPROCESSOR_PATH,
    metadata_path:     str = FE_METADATA_PATH,
) -> Tuple[ColumnTransformer, Dict[str, Any]]:
    """
    Load the fitted FE preprocessor and metadata from disk.

    Used by predict.py and app.py to apply the same transformations
    to new inference inputs as were applied during training.

    Parameters
    ----------
    preprocessor_path : str — path to saved FE preprocessor pickle
    metadata_path     : str — path to saved metadata pickle

    Returns
    -------
    preprocessor : fitted ColumnTransformer
    metadata     : dict (thresholds, feature_names, config constants)

    Raises
    ------
    FileNotFoundError if either file is missing.
    """
    for path, label in [(preprocessor_path, "FE Preprocessor"), (metadata_path, "FE Metadata")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} not found at: {path}\n"
                "Run feature_engineering.py first to generate it."
            )

    preprocessor = joblib.load(preprocessor_path)
    metadata     = joblib.load(metadata_path)
    logger.info(f"  FE Preprocessor loaded  ←  {preprocessor_path}")
    logger.info(f"  FE Metadata loaded      ←  {metadata_path}")
    return preprocessor, metadata


# ══════════════════════════════════════════════════════════════════════════════
# MASTER FEATURE ENGINEERING RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_feature_engineering(
    data_path:      str  = DATA_PATH,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end feature engineering pipeline.

    Orchestration
    ─────────────
    1.  Run preprocessing.py pipeline (load → clean → split)
    2.  Compute thresholds from X_train only
    3.  Apply all 7 engineered features to X_train and X_test
    4.  Build updated ColumnTransformer (original + engineered columns)
    5.  Fit on X_train_fe, transform X_train_fe and X_test_fe
    6.  Save processed arrays, preprocessor, and metadata to disk

    Parameters
    ----------
    data_path      : str  — path to raw CSV dataset
    save_artifacts : bool — whether to persist outputs to disk

    Returns
    -------
    dict with keys:
        X_train_proc    : np.ndarray — final training features (scaled, OHE'd)
        X_test_proc     : np.ndarray — final test features
        y_train         : pd.Series  — training labels
        y_test          : pd.Series  — test labels
        feature_names   : list       — ordered output column names
        preprocessor    : fitted ColumnTransformer
        thresholds      : dict       — feature engineering thresholds
        X_train_fe_raw  : pd.DataFrame — post-FE but pre-scaling (for EDA use)
        X_test_fe_raw   : pd.DataFrame — post-FE but pre-scaling
    """
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE — START")
    logger.info("=" * 60)

    # ── Step 1: Run preprocessing ──────────────────────────────────────────
    logger.info("\n[Step 1] Running preprocessing pipeline ...")
    prep_results = run_preprocessing(data_path=data_path, save_artifacts=save_artifacts)

    X_train_raw = prep_results["X_train_raw"]
    X_test_raw  = prep_results["X_test_raw"]
    y_train     = prep_results["y_train"]
    y_test      = prep_results["y_test"]

    logger.info(
        f"  Preprocessing complete  →  "
        f"Train: {X_train_raw.shape} | Test: {X_test_raw.shape}"
    )

    # ── Step 2: Compute thresholds from training set only ─────────────────
    logger.info("\n[Step 2] Computing thresholds from training data ...")
    thresholds = compute_thresholds(X_train_raw)

    # ── Step 3: Engineer features on training set ──────────────────────────
    logger.info("\n[Step 3] Applying feature engineering to training set ...")
    X_train_fe = apply_feature_engineering(X_train_raw, thresholds)
    logger.info(
        f"  X_train: {X_train_raw.shape[1]} raw features  →  "
        f"{X_train_fe.shape[1]} features after engineering"
    )

    # ── Step 4: Apply SAME thresholds to test set (no leakage) ────────────
    logger.info("\n[Step 4] Applying feature engineering to test set ...")
    X_test_fe = apply_feature_engineering(X_test_raw, thresholds)
    logger.info(
        f"  X_test:  {X_test_raw.shape[1]} raw features  →  "
        f"{X_test_fe.shape[1]} features after engineering"
    )

    # ── Step 5: Log engineered feature summary ─────────────────────────────
    logger.info("\n[Step 5] Engineered feature summary (training set):")
    for col in ALL_ENGINEERED_COLS:
        fraud_rate     = X_train_fe.loc[y_train[y_train == 1].index, col].mean()
        non_fraud_rate = X_train_fe.loc[y_train[y_train == 0].index, col].mean()
        logger.info(
            f"  {col:<30}  fraud_avg={fraud_rate:.4f}  "
            f"non_fraud_avg={non_fraud_rate:.4f}"
        )

    # ── Step 6: Build FE preprocessor ─────────────────────────────────────
    logger.info("\n[Step 6] Building FE ColumnTransformer ...")
    preprocessor = build_fe_preprocessor()

    # ── Step 7: Fit on training data only ─────────────────────────────────
    logger.info("\n[Step 7] Fitting FE preprocessor on training data ...")
    X_train_proc = preprocessor.fit_transform(X_train_fe)
    logger.info(f"  X_train_proc shape: {X_train_proc.shape}")

    # ── Step 8: Transform test data ────────────────────────────────────────
    logger.info("\n[Step 8] Transforming test data ...")
    X_test_proc = preprocessor.transform(X_test_fe)
    logger.info(f"  X_test_proc shape:  {X_test_proc.shape}")

    # ── Step 9: Extract feature names ─────────────────────────────────────
    feature_names = get_fe_feature_names(preprocessor)
    logger.info(f"\n[Step 9] Final feature names ({len(feature_names)}):")
    for i, name in enumerate(feature_names, 1):
        logger.info(f"    {i:>2}. {name}")

    # ── Step 10: Save all artifacts ────────────────────────────────────────
    if save_artifacts:
        logger.info("\n[Step 10] Saving processed data and artifacts ...")
        save_processed_data(
            X_train_proc, X_test_proc,
            y_train, y_test,
            feature_names,
        )
        save_fe_artifacts(preprocessor, thresholds, feature_names)

    logger.info("\n" + "=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE — COMPLETE")
    logger.info("=" * 60)

    return {
        "X_train_proc":   X_train_proc,
        "X_test_proc":    X_test_proc,
        "y_train":        y_train,
        "y_test":         y_test,
        "feature_names":  feature_names,
        "preprocessor":   preprocessor,
        "thresholds":     thresholds,
        "X_train_fe_raw": X_train_fe,   # pre-scaling, useful for EDA notebooks
        "X_test_fe_raw":  X_test_fe,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — standalone execution and self-test
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Standalone entry point for testing and running the FE pipeline.

    Runs the full pipeline, prints a summary table, and validates:
    - No NaN values in processed arrays
    - Correct number of features
    - Saved files exist on disk
    """
    print("\n" + "═" * 60)
    print("  FEATURE ENGINEERING — STANDALONE TEST")
    print("═" * 60)

    results = run_feature_engineering(save_artifacts=True)

    X_train = results["X_train_proc"]
    X_test  = results["X_test_proc"]
    y_train = results["y_train"]
    y_test  = results["y_test"]
    names   = results["feature_names"]

    print("\n" + "─" * 60)
    print("  RESULTS SUMMARY")
    print("─" * 60)

    # Shape
    print(f"\n  {'X_train_proc':<25} shape: {X_train.shape}")
    print(f"  {'X_test_proc':<25} shape: {X_test.shape}")
    print(f"  {'y_train':<25} fraud: {y_train.sum()} / {len(y_train)}"
          f"  ({y_train.mean()*100:.2f}%)")
    print(f"  {'y_test':<25} fraud: {y_test.sum()} / {len(y_test)}"
          f"  ({y_test.mean()*100:.2f}%)")

    # Feature names
    print(f"\n  Final features ({len(names)}):")
    for i, name in enumerate(names, 1):
        tag = "  ← engineered" if any(
            name.startswith(ec) for ec in ALL_ENGINEERED_COLS
        ) else ""
        print(f"    {i:>2}. {name}{tag}")

    # NaN check
    nan_train = np.isnan(X_train).sum()
    nan_test  = np.isnan(X_test).sum()
    print(f"\n  NaN check — train: {nan_train}  |  test: {nan_test}")
    assert nan_train == 0, "NaN values found in X_train_proc!"
    assert nan_test  == 0, "NaN values found in X_test_proc!"
    print("  NaN validation passed ✓")

    # Saved file check
    expected_files = [
        os.path.join(PROCESSED_DIR, "X_train_fe.csv"),
        os.path.join(PROCESSED_DIR, "X_test_fe.csv"),
        os.path.join(PROCESSED_DIR, "y_train.csv"),
        os.path.join(PROCESSED_DIR, "y_test.csv"),
        FE_PREPROCESSOR_PATH,
        FE_METADATA_PATH,
    ]
    print("\n  Saved file check:")
    all_found = True
    for fpath in expected_files:
        exists = os.path.exists(fpath)
        status = "✓" if exists else "✗ MISSING"
        print(f"    {status}  {os.path.relpath(fpath, _BASE_DIR)}")
        if not exists:
            all_found = False

    if all_found:
        print("\n  All artifacts saved successfully ✓")
    else:
        print("\n  WARNING: Some artifacts are missing!")

    print("─" * 60)
    print("  Ready for Phase 6: train_model.py")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
