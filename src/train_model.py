"""
Phase 6: Model Training Pipeline
=================================
Loads engineered features, applies SMOTE for class balancing, trains multiple
classifiers with Stratified 5-Fold CV, compares models, selects the best by
ROC-AUC, and persists the winner along with full training metadata.

Usage:
    python src/train_model.py

Outputs:
    reports/model_comparison.csv   – per-model CV metric summary
    models/best_model.pkl          – serialised best estimator
    models/training_metadata.pkl   – run metadata dict
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR   = Path(__file__).resolve().parents[1]   # project root
DATA_DIR   = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

REQUIRED_ARTIFACTS = [
    DATA_DIR  / "X_train_fe.csv",
    DATA_DIR  / "X_test_fe.csv",
    DATA_DIR  / "y_train.csv",
    DATA_DIR  / "y_test.csv",
    MODELS_DIR / "fe_preprocessor.pkl",
    MODELS_DIR / "fe_metadata.pkl",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    """Configure a root logger that writes to stdout and a rotating file."""
    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "train_model.log", mode="a"),
        ],
    )
    return logging.getLogger(__name__)


logger = _setup_logging()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def validate_artifacts(paths: list[Path]) -> None:
    """Raise FileNotFoundError if any required artifact is missing."""
    logger.info("Validating required Phase 5 artifacts …")
    missing = [p for p in paths if not p.exists()]
    if missing:
        msg = "Missing artifacts:\n  " + "\n  ".join(str(p) for p in missing)
        logger.error(msg)
        raise FileNotFoundError(msg)
    logger.info("  [OK] All %d artifacts present.", len(paths))


def validate_shapes(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Basic shape / alignment sanity checks."""
    logger.info("Validating dataset shapes …")
    assert len(X_train) == len(y_train), "X_train / y_train row mismatch"
    assert len(X_test)  == len(y_test),  "X_test  / y_test  row mismatch"
    assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch between train and test"
    logger.info(
        "  [OK] X_train=%s  X_test=%s  y_train=%s  y_test=%s",
        X_train.shape, X_test.shape, y_train.shape, y_test.shape,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the feature-engineered CSVs produced in Phase 5."""
    logger.info("=" * 60)
    logger.info("PHASE 6 — Model Training Pipeline")
    logger.info("=" * 60)
    logger.info("Loading processed datasets from: %s", DATA_DIR)

    X_train = pd.read_csv(DATA_DIR / "X_train_fe.csv")
    X_test  = pd.read_csv(DATA_DIR / "X_test_fe.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
    y_test  = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

    # Strip index column if it was accidentally saved
    for df in (X_train, X_test):
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

    validate_shapes(X_train, X_test, y_train, y_test)

    logger.info(
        "Class distribution — train: %s | test: %s",
        dict(y_train.value_counts().sort_index()),
        dict(y_test.value_counts().sort_index()),
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# SMOTE
# ---------------------------------------------------------------------------
def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Oversample the minority class in the training set with SMOTE.

    SMOTE is applied ONLY to training data to prevent data leakage.
    """
    logger.info("-" * 60)
    logger.info("Applying SMOTE to training data …")
    logger.info("  Before — shape: %s | class dist: %s", X_train.shape, dict(y_train.value_counts().sort_index()))

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    logger.info("  [OK] After  — shape: %s | class dist: %s", X_res.shape, dict(pd.Series(y_res).value_counts().sort_index()))
    return X_res, y_res


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
def get_models(random_state: int = 42) -> dict[str, Any]:
    """Return a dict of {model_name: estimator} ready for training."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0,
        ),
        "SVM": SVC(
            probability=True,          # needed for ROC-AUC
            class_weight="balanced",
            random_state=random_state,
        ),
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def _cv_scorers() -> dict[str, Any]:
    """Scorers accepted by sklearn's cross_validate."""
    return {
        "accuracy":  "accuracy",
        "precision": make_scorer(precision_score, average="weighted", zero_division=0),
        "recall":    make_scorer(recall_score,    average="weighted", zero_division=0),
        "f1":        make_scorer(f1_score,        average="weighted", zero_division=0),
        "roc_auc":   "roc_auc",
    }


def run_cross_validation(
    models: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run Stratified K-Fold CV for every model.

    Returns a DataFrame with mean ± std for each metric.
    """
    logger.info("-" * 60)
    logger.info("Running Stratified %d-Fold Cross-Validation …", n_splits)

    skf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scorers = _cv_scorers()
    rows    = []

    for name, model in models.items():
        logger.info("  Training: %-22s …", name)
        t0 = time.perf_counter()

        cv_results = cross_validate(
            estimator=model,
            X=X,
            y=y,
            cv=skf,
            scoring=scorers,
            return_train_score=False,
            n_jobs=-1,
        )

        elapsed = time.perf_counter() - t0

        row: dict[str, Any] = {"Model": name}
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            scores = cv_results[f"test_{metric}"]
            row[f"{metric}_mean"] = round(float(np.mean(scores)), 4)
            row[f"{metric}_std"]  = round(float(np.std(scores)),  4)

        row["fit_time_sec"] = round(elapsed, 2)
        rows.append(row)

        logger.info(
            "    acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  auc=%.4f  [%.1fs]",
            row["accuracy_mean"], row["precision_mean"], row["recall_mean"],
            row["f1_mean"], row["roc_auc_mean"], elapsed,
        )

    df = pd.DataFrame(rows).set_index("Model")
    return df


# ---------------------------------------------------------------------------
# Best-model selection & final fit
# ---------------------------------------------------------------------------
def select_and_fit_best(
    comparison_df: pd.DataFrame,
    models: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    selection_metric: str = "roc_auc_mean",
) -> tuple[str, Any]:
    """
    Select the model with the highest mean ROC-AUC and refit it on the full
    (SMOTE-augmented) training set.
    """
    logger.info("-" * 60)
    best_name = comparison_df[selection_metric].idxmax()
    best_score = comparison_df.loc[best_name, selection_metric]
    logger.info("Best model by %s: '%s'  (%.4f)", selection_metric, best_name, best_score)

    logger.info("Refitting '%s' on full training set …", best_name)
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    logger.info("  [OK] Refit complete.")
    return best_name, best_model


# ---------------------------------------------------------------------------
# Reporting & persistence
# ---------------------------------------------------------------------------
def save_comparison(df: pd.DataFrame) -> Path:
    """Save the model comparison DataFrame to CSV."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "model_comparison.csv"
    df.to_csv(out)
    logger.info("Model comparison saved -> %s", out)
    return out


def save_model(model: Any, path: Path) -> None:
    """Serialise a fitted estimator with pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    logger.info("Model saved -> %s  (%.1f KB)", path, path.stat().st_size / 1024)


def save_metadata(
    metadata: dict[str, Any],
    path: Path,
) -> None:
    """Persist training metadata as both pickle and JSON (human-readable)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as fh:
        pickle.dump(metadata, fh)

    json_path = path.with_suffix(".json")
    with open(json_path, "w") as fh:
        # Convert non-serialisable objects to strings for JSON
        safe_meta = {
            k: (v if isinstance(v, (str, int, float, bool, list, dict, type(None))) else str(v))
            for k, v in metadata.items()
        }
        json.dump(safe_meta, fh, indent=2)

    logger.info("Training metadata saved -> %s (+ .json)", path)


def print_summary(comparison_df: pd.DataFrame, best_name: str) -> None:
    """Pretty-print the final leaderboard."""
    logger.info("=" * 60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 60)
    display_cols = ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "roc_auc_mean"]
    logger.info("\n%s", comparison_df[display_cols].to_string())
    logger.info("=" * 60)
    logger.info("Selected best model: %s", best_name)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """End-to-end Phase 6 training pipeline."""
    run_start = time.perf_counter()
    timestamp = datetime.now().isoformat(timespec="seconds")

    # 1. Validate artifacts
    validate_artifacts(REQUIRED_ARTIFACTS)

    # 2. Load data
    X_train_df, X_test_df, y_train_s, y_test_s = load_data()
    feature_names: list[str] = X_train_df.columns.tolist()

    # 3. SMOTE (training data only)
    X_train_res, y_train_res = apply_smote(X_train_df, y_train_s)

    # Keep test data as numpy arrays for consistency
    X_test  = X_test_df.values
    y_test  = y_test_s.values

    # 4. Define models
    models = get_models(random_state=42)
    logger.info("-" * 60)
    logger.info("Models to train: %s", list(models.keys()))

    # 5. Stratified 5-Fold CV
    comparison_df = run_cross_validation(
        models=models,
        X=X_train_res,
        y=y_train_res,
        n_splits=5,
    )

    # 6. Save comparison report
    save_comparison(comparison_df)

    # 7. Select & refit best model
    best_name, best_model = select_and_fit_best(
        comparison_df=comparison_df,
        models=models,
        X_train=X_train_res,
        y_train=y_train_res,
    )

    # 8. Evaluate best model on held-out test set
    logger.info("-" * 60)
    logger.info("Evaluating best model on held-out test set …")
    y_pred      = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

    test_metrics: dict[str, float] = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(   y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1":        round(f1_score(       y_test, y_pred, average="weighted", zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(  y_test, y_pred_prob) if y_pred_prob is not None else float("nan"), 4),
    }
    logger.info("  Test-set metrics: %s", test_metrics)

    # 9. Save best model
    save_model(best_model, MODELS_DIR / "best_model.pkl")

    # 10. Build & save training metadata
    run_elapsed = round(time.perf_counter() - run_start, 2)
    metadata: dict[str, Any] = {
        "phase":              6,
        "timestamp":          timestamp,
        "run_duration_sec":   run_elapsed,
        "python_version":     sys.version,
        # Dataset info
        "X_train_shape_orig": list(X_train_df.shape),
        "X_train_shape_smote": list(X_train_res.shape),
        "X_test_shape":       list(X_test_df.shape),
        "feature_names":      feature_names,
        "n_features":         len(feature_names),
        # CV config
        "cv_folds":           5,
        "cv_stratified":      True,
        "smote_applied":      True,
        "selection_metric":   "roc_auc_mean",
        # Results
        "models_trained":     list(models.keys()),
        "best_model_name":    best_name,
        "best_model_cv_scores": comparison_df.loc[best_name].to_dict(),
        "best_model_test_metrics": test_metrics,
        # Paths
        "best_model_path":    str(MODELS_DIR / "best_model.pkl"),
        "comparison_path":    str(REPORTS_DIR / "model_comparison.csv"),
    }
    save_metadata(metadata, MODELS_DIR / "training_metadata.pkl")

    # 11. Final summary
    print_summary(comparison_df, best_name)
    logger.info("Phase 6 complete in %.1f seconds.", run_elapsed)
    logger.info("Next step -> Phase 7: Hyperparameter Tuning / Model Evaluation")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        logger.error("Artifact validation failed: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error during training: %s", exc)
        sys.exit(2)
