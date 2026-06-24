"""
Phase 8: Hyperparameter Tuning
==============================

Uses RandomizedSearchCV to optimize the best fraud detection model.

Outputs
-------
models/tuned_model.pkl
reports/tuning_results.csv
reports/tuning_summary.json

Usage
-----
python src/tune_model.py
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import randint

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
)

# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
LOGS_DIR = ROOT_DIR / "logs"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "tune_model.log")
    ]
)

logger = logging.getLogger(__name__)

# ============================================================
# LOAD DATA
# ============================================================

def load_training_data():

    logger.info("Loading training data...")

    X_train = pd.read_csv(DATA_DIR / "X_train_fe.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()

    if "Unnamed: 0" in X_train.columns:
        X_train.drop(columns=["Unnamed: 0"], inplace=True)

    logger.info(
        "Training shape: %s | Target shape: %s",
        X_train.shape,
        y_train.shape
    )

    return X_train, y_train


# ============================================================
# SEARCH SPACE
# ============================================================

def build_search_space():

    return {

        "n_estimators":
            randint(100, 800),

        "max_depth":
            [None, 5, 10, 15, 20, 25, 30],

        "min_samples_split":
            randint(2, 20),

        "min_samples_leaf":
            randint(1, 10),

        "max_features":
            ["sqrt", "log2"],

        "bootstrap":
            [True, False],

        "class_weight":
            [None, "balanced"]
    }


# ============================================================
# RANDOM SEARCH
# ============================================================

def tune_model(X_train, y_train):

    logger.info("Building base model...")

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    search_space = build_search_space()

    logger.info("Starting RandomizedSearchCV...")

    search = RandomizedSearchCV(

        estimator=rf,

        param_distributions=search_space,

        n_iter=50,

        scoring="average_precision",

        cv=5,

        verbose=2,

        random_state=42,

        n_jobs=-1,

        return_train_score=True
    )

    search.fit(X_train, y_train)

    logger.info(
        "Best Average Precision: %.6f",
        search.best_score_
    )

    logger.info(
        "Best Parameters: %s",
        search.best_params_
    )

    return search


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):

    output_path = MODELS_DIR / "tuned_model.pkl"

    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    logger.info("Saved tuned model -> %s", output_path)

    return output_path


# ============================================================
# SAVE RESULTS
# ============================================================

def save_search_results(search):

    results = pd.DataFrame(search.cv_results_)

    csv_path = REPORTS_DIR / "tuning_results.csv"

    results.to_csv(csv_path, index=False)

    logger.info("Saved search results -> %s", csv_path)

    return csv_path


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(search):

    summary = {

        "timestamp":
            datetime.now().isoformat(),

        "best_score":
            float(search.best_score_),

        "best_parameters":
            search.best_params_,

        "model":
            "RandomForestClassifier",

        "metric":
            "average_precision"
    }

    json_path = REPORTS_DIR / "tuning_summary.json"

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)

    logger.info("Saved summary -> %s", json_path)

    return json_path


# ============================================================
# OPTIONAL HOLDOUT EVALUATION
# ============================================================

def evaluate_best_model(search):

    try:

        X_test = pd.read_csv(DATA_DIR / "X_test_fe.csv")
        y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

        if "Unnamed: 0" in X_test.columns:
            X_test.drop(columns=["Unnamed: 0"], inplace=True)

        model = search.best_estimator_

        preds = model.predict(X_test)

        probs = model.predict_proba(X_test)[:, 1]

        ap = average_precision_score(
            y_test,
            probs
        )

        logger.info(
            "Test Average Precision: %.6f",
            ap
        )

        logger.info(
            "\n%s",
            classification_report(
                y_test,
                preds,
                digits=6
            )
        )

    except Exception as e:

        logger.warning(
            "Test evaluation skipped: %s",
            e
        )


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("PHASE 8 - Hyperparameter Tuning")
    logger.info("=" * 60)

    X_train, y_train = load_training_data()

    search = tune_model(
        X_train,
        y_train
    )

    save_model(
        search.best_estimator_
    )

    save_search_results(
        search
    )

    save_summary(
        search
    )

    evaluate_best_model(
        search
    )

    elapsed = time.perf_counter() - start

    logger.info("=" * 60)
    logger.info(
        "Tuning complete in %.2f seconds",
        elapsed
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()