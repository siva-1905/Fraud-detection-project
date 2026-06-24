from __future__ import annotations
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


"""
Phase 7: Model Evaluation Pipeline
====================================
Loads the best model and held-out test set produced in Phase 6, runs a
comprehensive evaluation suite, and persists every artefact needed for a
professional ML report or GitHub portfolio.

Outputs
-------
reports/figures/confusion_matrix.png
reports/figures/roc_curve.png
reports/figures/precision_recall_curve.png
reports/figures/feature_importance.png
reports/classification_report.txt
reports/evaluation_metrics.json

Usage
-----
    python src/evaluate_model.py
"""


import json
import logging
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT_DIR    = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT_DIR / "data" / "processed"
MODELS_DIR  = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR    = ROOT_DIR / "logs"

REQUIRED_ARTIFACTS: list[Path] = [
    MODELS_DIR / "best_model.pkl",
    DATA_DIR   / "X_test_fe.csv",
    DATA_DIR   / "y_test.csv",
]

# ---------------------------------------------------------------------------
# Plot style — clean, publication-ready
# ---------------------------------------------------------------------------
STYLE: dict[str, Any] = {
    "figure.facecolor":  "#FAFAFA",
    "axes.facecolor":    "#FFFFFF",
    "axes.edgecolor":    "#CCCCCC",
    "axes.grid":         True,
    "grid.color":        "#EEEEEE",
    "grid.linewidth":    0.8,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
}
plt.rcParams.update(STYLE)

PALETTE = {
    "primary":    "#2563EB",   # blue
    "secondary":  "#DC2626",   # red
    "accent":     "#16A34A",   # green
    "neutral":    "#6B7280",   # grey
    "highlight":  "#F59E0B",   # amber
    "cm_cmap":    "Blues",
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    """Stream + file logger with millisecond timestamps."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt     = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGS_DIR / "evaluate_model.log", mode="a"),
        ],
    )
    return logging.getLogger(__name__)


logger = _setup_logging()


# ===========================================================================
# 1. VALIDATION
# ===========================================================================
def validate_artifacts(paths: list[Path]) -> None:
    """Raise FileNotFoundError with a clear message if any artifact is absent."""
    logger.info("Validating required artifacts …")
    missing = [p for p in paths if not p.exists()]
    if missing:
        msg = "Missing artifacts:\n  " + "\n  ".join(str(p) for p in missing)
        logger.error(msg)
        raise FileNotFoundError(msg)
    logger.info("  ✓ All %d artifacts found.", len(paths))


# ===========================================================================
# 2. LOADING
# ===========================================================================
def load_model(path: Path) -> Any:
    """Deserialise and return the best estimator from pickle."""
    logger.info("Loading model from: %s", path)
    with open(path, "rb") as fh:
        model = pickle.load(fh)
    logger.info("  ✓ Model type: %s", type(model).__name__)
    return model


def load_test_data(
    x_path: Path,
    y_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load test features and labels.

    Returns
    -------
    X_test : np.ndarray
    y_test : np.ndarray
    feature_names : list[str]
    """
    logger.info("Loading test data …")
    X_df = pd.read_csv(x_path)
    y_s  = pd.read_csv(y_path).squeeze()

    # Drop stray index columns
    if "Unnamed: 0" in X_df.columns:
        X_df.drop(columns=["Unnamed: 0"], inplace=True)

    feature_names = X_df.columns.tolist()

    assert len(X_df) == len(y_s), "X_test / y_test row count mismatch."

    logger.info(
        "  ✓ X_test=%s  |  y_test=%s  |  class dist=%s",
        X_df.shape, y_s.shape,
        dict(y_s.value_counts().sort_index()),
    )
    return X_df.values, y_s.values, feature_names


# ===========================================================================
# 3. PREDICTION
# ===========================================================================
def generate_predictions(
    model: Any,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return hard labels and positive-class probabilities.

    Falls back to decision_function if predict_proba is unavailable.
    """
    logger.info("Generating predictions …")
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        raw    = model.decision_function(X_test)
        y_prob = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        logger.warning("  predict_proba unavailable; using normalised decision_function.")
    else:
        y_prob = y_pred.astype(float)
        logger.warning("  No probability method found; ROC / PR curves will be step functions.")

    logger.info("  ✓ Predictions complete  (positive rate = %.3f)", y_pred.mean())
    return y_pred, y_prob


# ===========================================================================
# 4. METRICS
# ===========================================================================
def compute_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute and log the five core evaluation metrics."""
    logger.info("Computing evaluation metrics …")

    metrics = {
        "accuracy":         round(accuracy_score(y_test, y_pred), 6),
        "precision":        round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 6),
        "recall":           round(recall_score(   y_test, y_pred, average="weighted", zero_division=0), 6),
        "f1_score":         round(f1_score(       y_test, y_pred, average="weighted", zero_division=0), 6),
        "roc_auc":          round(roc_auc_score(  y_test, y_prob), 6),
        "avg_precision":    round(average_precision_score(y_test, y_prob), 6),
    }

    logger.info("  Accuracy        : %.6f", metrics["accuracy"])
    logger.info("  Precision       : %.6f", metrics["precision"])
    logger.info("  Recall          : %.6f", metrics["recall"])
    logger.info("  F1 Score        : %.6f", metrics["f1_score"])
    logger.info("  ROC-AUC         : %.6f", metrics["roc_auc"])
    logger.info("  Avg Precision   : %.6f", metrics["avg_precision"])

    return metrics


# ===========================================================================
# 5. PLOTS
# ===========================================================================
def _save_fig(fig: plt.Figure, name: str) -> Path:
    """Save a figure and return its path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / name
    fig.savefig(out)
    plt.close(fig)
    logger.info("  ✓ Saved → %s", out)
    return out


# ── 5a. Confusion Matrix ────────────────────────────────────────────────────
def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> Path:
    """Annotated confusion matrix with raw counts and row-normalised rates."""
    logger.info("Plotting confusion matrix …")

    cm     = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    labels = class_names or [str(c) for c in np.unique(y_test)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrix", fontsize=14, fontweight="bold", y=1.01)

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_pct],
        ["Counts", "Row-Normalised (%)"],
        ["d", ".1f"],
    ):
        disp = ConfusionMatrixDisplay(confusion_matrix=data, display_labels=labels)
        disp.plot(ax=ax, colorbar=False, cmap=PALETTE["cm_cmap"])
        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        # Re-draw text with correct format
        for text in ax.texts:
            val = float(text.get_text())
            if fmt == "d":
                text.set_text(f"{int(val)}")
            else:
                text.set_text(f"{val:.1f}%")

    fig.tight_layout()
    return _save_fig(fig, "confusion_matrix.png")


# ── 5b. ROC Curve ───────────────────────────────────────────────────────────
def plot_roc_curve(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Best Model",
) -> Path:
    """ROC curve with shaded area-under-curve region."""
    logger.info("Plotting ROC curve …")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score   = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=PALETTE["primary"],  lw=2.0,
            label=f"{model_name}  (AUC = {auc_score:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["primary"])
    ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], lw=1.2,
            linestyle="--", label="Random Classifier")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    fig.tight_layout()
    return _save_fig(fig, "roc_curve.png")


# ── 5c. Precision-Recall Curve ──────────────────────────────────────────────
def plot_precision_recall_curve(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Best Model",
) -> Path:
    """Precision-Recall curve with average-precision annotation."""
    logger.info("Plotting Precision-Recall curve …")

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    baseline = y_test.mean()  # positive class prevalence

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color=PALETTE["accent"], lw=2.0,
            label=f"{model_name}  (AP = {ap:.4f})")
    ax.fill_between(recall, precision, alpha=0.08, color=PALETTE["accent"])
    ax.axhline(baseline, color=PALETTE["secondary"], linestyle="--", lw=1.2,
               label=f"No-Skill Baseline ({baseline:.3f})")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0.00, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    fig.tight_layout()
    return _save_fig(fig, "precision_recall_curve.png")


# ── 5d. Feature Importance ──────────────────────────────────────────────────
def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 18,
) -> Path | None:
    """
    Horizontal bar chart for tree-based feature importances.

    Returns None (gracefully) if the model exposes no importance attribute.
    """
    logger.info("Plotting feature importances …")

    importances: np.ndarray | None = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()
        logger.info("  Using |coef_| as proxy for feature importance.")
    else:
        logger.warning("  Model has no feature_importances_ or coef_; skipping plot.")
        return None

    # Sort and truncate
    indices   = np.argsort(importances)[-top_n:]
    feat_sub  = [feature_names[i] for i in indices]
    imp_sub   = importances[indices]
    colors    = [PALETTE["primary"] if v >= np.median(imp_sub) else PALETTE["neutral"]
                 for v in imp_sub]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    bars = ax.barh(feat_sub, imp_sub, color=colors, edgecolor="white", height=0.7)

    # Value labels
    for bar, val in zip(bars, imp_sub):
        ax.text(
            bar.get_width() + max(imp_sub) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left", fontsize=9, color="#374151",
        )

    ax.set_xlabel("Importance Score")
    ax.set_title(f"Feature Importance — Top {top_n} Features")
    ax.set_xlim(0, max(imp_sub) * 1.18)
    ax.invert_yaxis()

    fig.tight_layout()
    return _save_fig(fig, "feature_importance.png")


# ===========================================================================
# 6. CLASSIFICATION REPORT (text)
# ===========================================================================
def save_classification_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    model_name: str = "Best Model",
) -> Path:
    """Write sklearn's classification report to a formatted text file."""
    logger.info("Saving classification report …")

    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        zero_division=0,
        digits=6,
    )

    header = (
        f"{'=' * 60}\n"
        f"  Fraud Detection — Classification Report\n"
        f"  Model : {model_name}\n"
        f"  Date  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'=' * 60}\n\n"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "classification_report.txt"
    out.write_text(header + report)

    logger.info("  ✓ Saved → %s", out)
    logger.info("\n%s", header + report)
    return out


# ===========================================================================
# 7. EXPORT METRICS JSON
# ===========================================================================
def export_metrics_json(
    metrics: dict[str, float],
    model_name: str,
    model_type: str,
    feature_names: list[str],
    y_test: np.ndarray,
    y_pred: np.ndarray,
    figure_paths: dict[str, str],
) -> Path:
    """Serialise all evaluation metadata to a structured JSON file."""
    logger.info("Exporting evaluation metrics to JSON …")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (None, None, None, None)

    payload: dict[str, Any] = {
        "project":        "Fraud Detection",
        "phase":          7,
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
        "model": {
            "name":       model_name,
            "type":       model_type,
        },
        "dataset": {
            "test_samples":      int(len(y_test)),
            "n_features":        len(feature_names),
            "feature_names":     feature_names,
            "positive_rate":     round(float(y_test.mean()), 6),
        },
        "metrics": metrics,
        "confusion_matrix": {
            "true_negative":  int(tn)  if tn  is not None else None,
            "false_positive": int(fp)  if fp  is not None else None,
            "false_negative": int(fn)  if fn  is not None else None,
            "true_positive":  int(tp)  if tp  is not None else None,
        },
        "figures": figure_paths,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "evaluation_metrics.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("  ✓ Saved → %s", out)
    return out


# ===========================================================================
# 8. MAIN PIPELINE
# ===========================================================================
def main() -> None:
    """Orchestrate the full Phase 7 evaluation pipeline."""
    t0 = time.perf_counter()

    logger.info("=" * 60)
    logger.info("PHASE 7 — Model Evaluation Pipeline")
    logger.info("=" * 60)

    # ── 1. Validate ──────────────────────────────────────────────────────────
    validate_artifacts(REQUIRED_ARTIFACTS)

    # ── 2. Load ───────────────────────────────────────────────────────────────
    model = load_model(MODELS_DIR / "best_model.pkl")
    X_test, y_test, feature_names = load_test_data(
        DATA_DIR / "X_test_fe.csv",
        DATA_DIR / "y_test.csv",
    )
    model_name = type(model).__name__
    class_names = ["Legitimate", "Fraud"]   # adjust if your labels differ

    # ── 3. Predict ────────────────────────────────────────────────────────────
    y_pred, y_prob = generate_predictions(model, X_test)

    # ── 4. Metrics ────────────────────────────────────────────────────────────
    metrics = compute_metrics(y_test, y_pred, y_prob)

    # ── 5. Plots ──────────────────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Generating evaluation plots …")

    cm_path  = plot_confusion_matrix(y_test, y_pred, class_names)
    roc_path = plot_roc_curve(y_test, y_prob, model_name)
    pr_path  = plot_precision_recall_curve(y_test, y_prob, model_name)
    fi_path  = plot_feature_importance(model, feature_names, top_n=18)

    figure_paths: dict[str, str] = {
        "confusion_matrix":       str(cm_path),
        "roc_curve":              str(roc_path),
        "precision_recall_curve": str(pr_path),
        "feature_importance":     str(fi_path) if fi_path else "N/A",
    }

    # ── 6. Classification report ─────────────────────────────────────────────
    save_classification_report(y_test, y_pred, class_names, model_name)

    # ── 7. Export JSON ────────────────────────────────────────────────────────
    export_metrics_json(
        metrics=metrics,
        model_name=model_name,
        model_type=str(type(model)),
        feature_names=feature_names,
        y_test=y_test,
        y_pred=y_pred,
        figure_paths=figure_paths,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE  (%.1f seconds)", elapsed)
    logger.info("=" * 60)
    logger.info("Outputs:")
    for label, path in figure_paths.items():
        logger.info("  %-28s → %s", label, path)
    logger.info("  %-28s → %s", "classification_report", REPORTS_DIR / "classification_report.txt")
    logger.info("  %-28s → %s", "evaluation_metrics",    REPORTS_DIR / "evaluation_metrics.json")
    logger.info("=" * 60)
    logger.info("Next step → Phase 8: Hyperparameter Tuning (tune_model.py)")


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        logger.error("Artifact missing — %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unhandled error in Phase 7: %s", exc)
        sys.exit(2)
