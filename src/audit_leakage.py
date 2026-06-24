"""
Phase 7.5 — Data Leakage Audit
================================
Systematically investigates six leakage vectors that could explain
ROC-AUC = 1.0 / AP = 1.0 in the fraud detection pipeline.

Run order matters — each test narrows down the root cause.

Usage
-----
    python src/audit_leakage.py

Outputs
-------
    reports/leakage_audit/leakage_audit_report.json   — machine-readable verdicts
    reports/leakage_audit/leakage_audit_report.txt    — human-readable summary
    reports/leakage_audit/figures/                    — all diagnostic plots
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR    = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT_DIR / "data" / "processed"
RAW_DIR     = ROOT_DIR / "data" / "raw"          # for temporal check
MODELS_DIR  = ROOT_DIR / "models"
AUDIT_DIR   = ROOT_DIR / "reports" / "leakage_audit"
FIG_DIR     = AUDIT_DIR / "figures"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#FAFAFA", "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#CCCCCC",   "axes.grid": True,
    "grid.color": "#EEEEEE",       "font.size": 11,
    "axes.titlesize": 12,          "axes.titleweight": "bold",
    "figure.dpi": 130,             "savefig.dpi": 130,
    "savefig.bbox": "tight",
})

PASS  = "✅ PASS  — No leakage detected"
WARN  = "⚠️  WARN  — Investigate further"
FAIL  = "🚨 FAIL  — Leakage likely"


# ===========================================================================
# Helpers
# ===========================================================================
def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    logger.info("    Plot saved → %s", FIG_DIR / name)


def _load_artifacts() -> tuple[Any, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    logger.info("Loading artifacts …")
    with open(MODELS_DIR / "best_model.pkl", "rb") as f:
        model = pickle.load(f)

    X_train = pd.read_csv(DATA_DIR / "X_train_fe.csv")
    X_test  = pd.read_csv(DATA_DIR / "X_test_fe.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
    y_test  = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

    for df in (X_train, X_test):
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

    logger.info(
        "  X_train=%s  X_test=%s  fraud_train=%.3f  fraud_test=%.3f",
        X_train.shape, X_test.shape, y_train.mean(), y_test.mean(),
    )
    return model, X_train, X_test, y_train, y_test


# ===========================================================================
# TEST 1 — Feature Leakage (Dominant Importance)
# ===========================================================================
def test_feature_leakage(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Check whether a single feature is carrying the entire predictive signal.

    LEAKAGE SIGNAL   : One feature with importance > 0.40, or top-1 feature
                       alone achieves AUC > 0.95 when used as a lone predictor.
    VALID SIGNAL     : Top feature importance < 0.25, AUC drops to < 0.75
                       when top features are removed.
    """
    logger.info("=" * 60)
    logger.info("TEST 1 — Feature Leakage (Dominant Importance)")
    logger.info("=" * 60)

    verdicts = []

    # ── 1a. Feature importances ──────────────────────────────────────────────
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()
    else:
        logger.warning("  Model has no importance attribute — skipping 1a.")
        importances = None

    top_features: list[str] = []

    if importances is not None:
        feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
        top_feat = feat_imp.index[0]
        top_imp  = feat_imp.iloc[0]
        top5_imp = feat_imp.iloc[:5].sum()
        top_features = feat_imp.index[:5].tolist()

        logger.info("  Top-10 feature importances:")
        for feat, imp in feat_imp.head(10).items():
            bar = "█" * int(imp * 40)
            logger.info("    %-30s  %.4f  %s", feat, imp, bar)

        if top_imp > 0.40:
            v = FAIL
            logger.info("  🚨 '%s' dominates at %.4f — strong leakage indicator.", top_feat, top_imp)
        elif top_imp > 0.25:
            v = WARN
            logger.info("  ⚠️  '%s' is high at %.4f — worth investigating.", top_feat, top_imp)
        else:
            v = PASS
            logger.info("  ✅ Top feature importance %.4f — looks reasonable.", top_imp)

        verdicts.append({"check": "dominant_importance", "verdict": v,
                         "top_feature": top_feat, "top_importance": round(float(top_imp), 4),
                         "top5_cumulative": round(float(top5_imp), 4)})

        # Plot
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["#DC2626" if i == 0 else "#2563EB" for i in range(min(15, len(feat_imp)))]
        feat_imp.head(15).sort_values().plot(kind="barh", ax=ax, color=colors[::-1])
        ax.axvline(0.25, color="#F59E0B", linestyle="--", lw=1.5, label="0.25 warning threshold")
        ax.axvline(0.40, color="#DC2626", linestyle="--", lw=1.5, label="0.40 leakage threshold")
        ax.set_title("TEST 1a — Feature Importance  (red bar = top feature)")
        ax.set_xlabel("Importance Score")
        ax.legend()
        _save(fig, "t1a_feature_importance.png")

    # ── 1b. Single-feature AUC for top-5 features ───────────────────────────
    logger.info("  Running single-feature AUC test for top-5 features …")
    single_auc_results = {}
    for feat in top_features:
        x_single = X_test[[feat]].values
        try:
            prob = model.predict_proba(
                pd.DataFrame(
                    np.tile(X_test.mean().values, (len(X_test), 1)),
                    columns=X_train.columns
                ).assign(**{feat: x_single.ravel()})
            )[:, 1]
        except Exception:
            # fallback: train a tiny RF on just this feature
            rf_single = RandomForestClassifier(n_estimators=50, random_state=42)
            rf_single.fit(X_train[[feat]], y_train)
            prob = rf_single.predict_proba(X_test[[feat]])[:, 1]

        auc_single = roc_auc_score(y_test, prob)
        single_auc_results[feat] = round(auc_single, 4)
        flag = "🚨" if auc_single > 0.95 else ("⚠️ " if auc_single > 0.80 else "✅")
        logger.info("    %s  %-30s  single-feature AUC = %.4f", flag, feat, auc_single)

    max_single_auc = max(single_auc_results.values())
    if max_single_auc > 0.95:
        v2 = FAIL
    elif max_single_auc > 0.80:
        v2 = WARN
    else:
        v2 = PASS
    verdicts.append({"check": "single_feature_auc", "verdict": v2,
                     "results": single_auc_results})

    # ── 1c. Drop-top-N AUC test ──────────────────────────────────────────────
    logger.info("  Drop-top-N feature AUC test …")
    drop_auc_results = {}
    for n in [1, 3, 5]:
        drop_cols = feat_imp.index[:n].tolist() if importances is not None else []
        if not drop_cols:
            continue
        remaining = [c for c in X_train.columns if c not in drop_cols]
        rf_drop = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_drop.fit(X_train[remaining], y_train)
        auc_drop = roc_auc_score(y_test, rf_drop.predict_proba(X_test[remaining])[:, 1])
        drop_auc_results[f"drop_top_{n}"] = round(auc_drop, 4)
        flag = "🚨" if auc_drop < 0.70 else ("⚠️ " if auc_drop < 0.85 else "✅")
        logger.info("    %s  Drop top-%d → AUC = %.4f", flag, n, auc_drop)

    verdicts.append({"check": "drop_top_n_auc", "results": drop_auc_results,
                     "verdict": FAIL if any(v < 0.70 for v in drop_auc_results.values())
                                else WARN if any(v < 0.85 for v in drop_auc_results.values())
                                else PASS})

    return {"test": "feature_leakage", "verdicts": verdicts}


# ===========================================================================
# TEST 2 — Duplicate Train-Test Rows
# ===========================================================================
def test_duplicate_rows(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Detect rows that appear in both training and test sets.

    LEAKAGE SIGNAL   : Any overlapping rows (even a handful inflates metrics
                       when the dataset is small).
    VALID SIGNAL     : Zero overlapping rows across both features and labels.
    """
    logger.info("=" * 60)
    logger.info("TEST 2 — Duplicate Train-Test Row Overlap")
    logger.info("=" * 60)

    verdicts = []

    # ── 2a. Feature-only overlap ─────────────────────────────────────────────
    train_str = X_train.astype(str).agg("-".join, axis=1)
    test_str  = X_test.astype(str).agg("-".join, axis=1)
    overlap_x = test_str.isin(train_str.values)
    n_overlap_x = int(overlap_x.sum())
    pct_x = n_overlap_x / len(X_test) * 100

    logger.info("  Feature-row overlap : %d / %d test rows (%.2f%%)", n_overlap_x, len(X_test), pct_x)

    # ── 2b. Feature + label overlap ─────────────────────────────────────────
    train_full = (X_train.astype(str).agg("-".join, axis=1) + "|" + y_train.astype(str).values)
    test_full  = (X_test.astype(str).agg("-".join,  axis=1) + "|" + y_test.astype(str).values)
    overlap_xy = test_full.isin(train_full.values)
    n_overlap_xy = int(overlap_xy.sum())

    logger.info("  Feature+label overlap: %d / %d test rows (%.2f%%)",
                n_overlap_xy, len(X_test), n_overlap_xy / len(X_test) * 100)

    # ── 2c. Internal train duplicates ────────────────────────────────────────
    n_dup_train = int(X_train.duplicated().sum())
    n_dup_test  = int(X_test.duplicated().sum())
    logger.info("  Duplicates within X_train: %d  |  within X_test: %d",
                n_dup_train, n_dup_test)

    if n_overlap_xy > 0:
        v = FAIL
        logger.info("  🚨 %d rows exist in BOTH train and test — guaranteed leakage.", n_overlap_xy)
    elif n_overlap_x > 0:
        v = WARN
        logger.info("  ⚠️  %d feature rows overlap (different labels) — check carefully.", n_overlap_x)
    elif n_dup_train > 50:
        v = WARN
        logger.info("  ⚠️  %d duplicates in train set — may cause inflated CV scores.", n_dup_train)
    else:
        v = PASS
        logger.info("  ✅ No train-test row overlap detected.")

    verdicts.append({
        "check": "train_test_overlap",
        "verdict": v,
        "feature_overlap_rows": n_overlap_x,
        "feature_label_overlap_rows": n_overlap_xy,
        "overlap_pct": round(pct_x, 4),
        "train_internal_duplicates": n_dup_train,
        "test_internal_duplicates":  n_dup_test,
    })

    # ── Plot: flag overlapping test rows ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = [len(X_train), len(X_test), n_overlap_x, n_overlap_xy, n_dup_train]
    labels = ["Train rows", "Test rows", "Feature overlap", "Feat+Label overlap", "Train dups"]
    colors = ["#2563EB", "#2563EB", "#F59E0B" if n_overlap_x > 0 else "#16A34A",
              "#DC2626" if n_overlap_xy > 0 else "#16A34A", "#6B7280"]
    bars = ax.bar(labels, counts, color=colors, edgecolor="white")
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(val), ha="center", va="bottom", fontsize=10)
    ax.set_title("TEST 2 — Train / Test Row Overlap")
    ax.set_ylabel("Row Count")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, "t2_row_overlap.png")

    return {"test": "duplicate_rows", "verdicts": verdicts}


# ===========================================================================
# TEST 3 — SMOTE Leakage
# ===========================================================================
def test_smote_leakage(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Verify SMOTE was applied AFTER the train/test split by comparing
    pre-SMOTE and post-SMOTE class distributions and checking for
    synthetic test-set neighbours in training data.

    LEAKAGE SIGNAL   : Class ratio in y_train is already balanced (≈ 0.5)
                       before any resampling step, OR synthetic samples from
                       test-set region are present in train.
    VALID SIGNAL     : y_train class ratio reflects original imbalance;
                       balanced y_train is only produced inside the pipeline.
    """
    logger.info("=" * 60)
    logger.info("TEST 3 — SMOTE Leakage")
    logger.info("=" * 60)

    verdicts = []

    fraud_rate_train = float(y_train.mean())
    fraud_rate_test  = float(y_test.mean())
    n_train_0 = int((y_train == 0).sum())
    n_train_1 = int((y_train == 1).sum())

    logger.info("  y_train class dist  : {0: %d, 1: %d}  fraud_rate=%.4f",
                n_train_0, n_train_1, fraud_rate_train)
    logger.info("  y_test  class dist  : fraud_rate=%.4f", fraud_rate_test)

    # ── 3a. Is training set already balanced? ────────────────────────────────
    # If SMOTE was applied before splitting, y_train will be ~50/50
    if 0.45 <= fraud_rate_train <= 0.55:
        v_balance = FAIL
        logger.info("  🚨 y_train is nearly balanced (%.4f) — SMOTE likely applied before split.",
                    fraud_rate_train)
    elif 0.35 <= fraud_rate_train <= 0.65:
        v_balance = WARN
        logger.info("  ⚠️  y_train balance (%.4f) is suspicious for raw fraud data.", fraud_rate_train)
    else:
        v_balance = PASS
        logger.info("  ✅ y_train fraud rate %.4f — consistent with imbalanced raw data.", fraud_rate_train)

    verdicts.append({
        "check": "train_class_balance",
        "verdict": v_balance,
        "fraud_rate_train": round(fraud_rate_train, 4),
        "fraud_rate_test":  round(fraud_rate_test,  4),
        "n_class_0": n_train_0,
        "n_class_1": n_train_1,
    })

    # ── 3b. Nearest-neighbour proximity test ─────────────────────────────────
    # Synthetic SMOTE samples are interpolations of real minority samples.
    # If test minority samples are in the train set's neighbourhood, leakage occurred.
    logger.info("  Running nearest-neighbour proximity test (minority class) …")

    try:
        from sklearn.neighbors import NearestNeighbors

        minority_train = X_train[y_train == 1].values
        minority_test  = X_test[y_test  == 1].values

        if len(minority_train) > 0 and len(minority_test) > 0:
            nn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
            nn.fit(minority_train)
            distances, _ = nn.kneighbors(minority_test)
            min_dist  = float(distances.min())
            mean_dist = float(distances.mean())
            zero_dist = int((distances < 1e-6).sum())

            logger.info("  Min distance (test minority → train minority) : %.6f", min_dist)
            logger.info("  Mean distance                                  : %.6f", mean_dist)
            logger.info("  Exact matches (distance ≈ 0)                  : %d", zero_dist)

            if zero_dist > 0:
                v_nn = FAIL
                logger.info("  🚨 %d test minority rows are IDENTICAL to train rows.", zero_dist)
            elif min_dist < 1e-3:
                v_nn = WARN
                logger.info("  ⚠️  Very small min distance — check for near-duplicate interpolations.")
            else:
                v_nn = PASS
                logger.info("  ✅ No near-zero distances — SMOTE appears correctly applied post-split.")

            verdicts.append({
                "check": "smote_nn_proximity",
                "verdict": v_nn,
                "min_distance": round(min_dist, 6),
                "mean_distance": round(mean_dist, 6),
                "exact_matches": zero_dist,
            })

            # Plot distance distribution
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(distances.ravel(), bins=50, color="#2563EB", edgecolor="white", alpha=0.8)
            ax.axvline(1e-3, color="#DC2626", linestyle="--", lw=1.5,
                       label="Near-zero threshold (1e-3)")
            ax.set_title("TEST 3b — Distance: Test Minority → Nearest Train Minority")
            ax.set_xlabel("Euclidean Distance")
            ax.set_ylabel("Count")
            ax.legend()
            _save(fig, "t3b_smote_nn_distances.png")

    except ImportError:
        logger.warning("  sklearn.neighbors unavailable — skipping proximity test.")

    # ── 3c. Class ratio plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (name, dist) in zip(axes, [
        ("Train (saved y_train.csv)", {0: n_train_0, 1: n_train_1}),
        ("Test  (y_test.csv)",        {0: int((y_test == 0).sum()), 1: int((y_test == 1).sum())}),
    ]):
        ax.bar(list(dist.keys()), list(dist.values()),
               color=["#2563EB", "#DC2626"], edgecolor="white")
        ax.set_title(name)
        ax.set_xlabel("Class  (0=Legit, 1=Fraud)")
        ax.set_ylabel("Count")
        for k, v in dist.items():
            ax.text(k, v + max(dist.values()) * 0.01, str(v), ha="center")
    fig.suptitle("TEST 3a — Class Distribution in Saved CSV Files", fontweight="bold")
    fig.tight_layout()
    _save(fig, "t3a_class_distribution.png")

    return {"test": "smote_leakage", "verdicts": verdicts}


# ===========================================================================
# TEST 4 — Temporal Leakage
# ===========================================================================
def test_temporal_leakage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Checks whether the train/test split respects time ordering.

    LEAKAGE SIGNAL   : A time-based re-split produces AUC significantly lower
                       than the original random split (> 0.05 drop).
    VALID SIGNAL     : AUC on time-based split is within 0.03 of original,
                       or no timestamp feature exists.
    """
    logger.info("=" * 60)
    logger.info("TEST 4 — Temporal Leakage")
    logger.info("=" * 60)

    verdicts = []

    # ── 4a. Detect timestamp-like columns ────────────────────────────────────
    all_cols = X_train.columns.tolist()
    time_candidates = [c for c in all_cols if any(
        kw in c.lower() for kw in
        ["time", "date", "hour", "day", "week", "month", "year",
         "timestamp", "ts_", "_ts", "epoch", "unix"]
    )]

    logger.info("  Timestamp-like columns found: %s", time_candidates if time_candidates else "None")

    if not time_candidates:
        logger.info("  No explicit time columns — checking for monotonic numeric features …")
        # Check if any numeric column is monotonically increasing (proxy for a timestamp)
        for col in X_train.select_dtypes(include=[np.number]).columns:
            combined = pd.concat([X_train[col], X_test[col]], ignore_index=True)
            if combined.is_monotonic_increasing:
                time_candidates.append(col)
                logger.info("    Found monotonic column (possible time proxy): %s", col)

    # ── 4b. Time-based re-split AUC comparison ───────────────────────────────
    logger.info("  Performing time-based re-split using combined dataset …")

    X_all = pd.concat([X_train, X_test], ignore_index=True)
    y_all = pd.concat([y_train, y_test], ignore_index=True)
    n_total = len(X_all)
    split_idx = int(n_total * 0.80)

    # Sort by time column if found, otherwise use row order (proxy for insertion order)
    if time_candidates:
        sort_col = time_candidates[0]
        sort_order = X_all[sort_col].argsort().values
        X_sorted = X_all.iloc[sort_order].reset_index(drop=True)
        y_sorted = y_all.iloc[sort_order].reset_index(drop=True)
        logger.info("  Sorted by column: '%s'", sort_col)
    else:
        X_sorted = X_all.copy()
        y_sorted = y_all.copy()
        logger.info("  No time column — using row-order as temporal proxy.")

    X_tr_time = X_sorted.iloc[:split_idx]
    X_te_time = X_sorted.iloc[split_idx:]
    y_tr_time = y_sorted.iloc[:split_idx]
    y_te_time = y_sorted.iloc[split_idx:]

    fraud_in_time_test = y_te_time.mean()
    logger.info("  Time-based test fraud rate: %.4f  (random split: %.4f)",
                fraud_in_time_test, y_test.mean())

    if len(y_te_time.unique()) < 2:
        logger.warning("  ⚠️  Time-based test split has only one class — cannot compute AUC.")
        verdicts.append({"check": "temporal_split_auc", "verdict": WARN,
                         "note": "Single class in time-based test split — check data ordering."})
        return {"test": "temporal_leakage", "verdicts": verdicts}

    rf_time = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_time.fit(X_tr_time, y_tr_time)
    auc_time   = roc_auc_score(y_te_time, rf_time.predict_proba(X_te_time)[:, 1])
    auc_random = roc_auc_score(y_test,    rf_time.predict_proba(X_test)[:, 1])   # reuse same model

    delta = abs(auc_random - auc_time)
    logger.info("  AUC — random split : %.4f  |  time-based split : %.4f  |  Δ = %.4f",
                auc_random, auc_time, delta)

    if delta > 0.08:
        v = FAIL
        logger.info("  🚨 Large AUC drop (Δ=%.4f) — model exploits future information.", delta)
    elif delta > 0.04:
        v = WARN
        logger.info("  ⚠️  Moderate AUC drop (Δ=%.4f) — investigate further.", delta)
    else:
        v = PASS
        logger.info("  ✅ Small AUC drop (Δ=%.4f) — temporal leakage unlikely.", delta)

    verdicts.append({
        "check": "temporal_split_auc",
        "verdict": v,
        "auc_random_split":    round(auc_random, 4),
        "auc_temporal_split":  round(auc_time,   4),
        "delta":               round(delta,       4),
        "time_columns_found":  time_candidates,
    })

    # ── Plot: AUC comparison ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Random Split\n(current)", "Time-Based Split\n(correct)"],
        [auc_random, auc_time],
        color=["#DC2626" if delta > 0.05 else "#16A34A", "#2563EB"],
        width=0.4, edgecolor="white",
    )
    for bar, val in zip(bars, [auc_random, auc_time]):
        ax.text(bar.get_x() + bar.get_width() / 2, val - 0.02,
                f"{val:.4f}", ha="center", va="top", fontsize=12,
                color="white", fontweight="bold")
    ax.set_ylim(max(0, min(auc_random, auc_time) - 0.15), 1.02)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"TEST 4 — Temporal vs Random Split AUC  (Δ = {delta:.4f})")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    _save(fig, "t4_temporal_split_auc.png")

    return {"test": "temporal_leakage", "verdicts": verdicts}


# ===========================================================================
# TEST 5 — Target Correlation
# ===========================================================================
def test_target_correlation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    """
    Compute point-biserial correlation between every feature and the target.

    LEAKAGE SIGNAL   : Any feature with |correlation| > 0.90 is a near-perfect
                       linear proxy for the label.
    VALID SIGNAL     : All correlations below 0.70; no single dominant feature.
    """
    logger.info("=" * 60)
    logger.info("TEST 5 — Target Correlation")
    logger.info("=" * 60)

    verdicts = []
    correlations = {}
    pvalues = {}

    for col in X_train.select_dtypes(include=[np.number]).columns:
        r, p = stats.pointbiserialr(y_train, X_train[col].fillna(0))
        correlations[col] = round(float(r), 4)
        pvalues[col]      = round(float(p), 6)

    corr_series = pd.Series(correlations).abs().sort_values(ascending=False)
    top_corr    = corr_series.head(10)

    logger.info("  Top-10 absolute correlations with target:")
    for feat, r in top_corr.items():
        flag = "🚨" if r > 0.90 else ("⚠️ " if r > 0.70 else "✅")
        logger.info("    %s  %-30s  |r| = %.4f  (p=%.2e)", flag, feat, r, pvalues[feat])

    high_corr_features = corr_series[corr_series > 0.90].index.tolist()
    warn_corr_features = corr_series[(corr_series > 0.70) & (corr_series <= 0.90)].index.tolist()

    if high_corr_features:
        v = FAIL
        logger.info("  🚨 Features with |r| > 0.90: %s", high_corr_features)
    elif warn_corr_features:
        v = WARN
        logger.info("  ⚠️  Features with |r| > 0.70: %s", warn_corr_features)
    else:
        v = PASS
        logger.info("  ✅ No features with |r| > 0.70 — target correlation is healthy.")

    verdicts.append({
        "check":               "target_correlation",
        "verdict":             v,
        "high_corr_features":  high_corr_features,
        "warn_corr_features":  warn_corr_features,
        "top10_correlations":  top_corr.to_dict(),
    })

    # ── Plot: correlation bar chart ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#DC2626" if v > 0.90 else "#F59E0B" if v > 0.70 else "#2563EB"
              for v in top_corr.values]
    top_corr.sort_values().plot(kind="barh", ax=ax, color=colors[::-1])
    ax.axvline(0.70, color="#F59E0B", linestyle="--", lw=1.5, label="|r| = 0.70 warning")
    ax.axvline(0.90, color="#DC2626", linestyle="--", lw=1.5, label="|r| = 0.90 leakage")
    ax.set_title("TEST 5 — Absolute Target Correlation  (top 10 features)")
    ax.set_xlabel("|Point-Biserial Correlation|")
    ax.legend()
    _save(fig, "t5_target_correlation.png")

    return {"test": "target_correlation", "verdicts": verdicts}


# ===========================================================================
# TEST 6 — Permutation Sanity Test
# ===========================================================================
def test_permutation_sanity(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Two sub-tests:

    6a. Shuffled-label training: Train a fresh RF on X_train with y_train
        randomly shuffled. A valid model should produce AUC ≈ 0.5.
        If AUC > 0.70, your features have structure that correlates with
        the original label assignment — strong leakage signal.

    6b. Permutation feature importance: Shuffle each feature column one at a
        time on the test set and measure AUC drop. Features that cause a
        large drop are the true drivers; features that cause NO drop are
        irrelevant noise (or the model ignores them).

    LEAKAGE SIGNAL   : 6a AUC > 0.65 with shuffled labels.
    VALID SIGNAL     : 6a AUC between 0.45–0.55 (pure chance).
    """
    logger.info("=" * 60)
    logger.info("TEST 6 — Permutation Sanity Tests")
    logger.info("=" * 60)

    verdicts = []

    # ── 6a. Shuffled-label model ─────────────────────────────────────────────
    logger.info("  6a. Training model on SHUFFLED labels …")
    rng = np.random.RandomState(42)
    y_shuffled = y_train.copy()
    y_shuffled = pd.Series(rng.permutation(y_shuffled.values), index=y_shuffled.index)

    rf_shuffled = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_shuffled.fit(X_train, y_shuffled)
    auc_shuffled = roc_auc_score(y_test, rf_shuffled.predict_proba(X_test)[:, 1])

    logger.info("  AUC with shuffled labels: %.4f  (expected: ~0.50)", auc_shuffled)

    if auc_shuffled > 0.65:
        v6a = FAIL
        logger.info("  🚨 AUC=%.4f with random labels — features directly encode the label.", auc_shuffled)
    elif auc_shuffled > 0.55:
        v6a = WARN
        logger.info("  ⚠️  AUC=%.4f slightly above chance — mild structural concern.", auc_shuffled)
    else:
        v6a = PASS
        logger.info("  ✅ AUC=%.4f ≈ chance — shuffled label test passed.", auc_shuffled)

    verdicts.append({"check": "shuffled_label_auc", "verdict": v6a,
                     "auc_shuffled_labels": round(auc_shuffled, 4)})

    # ── 6b. Permutation feature importance on test set ───────────────────────
    logger.info("  6b. Computing permutation importance on test set (n_repeats=10) …")
    perm_result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10, random_state=42,
        scoring="roc_auc", n_jobs=-1,
    )
    perm_imp = pd.Series(perm_result.importances_mean, index=X_train.columns)
    perm_imp_sorted = perm_imp.sort_values(ascending=False)

    logger.info("  Top-10 permutation importances (mean AUC drop per feature):")
    for feat, imp in perm_imp_sorted.head(10).items():
        flag = "🚨" if imp > 0.20 else ("⚠️ " if imp > 0.10 else "✅")
        logger.info("    %s  %-30s  mean AUC drop = %.4f  ±%.4f",
                    flag, feat, imp, perm_result.importances_std[X_train.columns.get_loc(feat)])

    top_perm_feat = perm_imp_sorted.index[0]
    top_perm_val  = perm_imp_sorted.iloc[0]
    zero_importance_features = perm_imp[perm_imp <= 0].index.tolist()

    if top_perm_val > 0.30:
        v6b = FAIL
        logger.info("  🚨 Single feature '%s' accounts for AUC drop of %.4f — near-leak.",
                    top_perm_feat, top_perm_val)
    elif len(zero_importance_features) > len(X_train.columns) * 0.5:
        v6b = WARN
        logger.info("  ⚠️  >50%% of features have zero permutation importance — model very sparse.")
    else:
        v6b = PASS
        logger.info("  ✅ Permutation importances distributed across features — looks healthy.")

    verdicts.append({
        "check":                     "permutation_importance",
        "verdict":                   v6b,
        "top_feature":               top_perm_feat,
        "top_feature_auc_drop":      round(float(top_perm_val), 4),
        "zero_importance_features":  zero_importance_features[:10],
    })

    # ── 6c. Null model baseline ──────────────────────────────────────────────
    # A classifier that always predicts the majority class
    majority_prob = float(y_train.mean())
    null_probs    = np.full(len(y_test), majority_prob)
    auc_null      = roc_auc_score(y_test, null_probs)
    logger.info("  Null model (constant=%.4f) AUC: %.4f", majority_prob, auc_null)
    verdicts.append({"check": "null_model_auc", "verdict": PASS,
                     "auc_null_model": round(auc_null, 4),
                     "note": "Your AUC should far exceed this baseline."})

    # ── Plots ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: shuffled vs real AUC
    auc_real = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    axes[0].bar(
        ["Real Labels\n(your model)", "Shuffled Labels\n(sanity)", "Null Model\n(baseline)"],
        [auc_real, auc_shuffled, auc_null],
        color=["#2563EB",
               "#DC2626" if auc_shuffled > 0.65 else "#16A34A",
               "#6B7280"],
        edgecolor="white",
    )
    axes[0].axhline(0.5, color="#F59E0B", linestyle="--", lw=1.2, label="Random = 0.5")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("TEST 6a — Shuffled Label Sanity Check")
    axes[0].legend()
    for i, v in enumerate([auc_real, auc_shuffled, auc_null]):
        axes[0].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=10)

    # Right: permutation importance
    top15 = perm_imp_sorted.head(15).sort_values()
    colors_perm = ["#DC2626" if v > 0.20 else "#F59E0B" if v > 0.10 else "#2563EB"
                   for v in top15.values]
    axes[1].barh(top15.index, top15.values, color=colors_perm, edgecolor="white")
    axes[1].axvline(0.10, color="#F59E0B", linestyle="--", lw=1.2, label="0.10 warning")
    axes[1].axvline(0.20, color="#DC2626", linestyle="--", lw=1.2, label="0.20 alert")
    axes[1].set_title("TEST 6b — Permutation Feature Importance (AUC drop)")
    axes[1].set_xlabel("Mean AUC Decrease")
    axes[1].legend()

    fig.tight_layout()
    _save(fig, "t6_permutation_sanity.png")

    return {"test": "permutation_sanity", "verdicts": verdicts}


# ===========================================================================
# FINAL REPORT
# ===========================================================================
def _verdict_score(v: str) -> int:
    return 2 if "PASS" in v else 1 if "WARN" in v else 0


def generate_report(all_results: list[dict]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Flatten verdicts ─────────────────────────────────────────────────────
    flat_verdicts: list[dict] = []
    for result in all_results:
        for v in result.get("verdicts", []):
            flat_verdicts.append({"test": result["test"], **v})

    fails = [v for v in flat_verdicts if "FAIL" in v["verdict"]]
    warns = [v for v in flat_verdicts if "WARN" in v["verdict"]]
    passes = [v for v in flat_verdicts if "PASS" in v["verdict"]]

    overall = "FAIL" if fails else "WARN" if warns else "PASS"
    score   = sum(_verdict_score(v["verdict"]) for v in flat_verdicts)
    max_score = len(flat_verdicts) * 2

    # ── JSON ─────────────────────────────────────────────────────────────────
    report_json = {
        "project":    "Fraud Detection — Leakage Audit",
        "phase":      "7.5",
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "overall_verdict": overall,
        "score":      f"{score}/{max_score}",
        "summary": {
            "FAIL": len(fails),
            "WARN": len(warns),
            "PASS": len(passes),
        },
        "details": flat_verdicts,
        "figures_dir": str(FIG_DIR),
    }

    json_path = AUDIT_DIR / "leakage_audit_report.json"
    with open(json_path, "w") as f:
        json.dump(report_json, f, indent=2)

    # ── Text summary ─────────────────────────────────────────────────────────
    lines = [
        "=" * 65,
        "  FRAUD DETECTION — DATA LEAKAGE AUDIT REPORT",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
        f"  OVERALL VERDICT : {overall}",
        f"  SCORE           : {score} / {max_score}",
        f"  FAILs           : {len(fails)}",
        f"  WARNs           : {len(warns)}",
        f"  PASSes          : {len(passes)}",
        "=" * 65,
    ]

    if fails:
        lines.append("\n🚨 FAILED CHECKS — Immediate action required:")
        for v in fails:
            lines.append(f"  [{v['test']}] {v['check']}")
            lines.append(f"    → {v['verdict']}")

    if warns:
        lines.append("\n⚠️  WARNING CHECKS — Investigate before Phase 8:")
        for v in warns:
            lines.append(f"  [{v['test']}] {v['check']}")

    if overall == "PASS":
        lines.append("\n✅ All checks passed. Model appears valid.")
        lines.append("   ROC-AUC=1.0 may still reflect dataset simplicity — proceed to Phase 8.")
    elif overall == "FAIL":
        lines.append("\n🚨 STOP — Do not proceed to Phase 8 until FAILs are resolved.")
        lines.append("   Fix leakage, re-run pipeline from Phase 5, then re-audit.")

    lines += [
        "\n" + "=" * 65,
        "  NEXT STEPS",
        "=" * 65,
        "  1. Open reports/leakage_audit/figures/ and review all plots.",
        "  2. Address every FAIL before continuing.",
        "  3. After fixing, re-run: python src/train_model.py && python src/audit_leakage.py",
        "  4. If all checks pass, proceed to: python src/phase8_tune_model.py",
        "=" * 65,
    ]

    txt_path = AUDIT_DIR / "leakage_audit_report.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("\n%s", "\n".join(lines))
    logger.info("Full JSON report → %s", json_path)
    logger.info("Text report      → %s", txt_path)


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    logger.info("=" * 60)
    logger.info("PHASE 7.5 — Data Leakage Audit")
    logger.info("ROC-AUC=1.0 / AP=1.0 investigation")
    logger.info("=" * 60)

    model, X_train, X_test, y_train, y_test = _load_artifacts()

    results = [
        test_feature_leakage(model, X_train, y_train, X_test, y_test),
        test_duplicate_rows(X_train, X_test, y_train, y_test),
        test_smote_leakage(X_train, y_train, X_test, y_test),
        test_temporal_leakage(X_train, X_test, y_train, y_test),
        test_target_correlation(X_train, y_train),
        test_permutation_sanity(model, X_train, X_test, y_train, y_test),
    ]

    generate_report(results)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        logger.error("Missing artifact: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Audit failed: %s", e)
        sys.exit(2)
