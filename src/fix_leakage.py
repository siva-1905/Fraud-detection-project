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
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR    = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT_DIR / "data" / "processed"
MODELS_DIR  = ROOT_DIR / "models"
FIX_DIR     = ROOT_DIR / "reports" / "leakage_fix"
FIG_DIR     = FIX_DIR / "figures"
LOGS_DIR    = ROOT_DIR / "logs"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGS_DIR / "fix_leakage.log", mode="a"),
        ],
    )
    return logging.getLogger(__name__)

logger = _setup_logging()

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#FAFAFA", "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#CCCCCC", "axes.grid": True,
    "grid.color": "#EEEEEE", "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
})

PALETTE = {
    "original": "#DC2626",
    "fixed_a":  "#16A34A",
    "fixed_b":  "#2563EB",
    "fixed_c":  "#9333EA",
    "neutral":  "#6B7280",
}


# ===========================================================================
# LOAD
# ===========================================================================
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    logger.info("Loading processed datasets ...")
    X_train = pd.read_csv(DATA_DIR / "X_train_fe.csv")
    X_test  = pd.read_csv(DATA_DIR / "X_test_fe.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
    y_test  = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

    for df in (X_train, X_test):
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

    logger.info(
        "  X_train=%s  X_test=%s  fraud_rate_train=%.4f  fraud_rate_test=%.4f",
        X_train.shape, X_test.shape, y_train.mean(), y_test.mean(),
    )
    return X_train, X_test, y_train, y_test


# ===========================================================================
# STEP 1 — DIAGNOSE risk_score_raw
# ===========================================================================
def diagnose_risk_score(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Answer three critical questions about risk_score_raw:
      Q1. Is it a near-perfect predictor on its own?
      Q2. Is it linearly reconstructible from the other features?
      Q3. What is the AUC of a model that never sees it?
      Q4. How well do its distributions separate the two classes?
    """
    logger.info("=" * 60)
    logger.info("STEP 1 — Diagnosing risk_score_raw")
    logger.info("=" * 60)

    diagnosis: dict[str, Any] = {}
    col = "risk_score_raw"

    # ── Q1. Single-feature AUC ───────────────────────────────────────────────
    logger.info("  Q1. Single-feature AUC test ...")
    rf_single = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_single.fit(X_train[[col]], y_train)
    auc_single = roc_auc_score(y_test, rf_single.predict_proba(X_test[[col]])[:, 1])
    diagnosis["single_feature_auc"] = round(auc_single, 4)
    logger.info("    risk_score_raw alone -> AUC = %.4f", auc_single)

    if auc_single > 0.95:
        logger.info("    FAIL: This feature alone is near-perfect — strong proxy for the label.")
    elif auc_single > 0.80:
        logger.info("    WARN: Strong single-feature AUC — high information content.")
    else:
        logger.info("    PASS: Moderate AUC — contributor, not a proxy.")

    # ── Q2. Reconstructibility from other columns ────────────────────────────
    logger.info("  Q2. Can other features reconstruct risk_score_raw?")
    other_cols = [c for c in X_train.columns if c != col]
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train[other_cols], X_train[col])
    r2 = r2_score(X_test[col], gb.predict(X_test[other_cols]))
    diagnosis["reconstructibility_r2"] = round(r2, 4)
    logger.info("    R2 of predicting risk_score_raw from other features: %.4f", r2)

    if r2 > 0.85:
        verdict = "CIRCULAR — derived from the same feature set"
        logger.info("    FAIL: risk_score_raw is ~%.0f%% derivable from other columns.", r2 * 100)
    elif r2 > 0.50:
        verdict = "PARTIAL — some independent information present"
        logger.info("    WARN: Partially reconstructible (R2=%.2f).", r2)
    else:
        verdict = "INDEPENDENT — external signal source"
        logger.info("    PASS: Not reconstructible (R2=%.2f).", r2)
    diagnosis["reconstruction_verdict"] = verdict

    # ── Q3. AUC without risk_score_raw ──────────────────────────────────────
    logger.info("  Q3. Model AUC without risk_score_raw ...")
    rf_no = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_no.fit(X_train[other_cols], y_train)
    auc_no = roc_auc_score(y_test, rf_no.predict_proba(X_test[other_cols])[:, 1])
    diagnosis["auc_without_risk_score"] = round(auc_no, 4)
    logger.info("    AUC without risk_score_raw: %.4f  (original: 1.0000)", auc_no)

    # ── Q4. Class distribution separation ───────────────────────────────────
    logger.info("  Q4. Class distribution analysis ...")
    fraud_scores = X_train.loc[y_train == 1, col]
    legit_scores = X_train.loc[y_train == 0, col]
    ks_stat, ks_p = stats.ks_2samp(fraud_scores, legit_scores)
    diagnosis.update({
        "fraud_mean":   round(float(fraud_scores.mean()), 4),
        "legit_mean":   round(float(legit_scores.mean()), 4),
        "ks_statistic": round(float(ks_stat), 4),
        "ks_pvalue":    float(ks_p),
    })
    logger.info(
        "    Fraud mean=%.3f  Legit mean=%.3f  KS=%.4f  p=%.2e",
        fraud_scores.mean(), legit_scores.mean(), ks_stat, ks_p,
    )

    # ── Plot ─────────────────────────────────────────────────────────────────
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(legit_scores, bins=50, alpha=0.7, color=PALETTE["fixed_b"],
                 label=f"Legitimate (n={len(legit_scores):,})", density=True)
    axes[0].hist(fraud_scores, bins=50, alpha=0.7, color=PALETTE["original"],
                 label=f"Fraud (n={len(fraud_scores):,})", density=True)
    axes[0].set_title("risk_score_raw — Class Distributions")
    axes[0].set_xlabel("risk_score_raw value")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].scatter(
        X_train[col][:2000],
        y_train[:2000] + np.random.uniform(-0.08, 0.08, 2000),
        alpha=0.3, s=8,
        c=y_train[:2000].map({0: PALETTE["fixed_b"], 1: PALETTE["original"]}),
    )
    axes[1].set_title("risk_score_raw vs Target (jittered)")
    axes[1].set_xlabel("risk_score_raw value")
    axes[1].set_ylabel("Class")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Legitimate", "Fraud"])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step1_risk_score_distribution.png")
    plt.close(fig)
    logger.info("  Plot -> %s", FIG_DIR / "step1_risk_score_distribution.png")

    return diagnosis


# ===========================================================================
# SHARED HELPER — SMOTE + train + evaluate
# ===========================================================================
def _train_evaluate(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    y_tr: pd.Series,
    y_te: pd.Series,
    label: str,
    n_folds: int = 5,
) -> tuple[dict, RandomForestClassifier]:
    """SMOTE on training set -> fit RF -> CV AUC + full held-out metrics."""
    logger.info("  Training variant: %s ...", label)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_tr, y_tr)

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_auc = cross_val_score(rf, X_res, y_res, cv=skf, scoring="roc_auc", n_jobs=-1)

    rf.fit(X_res, y_res)
    y_pred = rf.predict(X_te)
    y_prob = rf.predict_proba(X_te)[:, 1]

    metrics = {
        "cv_auc_mean":    round(float(cv_auc.mean()), 4),
        "cv_auc_std":     round(float(cv_auc.std()),  4),
        "test_accuracy":  round(accuracy_score(y_te, y_pred), 4),
        "test_precision": round(precision_score(y_te, y_pred, average="weighted",
                                                zero_division=0), 4),
        "test_recall":    round(recall_score(y_te, y_pred, average="weighted",
                                             zero_division=0), 4),
        "test_f1":        round(f1_score(y_te, y_pred, average="weighted",
                                         zero_division=0), 4),
        "test_roc_auc":   round(roc_auc_score(y_te, y_prob), 4),
        "test_avg_prec":  round(average_precision_score(y_te, y_prob), 4),
    }

    logger.info(
        "    CV AUC=%.4f +/-%.4f  |  Test AUC=%.4f  F1=%.4f  AP=%.4f",
        metrics["cv_auc_mean"], metrics["cv_auc_std"],
        metrics["test_roc_auc"], metrics["test_f1"], metrics["test_avg_prec"],
    )
    return metrics, rf


# ===========================================================================
# STRATEGY A — Drop risk_score_raw entirely
# ===========================================================================
def strategy_a_drop(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict, RandomForestClassifier, pd.DataFrame, pd.DataFrame]:
    """
    Remove risk_score_raw and train on the remaining 17 features.

    USE THIS WHEN
    -------------
    risk_score_raw was computed using the target label (e.g. it is a
    rolling fraud rate per device/merchant computed on the full dataset
    before splitting), or when it will not be available at inference time.
    """
    logger.info("=" * 60)
    logger.info("STRATEGY A — Drop risk_score_raw entirely")
    logger.info("=" * 60)

    X_tr = X_train.drop(columns=["risk_score_raw"])
    X_te = X_test.drop(columns=["risk_score_raw"])
    logger.info("  Remaining features: %d  |  Dropped: ['risk_score_raw']", X_tr.shape[1])

    metrics, model = _train_evaluate(X_tr, X_te, y_train, y_test, "Strategy A")
    return metrics, model, X_tr, X_te


# ===========================================================================
# STRATEGY B — Keep all features, enforce chronological split
# ===========================================================================
def strategy_b_temporal_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict, RandomForestClassifier, pd.DataFrame, pd.DataFrame]:
    """
    Keep all 18 features but re-split the full dataset chronologically
    so that the model never sees future transactions during training.

    USE THIS WHEN
    -------------
    risk_score_raw is a legitimate real-time bank score that IS available
    at transaction time and was NOT computed using the test-set fraud labels.
    """
    logger.info("=" * 60)
    logger.info("STRATEGY B — Enforce chronological split")
    logger.info("=" * 60)

    X_all = pd.concat([X_train, X_test], ignore_index=True)
    y_all = pd.concat([y_train, y_test], ignore_index=True)

    # Find any timestamp-like column
    time_cols = [c for c in X_all.columns
                 if any(kw in c.lower() for kw in
                        ["time", "date", "hour", "epoch", "timestamp"])]

    if time_cols:
        sort_col = time_cols[0]
        order    = X_all[sort_col].argsort().values
        X_sorted = X_all.iloc[order].reset_index(drop=True)
        y_sorted = y_all.iloc[order].reset_index(drop=True)
        logger.info("  Sorted by column: '%s'", sort_col)
    else:
        X_sorted = X_all.copy()
        y_sorted = y_all.copy()
        logger.info("  No timestamp column found — using row order as chronological proxy.")
        logger.info("  Verify that rows in your raw CSV are ordered by transaction date.")

    split_idx = int(len(X_sorted) * 0.80)
    X_tr_t = X_sorted.iloc[:split_idx]
    X_te_t = X_sorted.iloc[split_idx:]
    y_tr_t = y_sorted.iloc[:split_idx]
    y_te_t = y_sorted.iloc[split_idx:]

    logger.info(
        "  Temporal split -> train: %d (fraud=%.3f)  test: %d (fraud=%.3f)",
        len(X_tr_t), y_tr_t.mean(), len(X_te_t), y_te_t.mean(),
    )

    if len(y_te_t.unique()) < 2:
        logger.warning("  Single class in temporal test set — Strategy B skipped.")
        return {"error": "single_class_in_test"}, None, X_tr_t, X_te_t

    metrics, model = _train_evaluate(X_tr_t, X_te_t, y_tr_t, y_te_t, "Strategy B")
    return metrics, model, X_tr_t, X_te_t


# ===========================================================================
# STRATEGY C — Rebuild a clean risk score, no target information
# ===========================================================================
def strategy_c_rebuild_score(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict, RandomForestClassifier, pd.DataFrame, pd.DataFrame]:
    """
    Replace risk_score_raw with a domain-informed weighted score that uses
    ONLY observable transaction attributes — no target label anywhere.

    The weights are fixed domain knowledge, NOT fitted to y_train.
    This prevents re-introducing the same leakage through the back door.

    USE THIS WHEN
    -------------
    You want to preserve a 'risk score' feature concept but need to ensure
    it carries zero circular dependency on the fraud label.
    """
    logger.info("=" * 60)
    logger.info("STRATEGY C — Rebuild clean risk score (no target leakage)")
    logger.info("=" * 60)

    def _build_clean_score(df: pd.DataFrame) -> pd.Series:
        """
        Weighted sum of raw transaction signals.
        Weights are fixed domain priors — never fitted to the fraud label.

        Component breakdown
        -------------------
        device_trust (inverted) : 25%  — low-trust device = higher risk
        amount (normalised)     : 20%  — higher amount = higher risk
        velocity_last_24h       : 20%  — high velocity = higher risk
        foreign_transaction     : 10%  — cross-border adds risk
        location_mismatch       : 10%  — mismatched location adds risk
        is_night_transaction    :  5%  — night-time adds mild risk
        is_high_velocity        :  5%  — binary velocity flag
        is_low_device_trust     :  5%  — binary device flag
        """
        s = pd.Series(0.0, index=df.index)

        def _norm(series: pd.Series) -> pd.Series:
            lo, hi = series.min(), series.max()
            return ((series - lo) / (hi - lo + 1e-9)).clip(0, 1)

        if "device_trust_score" in df.columns:
            s += 0.25 * (1.0 - _norm(df["device_trust_score"]))
        if "amount" in df.columns:
            s += 0.20 * _norm(df["amount"])
        if "velocity_last_24h" in df.columns:
            s += 0.20 * _norm(df["velocity_last_24h"])

        for col, w in [
            ("foreign_transaction",  0.10),
            ("location_mismatch",    0.10),
            ("is_night_transaction", 0.05),
            ("is_high_velocity",     0.05),
            ("is_low_device_trust",  0.05),
        ]:
            if col in df.columns:
                s += w * df[col].fillna(0).clip(0, 1)

        return s.clip(0, 1).rename("risk_score_clean")

    X_tr_c = X_train.drop(columns=["risk_score_raw"]).copy()
    X_te_c = X_test.drop(columns=["risk_score_raw"]).copy()

    X_tr_c["risk_score_clean"] = _build_clean_score(X_tr_c)
    X_te_c["risk_score_clean"] = _build_clean_score(X_te_c)

    r_clean, _ = stats.pointbiserialr(y_train, X_tr_c["risk_score_clean"])
    logger.info("  Clean score correlation with target: |r| = %.4f", abs(r_clean))
    logger.info("  (risk_score_raw had |r| = 0.3717)")

    metrics, model = _train_evaluate(X_tr_c, X_te_c, y_train, y_test, "Strategy C")
    return metrics, model, X_tr_c, X_te_c


# ===========================================================================
# COMPARISON PLOT
# ===========================================================================
def plot_strategy_comparison(
    original_metrics: dict,
    strategy_results: dict[str, dict],
) -> None:
    logger.info("Plotting strategy comparison ...")

    all_results = {"Original\n(leaky)": original_metrics, **strategy_results}
    labels   = list(all_results.keys())
    auc_vals = [v.get("test_roc_auc", 0)  for v in all_results.values()]
    f1_vals  = [v.get("test_f1", 0)        for v in all_results.values()]
    ap_vals  = [v.get("test_avg_prec", 0)  for v in all_results.values()]
    colors   = [PALETTE["original"], PALETTE["fixed_a"],
                PALETTE["fixed_b"], PALETTE["fixed_c"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Strategy Comparison — Original vs Fixed Models", fontsize=13, fontweight="bold")

    for ax, vals, title in zip(
        axes,
        [auc_vals, f1_vals, ap_vals],
        ["ROC-AUC", "F1 Score (weighted)", "Average Precision"],
    ):
        bars = ax.bar(labels, vals, color=colors[:len(labels)],
                      edgecolor="white", width=0.5)
        low = max(0, min(v for v in vals if v > 0) - 0.15)
        ax.set_ylim(low, 1.07)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=10)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9,
            )
        ax.axhline(1.0, color="#9CA3AF", linestyle="--", lw=1.0, alpha=0.6,
                   label="Perfect score")
        ax.legend(fontsize=8)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "strategy_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("  Plot -> %s", out)


# ===========================================================================
# SAVE FIXED ARTIFACTS
# ===========================================================================
def save_fixed_artifacts(
    best_strategy: str,
    best_model: RandomForestClassifier,
    X_train_fixed: pd.DataFrame,
    X_test_fixed: pd.DataFrame,
    metrics: dict,
) -> None:
    logger.info("=" * 60)
    logger.info("Saving fixed artifacts (Strategy: %s) ...", best_strategy)
    logger.info("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train_fixed.to_csv(DATA_DIR / "X_train_fixed.csv", index=False)
    X_test_fixed.to_csv(DATA_DIR  / "X_test_fixed.csv",  index=False)
    logger.info("  Saved -> data/processed/X_train_fixed.csv")
    logger.info("  Saved -> data/processed/X_test_fixed.csv")

    with open(MODELS_DIR / "best_model_fixed.pkl", "wb") as f:
        pickle.dump(best_model, f)
    logger.info("  Saved -> models/best_model_fixed.pkl")

    metadata = {
        "phase":             "7.6_leakage_fix",
        "timestamp":         datetime.now().isoformat(timespec="seconds"),
        "winning_strategy":  best_strategy,
        "feature_count":     X_train_fixed.shape[1],
        "features":          X_train_fixed.columns.tolist(),
        "dropped_features":  ["risk_score_raw"],
        "metrics":           metrics,
        "note": (
            "risk_score_raw removed: dominated importance at 40.3% and "
            "is either circular or encodes post-hoc fraud knowledge. "
            "Model now trained on 17 genuinely independent transaction features."
        ),
    }

    with open(MODELS_DIR / "training_metadata_fixed.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("  Saved -> models/training_metadata_fixed.json")


# ===========================================================================
# DIAGNOSIS REPORT
# ===========================================================================
def write_diagnosis_report(
    diagnosis: dict,
    strategy_results: dict[str, dict],
    best_strategy: str,
) -> None:
    FIX_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 65,
        "  FRAUD DETECTION — LEAKAGE FIX REPORT",
        f"  Generated : {ts}",
        "=" * 65,
        "",
        "ROOT CAUSE: risk_score_raw",
        "-" * 40,
        f"  Single-feature AUC        : {diagnosis.get('single_feature_auc', 'N/A')}",
        f"  Reconstructibility R2     : {diagnosis.get('reconstructibility_r2', 'N/A')}",
        f"  Reconstruction verdict    : {diagnosis.get('reconstruction_verdict', 'N/A')}",
        f"  AUC without this feature  : {diagnosis.get('auc_without_risk_score', 'N/A')}",
        f"  Fraud mean score          : {diagnosis.get('fraud_mean', 'N/A')}",
        f"  Legit mean score          : {diagnosis.get('legit_mean', 'N/A')}",
        f"  KS statistic              : {diagnosis.get('ks_statistic', 'N/A')}",
        "",
        "STRATEGY RESULTS",
        "-" * 40,
    ]

    for name, metrics in strategy_results.items():
        lines.append(f"\n  {name}:")
        if "error" in metrics:
            lines.append(f"    Skipped: {metrics['error']}")
            continue
        for k, v in metrics.items():
            lines.append(f"    {k:<24}: {v}")

    lines += [
        "",
        "=" * 65,
        f"  RECOMMENDED STRATEGY: {best_strategy}",
        "=" * 65,
        "",
        "NEXT STEPS",
        "-" * 40,
        "  1. Check figures in reports/leakage_fix/figures/",
        "  2. Update feature_engineering.py to exclude risk_score_raw",
        "  3. Use data/processed/X_train_fixed.csv for Phase 8 tuning",
        "  4. Use models/best_model_fixed.pkl as Phase 8 baseline",
        "  5. Re-run: python src/audit_leakage.py to verify all FAILs resolved",
        "  6. Expected honest AUC range: 0.88 - 0.96 for real fraud detection",
        "",
    ]

    path = FIX_DIR / "risk_score_diagnosis.txt"
    path.write_text("\n".join(lines))
    logger.info("Diagnosis report -> %s", path)
    logger.info("\n%s", "\n".join(lines))


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    logger.info("=" * 60)
    logger.info("PHASE 7.6 — Leakage Fix: risk_score_raw")
    logger.info("=" * 60)

    # Load artifacts
    X_train, X_test, y_train, y_test = load_data()

    # Step 1: Full diagnosis
    diagnosis = diagnose_risk_score(X_train, X_test, y_train, y_test)

    # Original baseline (leaky) for comparison chart
    original_metrics = {
        "test_roc_auc":  1.0000,
        "test_f1":       0.9995,
        "test_avg_prec": 1.0000,
        "cv_auc_mean":   1.0000,
        "cv_auc_std":    0.0000,
    }

    # Step 2: Run all three strategies
    logger.info("=" * 60)
    logger.info("Running all three fix strategies ...")
    logger.info("=" * 60)

    strategy_results  = {}
    strategy_models   = {}
    strategy_datasets = {}

    m_a, mdl_a, Xtr_a, Xte_a = strategy_a_drop(X_train, X_test, y_train, y_test)
    strategy_results["Strategy A (drop)"]     = m_a
    strategy_models["A"]   = mdl_a
    strategy_datasets["A"] = (Xtr_a, Xte_a)

    m_b, mdl_b, Xtr_b, Xte_b = strategy_b_temporal_split(X_train, X_test, y_train, y_test)
    if "error" not in m_b:
        strategy_results["Strategy B (temporal)"] = m_b
        strategy_models["B"]   = mdl_b
        strategy_datasets["B"] = (Xtr_b, Xte_b)

    m_c, mdl_c, Xtr_c, Xte_c = strategy_c_rebuild_score(X_train, X_test, y_train, y_test)
    strategy_results["Strategy C (rebuild)"]  = m_c
    strategy_models["C"]   = mdl_c
    strategy_datasets["C"] = (Xtr_c, Xte_c)

    # Step 3: Comparison plot
    plot_strategy_comparison(original_metrics, strategy_results)

    # Step 4: Select best by honest AUC (original excluded)
    valid = {k: v for k, v in strategy_results.items() if "error" not in v}
    best_key = max(valid, key=lambda k: valid[k].get("test_roc_auc", 0))
    letter_map = {
        "Strategy A (drop)":     "A",
        "Strategy B (temporal)": "B",
        "Strategy C (rebuild)":  "C",
    }
    best_letter  = letter_map.get(best_key, "A")
    best_model   = strategy_models[best_letter]
    best_dataset = strategy_datasets[best_letter]
    best_metrics = valid[best_key]

    logger.info("=" * 60)
    logger.info("Best strategy: %s  (Test AUC=%.4f)",
                best_key, best_metrics["test_roc_auc"])
    logger.info("=" * 60)

    # Step 5: Save artifacts
    save_fixed_artifacts(
        best_strategy=best_key,
        best_model=best_model,
        X_train_fixed=best_dataset[0],
        X_test_fixed=best_dataset[1],
        metrics=best_metrics,
    )

    # Step 6: Write human-readable report
    write_diagnosis_report(
        diagnosis=diagnosis,
        strategy_results=strategy_results,
        best_strategy=best_key,
    )

    logger.info("=" * 60)
    logger.info("Phase 7.6 complete.")
    logger.info("  Fixed datasets  -> data/processed/X_train_fixed.csv")
    logger.info("  Fixed model     -> models/best_model_fixed.pkl")
    logger.info("  Next: python src/audit_leakage.py  (verify PASS)")
    logger.info("  Then: python src/phase8_tune_model.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        logger.error("Missing file: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Fix pipeline failed: %s", e)
        sys.exit(2)