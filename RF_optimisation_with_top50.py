#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 08:41:24 2025

@author: jaanajurvansuu

Random Forest (Hard Labels, Weighted, with top 50 lag and roll features) Optimization Script

This script trains and evaluates weighted Random Forest models 
on temporally ordered Saprolegnia infection data using nested
time-series cross-validation. 

A broad deterministic grid of hyperparameters is searched to identify robust configurations.
- Performance is benchmarked against constant and persistence (t−1) baselines 
using Brier score, LogLoss, ROC-AUC, and PR-AUC.
- Outputs include cross-validated metrics, optimal parameters, and calibration/diagnostic figures 
(calibration/hist, confusion, PR/F1 vs threshold, TPR/TNR vs threshold).
"""

import os, random, warnings
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    log_loss, brier_score_loss, roc_auc_score, average_precision_score,
    confusion_matrix, precision_score, recall_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------ Determinism & Threads ------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)
for var in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(var, "1")

# ------------------------------ Config ------------------------------
CSV_PATH = "SaprolegniaInfection_LabeledData_2025.csv"
SELECTED_FEATURES_CSV = "features/topK_global_majority_K50.csv"  # must contain 50 features
MAX_LAG = 28
OUTER_SPLITS = 3
INNER_SPLITS_BASE = 2    # inner CV folds
OUT_DIR = Path("rf_hw_k50_grid_figures")
RESULTS_CSV = "rf_hw_k50_grid_results.csv"
PARAMS_PER_FOLD_CSV = "rf_hw_k50_params_per_fold.csv"
PARAMS_CONSENSUS_CSV = "rf_hw_k50_params_consensus.csv"

# ---------------------- Plot font -------------------------
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

DBI_F = 300
# ------------------------------ RF Grid (wide, deterministic) ------------------------------

RF_GRID_HARD = {
    "rf__n_estimators":      [400, 600, 800, 1000],
    "rf__max_depth":         [3, 4, 6, 8, None],
    "rf__min_samples_leaf":  [2, 4, 6, 10],
    "rf__min_samples_split": [2, 4],
    "rf__max_features":      ["sqrt", 0.5],
    "rf__bootstrap":         [True],
    "rf__max_samples":       [None, 0.8],
    "rf__criterion":         ["gini", "log_loss"],
    "rf__class_weight":      [None, "balanced"],
    "rf__min_impurity_decrease": [0.0],
    "rf__ccp_alpha":         [0.0],
}

# ------------------------------ Utilities ------------------------------
def build_time_features(df: pd.DataFrame, base_cols, max_lag=28,
                        use_roll_mean=True, use_roll_std=True):
    """Past-only: lags 1..max_lag; rolling stats over shift(1)."""
    frames = []
    for col in base_cols:
        for k in range(1, max_lag+1):
            frames.append(df[col].shift(k).rename(f"{col}_lag{k}"))
        past = df[col].shift(1)
        if use_roll_mean:
            for w in range(1, max_lag+1):
                frames.append(past.rolling(w, min_periods=1).mean().rename(f"{col}_roll_mean{w}"))
        if use_roll_std:
            for w in range(1, max_lag+1):
                frames.append(past.rolling(w, min_periods=1).std(ddof=0).rename(f"{col}_roll_std{w}"))
    return pd.concat(frames, axis=1)

def inverse_variance_weights(var_series: pd.Series, eps=1e-6):
    """w ∝ 1/(var+eps), normalized to mean=1."""
    v = np.asarray(var_series, float)
    v = np.where(np.isfinite(v), v, np.nan)
    v = np.where(np.isnan(v), np.nanmedian(v) if np.isfinite(np.nanmedian(v)) else 1.0, v)
    v = np.maximum(v, eps)
    w = 1.0 / (v + eps)
    return w / np.mean(w)

def _mean_sd(a):
    a = np.asarray(a, float)
    return float(np.nanmean(a)), float(np.nanstd(a, ddof=1))

def summarize_params(per_fold_rows):
    """Consensus over categorical/numeric hyperparams."""
    if not per_fold_rows: return {}
    dfp = pd.DataFrame(per_fold_rows).copy()
    if "fold" in dfp: dfp = dfp.drop(columns=["fold"])
    summary = {}
    for c in dfp.columns:
        vals = dfp[c].dropna().tolist()
        if not vals: summary[c] = None; continue
        is_num = all(isinstance(v,(int,float,np.integer,np.floating)) or v is None for v in vals)
        summary[c] = float(np.mean([v for v in vals if v is not None])) if is_num else pd.Series(vals).mode().iloc[0]
    return summary

def metrics_vs_threshold(y_true_bin: np.ndarray, p: np.ndarray, n_grid: int = 200):
    y = y_true_bin.astype(int); p = np.clip(np.asarray(p, float), 0, 1)
    thr = np.linspace(0.01, 0.99, n_grid)
    prec, rec, f1, tpr, tnr = [], [], [], [], []
    for t in thr:
        yhat = (p >= t).astype(int)
        pr  = precision_score(y, yhat, zero_division=0)
        rc  = recall_score(y, yhat, zero_division=0)
        f1s = f1_score(y, yhat, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
        tpr_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr_ = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec.append(pr); rec.append(rc); f1.append(f1s); tpr.append(tpr_); tnr.append(tnr_)
    return thr, np.array(prec), np.array(rec), np.array(f1), np.array(tpr), np.array(tnr)

# ------------------------------ Load Data ------------------------------
df = pd.read_csv(CSV_PATH, parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

# ------------------------------ Feature Engineering ------------------------------
EXCLUDE = {"Timestamp","InfectionDensity","InfectionDensity_var",
           "SoftLabel_Occurrence","SoftLabelOccurrence_var"}
numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
base_cols = [c for c in numeric_cols if c not in EXCLUDE]
if not base_cols:
    raise RuntimeError("No numeric base features after exclusions.")

X_full = build_time_features(df, base_cols, max_lag=MAX_LAG, use_roll_mean=True, use_roll_std=True)

# Align and drop first MAX_LAG rows
df_work = pd.concat([df.reset_index(drop=True), X_full.reset_index(drop=True)], axis=1)
df_work = df_work.iloc[MAX_LAG:].reset_index(drop=True)
X_all   = X_full.iloc[MAX_LAG:].reset_index(drop=True)

# Targets (hard labels only for this script)
y_hard_all = (df_work["InfectionDensity"].astype(float).to_numpy() > 0).astype(int)

# Valid rows only
valid_X = np.isfinite(X_all.to_numpy()).all(axis=1)
valid_y = np.isfinite(y_hard_all)
mask = valid_X & valid_y
if not mask.any():
    raise RuntimeError("No valid rows after feature engineering.")

X_all = X_all.loc[mask].reset_index(drop=True)
y_hard_all = y_hard_all[mask]
df_work = df_work.loc[mask].reset_index(drop=True)

# ------------------------------ Weights (Hard) ------------------------------
if "InfectionDensity_var" in df_work.columns:
    w_hard_all = inverse_variance_weights(df_work["InfectionDensity_var"])
else:
    w_hard_all = np.ones(len(df_work), float)

# ------------------------------ Load Selected K=50 Features ------------------------------
if not Path(SELECTED_FEATURES_CSV).exists():
    raise RuntimeError(f"Missing selected feature file: {SELECTED_FEATURES_CSV}")
sel = pd.read_csv(SELECTED_FEATURES_CSV)
colname = "feature" if "feature" in sel.columns else sel.columns[0]
selected_features = sel[colname].dropna().astype(str).map(str.strip).tolist()
selected_features = [f for f in selected_features if f in X_all.columns]
if len(selected_features) != 50:
    raise RuntimeError(f"Expected 50 features after intersection, got {len(selected_features)}.")
X_sel = X_all[selected_features].copy()

# ------------------------------ Base Estimator & Pipeline ------------------------------
rf_base = RandomForestClassifier(
    n_estimators=700,  # will be overridden by grid
    max_depth=None,
    min_samples_leaf=2,
    bootstrap=True,
    random_state=SEED,
    n_jobs=1
)
pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                 ("rf", rf_base)])

# ------------------------------ Nested CV: Outer + Inner Grid Search ------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
outer = list(TimeSeriesSplit(n_splits=OUTER_SPLITS, gap=0).split(np.arange(len(X_sel))))

per_fold_params = []
oof_pred = np.full(len(X_sel), np.nan, float)

brier_list, logloss_list, roc_list, pr_list = [], [], [], []
brier_const, logloss_const = [], []
brier_pers,  logloss_pers  = [], []

# Precompute grid combinations
grid_keys, grid_vals = zip(*RF_GRID_HARD.items())
grid_combos = list(product(*grid_vals))

for fold_id, (tr, te) in enumerate(outer):
    print(f"\n[Outer fold {fold_id+1}/{OUTER_SPLITS}] Grid evaluating {len(grid_combos)} combos...")
    # inner CV definition (forward chaining over the training window)
    inner_n = max(2, min(INNER_SPLITS_BASE, max(2, len(tr)//50)))
    inner = list(TimeSeriesSplit(n_splits=inner_n, gap=0).split(np.arange(len(tr))))

    Xtr_all, Xte = X_sel.iloc[tr], X_sel.iloc[te]
    ytr_all, yte = y_hard_all[tr], y_hard_all[te]
    wtr_all = w_hard_all[tr]

    # ----- Inner selection objective (unweighted log loss on validation) -----
    best_score, best_params = np.inf, None
    for comb in grid_combos:
        params = dict(zip(grid_keys, comb))
        est = pipe.set_params(**params)
        val_losses = []

        for tr_in, va_in in inner:
            tr_idx = np.array(tr)[tr_in]; va_idx = np.array(tr)[va_in]
            Xtr, Xva = X_sel.iloc[tr_idx], X_sel.iloc[va_idx]
            ytr, yva = y_hard_all[tr_idx], y_hard_all[va_idx]
            wtr_fold = w_hard_all[tr_idx]

            # Degenerate case: if ytr is single class, use prevalence
            if np.unique(ytr).size < 2:
                pva = np.full(len(va_idx), float(np.mean(ytr)), dtype=float)
            else:
                est.fit(Xtr, ytr, rf__sample_weight=wtr_fold)  # fit uses weights
                pva = est.predict_proba(Xva)[:, 1]

            pva = np.clip(pva, 1e-12, 1-1e-12)
            # UNWEIGHTED validation loss for selection (consistent with earlier script)
            val_losses.append(log_loss(yva, pva, labels=[0,1]))

        mean_val = float(np.mean(val_losses)) if val_losses else np.inf
        if mean_val < best_score:
            best_score, best_params = mean_val, params

    best_clean = {k.replace("rf__", ""): v for k, v in best_params.items()}
    per_fold_params.append({"fold": fold_id, **best_clean})
    print(f"[Fold {fold_id}] best params: {best_clean} | mean val logloss={best_score:.4f}")

    # ----- Refit on full outer-train with weights, predict outer-test -----
    est_best = pipe.set_params(**best_params)
    est_best.fit(Xtr_all, ytr_all, rf__sample_weight=wtr_all)
    p_te = est_best.predict_proba(Xte)[:, 1]
    p_te = np.clip(p_te, 1e-12, 1-1e-12)
    oof_pred[te] = p_te

    # ----- Baselines -----
    p_const = np.full_like(yte, float(np.mean(ytr_all)), dtype=float)
    p_persist = np.r_[np.nan, y_hard_all[:-1]].astype(float)[te]
    mP = np.isfinite(p_persist)

    # ----- Metrics (unweighted evaluation) -----
    if np.unique(yte).size == 2:
        brier = brier_score_loss(yte, p_te)
        ll    = log_loss(yte, p_te, labels=[0,1])
        roc   = roc_auc_score(yte, p_te)
        pr    = average_precision_score(yte, p_te)
    else:
        brier = ll = roc = pr = np.nan
    brier_list.append(brier); logloss_list.append(ll); roc_list.append(roc); pr_list.append(pr)

    if np.unique(yte).size == 2:
        b0 = brier_score_loss(yte, p_const); l0 = log_loss(yte, p_const, labels=[0,1])
    else:
        b0 = l0 = np.nan
    brier_const.append(b0); logloss_const.append(l0)

    if mP.any():
        yP = yte[mP]; pP = np.clip(p_persist[mP], 1e-12, 1-1e-12)
        if np.unique(yP).size == 2:
            bP = brier_score_loss(yP, pP); lP = log_loss(yP, pP, labels=[0,1])
        else:
            bP = lP = np.nan
    else:
        bP = lP = np.nan
    brier_pers.append(bP); logloss_pers.append(lP)

# ------------------------------ Aggregate & Save ------------------------------
res = {
    "Model": "RF_hard_weighted_K50_grid",
    "Brier_mean":  _mean_sd(brier_list)[0], "Brier_sd":  _mean_sd(brier_list)[1],
    "LogLoss_mean":_mean_sd(logloss_list)[0], "LogLoss_sd": _mean_sd(logloss_list)[1],
    "ROC_AUC_mean":_mean_sd(roc_list)[0],    "ROC_AUC_sd": _mean_sd(roc_list)[1],
    "PR_AUC_mean": _mean_sd(pr_list)[0],     "PR_AUC_sd":  _mean_sd(pr_list)[1],
    "Baseline_Brier_mean": _mean_sd(brier_const)[0], "Baseline_Brier_sd": _mean_sd(brier_const)[1],
    "Baseline_LogLoss_mean": _mean_sd(logloss_const)[0], "Baseline_LogLoss_sd": _mean_sd(logloss_const)[1],
    "Persist_Brier_mean": _mean_sd(brier_pers)[0], "Persist_Brier_sd": _mean_sd(brier_pers)[1],
    "Persist_LogLoss_mean": _mean_sd(logloss_pers)[0], "Persist_LogLoss_sd": _mean_sd(logloss_pers)[1],
}
pd.DataFrame([res]).to_csv(RESULTS_CSV, index=False)
pd.DataFrame(per_fold_params).to_csv(PARAMS_PER_FOLD_CSV, index=False)
pd.DataFrame([summarize_params(per_fold_params)]).to_csv(PARAMS_CONSENSUS_CSV, index=False)
print(f"\nSaved metrics -> {RESULTS_CSV}")
print(f"Saved per-fold params -> {PARAMS_PER_FOLD_CSV}")
print(f"Saved consensus params -> {PARAMS_CONSENSUS_CSV}")

# ------------------------------ Diagnostics (Calibration/Hist, Confusion, Threshold Curves) ------------------------------
mask_pred = np.isfinite(oof_pred)
p_all = oof_pred[mask_pred]; y_all = y_hard_all[mask_pred]

# 1) Calibration (deciles) + Score Histogram
edges = np.linspace(0, len(p_all), 11, dtype=int)
obs, pred_vals = [], []
order = np.argsort(p_all); p_ord, y_ord = p_all[order], y_all[order]
for i in range(10):
    sl = slice(edges[i], edges[i+1])
    if edges[i+1]-edges[i] == 0: continue
    pred_vals.append(float(np.mean(p_ord[sl]))); obs.append(float(np.mean(y_ord[sl])))

OUT_DIR.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot([0,1], [0,1], '--', alpha=0.6); ax[0].plot(pred_vals, obs, 'o-')
ax[0].set_xlabel("Predicted p"); ax[0].set_ylabel("Observed freq")
ax[0].set_title("Calibration vs Perfect (deciles)")
bins = np.linspace(0,1,21)
ax[1].hist(p_all[y_all==0], bins=bins, alpha=0.6, label="Class 0")
ax[1].hist(p_all[y_all==1], bins=bins, alpha=0.6, label="Class 1")
ax[1].set_xlim(0,1); ax[1].legend(); ax[1].set_title("Score Histogram")
fig.tight_layout(); fig.savefig(OUT_DIR / "calibration_hist.png", dpi=DBI_F); plt.close(fig)

# 2) Precision / Recall / F1 vs Threshold
thr, prec, rec, f1s, tpr, tnr = metrics_vs_threshold(y_all, p_all)
fig, ax = plt.subplots(1,1, figsize=(8,5))

# Main curves
ax.plot(thr, prec, label="Precision")
ax.plot(thr, rec,  label="Recall")
ax.plot(thr, f1s,  label="F1")

# --- Naïve always-positive baselines ---
pi = float(np.mean(y_all))
f1_ap = 2 * pi / (1 + pi)
ax.axhline(pi,    color="gray", linestyle=":",  alpha=0.7,
           label=f"Always-Positive Precision = π = {pi:.2f}")
ax.axhline(1.0,   color="gray", linestyle="--", alpha=0.5,
           label="Always-Positive Recall = 1")
ax.axhline(f1_ap, color="gray", linestyle="-.", alpha=0.7,
           label=f"Always-Positive F1 = {f1_ap:.2f}")

# --- Mark F1 maximum ---
best_idx = int(np.nanargmax(f1s))
best_thr = float(thr[best_idx])
best_f1 = float(f1s[best_idx])

# Formatting
ax.set_xlabel("Threshold")
ax.set_ylabel("Score")
ax.set_ylim(0,1.02)
ax.set_xlim(0,1.0)
ax.set_title("Precision / Recall / F1 vs Threshold")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(OUT_DIR / "precision_recall_f1_vs_threshold.png")
plt.close(fig)

print(f"Prevalence π={pi:.3f} | Always-Positive F1={f1_ap:.3f} | Max F1={best_f1:.3f} at threshold={best_thr:.3f}")

# 3) TPR / TNR vs Threshold
fig, ax = plt.subplots(1,1, figsize=(8,5))
ax.plot(thr, tpr, label="TPR (Sensitivity)")
ax.plot(thr, tnr, label="TNR (Specificity)")
ax.set_xlabel("Threshold"); ax.set_ylabel("Rate"); ax.set_ylim(0,1.02)
ax.set_title("TPR / TNR vs Threshold")
ax.legend()
fig.tight_layout(); fig.savefig(OUT_DIR / "tpr_tnr_vs_threshold.png", dpi=DBI_F); plt.close(fig)

# --- Mark intersection (TPR ≈ TNR) and Youden-J maximum ---
diff = np.abs(tpr - tnr)
cross_idx = int(np.nanargmin(diff))
cross_thr  = float(thr[cross_idx])
cross_tpr  = float(tpr[cross_idx])
cross_tnr  = float(tnr[cross_idx])

# Youden’s J statistic (TPR + TNR − 1)
youden = tpr + tnr - 1
best_j_idx = int(np.nanargmax(youden))
best_j_thr = float(thr[best_j_idx])
best_j_val = float(youden[best_j_idx])

# Plot cues
ax.scatter(cross_thr, cross_tpr, color="red", s=60,
           label=f"TPR=TNR≈{cross_tpr:.2f} @ thr={cross_thr:.2f}")
ax.axvline(cross_thr, color="red", linestyle="--", alpha=0.5)

ax.scatter(best_j_thr, tpr[best_j_idx], color="purple", s=50,
           label=f"Max Youden J={best_j_val:.2f} @ thr={best_j_thr:.2f}")
ax.axvline(best_j_thr, color="purple", linestyle=":", alpha=0.5)

print(f"TPR=TNR≈{cross_tpr:.3f} at thr={cross_thr:.3f} | "
      f"Max Youden J={best_j_val:.3f} at thr={best_j_thr:.3f}")
print(f"Figures dir -> {OUT_DIR.resolve()}")

# 4) Confusion matrix at fixed threshold (e.g., 0.5)
THR_FIXED = 0.5
yhat = (p_all >= THR_FIXED).astype(int)
cm = confusion_matrix(y_all, yhat, labels=[0, 1])

fig, ax = plt.subplots(figsize=(4.8, 4.2))
sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix (t={THR_FIXED:.2f})")
fig.tight_layout()
fig.savefig(OUT_DIR / f"confusion_t{THR_FIXED:.2f}.png")
plt.close(fig)

# Find broad F1 plateau
thr, prec, rec, f1s, _, _ = metrics_vs_threshold(y_all, p_all)
plateau = thr[f1s > (np.nanmax(f1s) - 0.01)]   # within 0.01 of max F1
thr_rep = float(np.mean(plateau))

yhat = (p_all >= thr_rep).astype(int)
cm = confusion_matrix(y_all, yhat, labels=[0, 1])

fig, ax = plt.subplots(figsize=(4.8, 4.2))
sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix (t≈{thr_rep:.2f}, F1 plateau)")
fig.tight_layout()
fig.savefig(OUT_DIR / f"confusion_t{thr_rep:.2f}_plateau.png")
plt.close(fig)
