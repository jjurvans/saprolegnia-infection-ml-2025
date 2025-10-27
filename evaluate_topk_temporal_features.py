#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 12:03:35 2025

@author: jaanajurvansuu

Evaluate precomputed Top-K feature sets and save four scenario figures:
Each figure has two panels: Brier and LogLoss.

Inputs 
features/topK_global_majority_K{K}.csv   for K in {10,15,...,50}

Outputs
- baseline_metrics.csv
- results_per_K_precomputed.csv
- soft_unweighted_set.png
- soft_weighted_set.png
- hard_unweighted_set.png
- hard_weighted_set.png
"""

# ===================== IMPORTS =====================
import os, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ===================== DETERMINISM =====================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
for var in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(var, "1")

# ===================== PLOT FONTS =====================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

# ===================== CONFIG =====================
CSV = "SaprolegniaInfection_LabeledData_2025.csv"
MAX_LAG = 28
K_GRID  = list(range(10, 51, 5))  # 10..50 step 5
N_OUTER = 3
DPI     = 300

# ===================== UTILITIES =====================
def build_time_features(df: pd.DataFrame, base_cols, max_lag=28,
                        use_roll_mean=True, use_roll_std=True):
    """Past-only features: lags start at 1; rolling stats use shift(1)."""
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
    v = var_series.astype(float).to_numpy()
    v = np.where(np.isfinite(v), v, np.nan)
    v = np.where(np.isnan(v), np.nanmedian(v), v)
    v = np.maximum(v, eps)
    w = 1.0 / (v + eps)
    return w / np.mean(w)

def get_models(task):
    nj = 1
    if task == "soft":
        rf = RandomForestRegressor(n_estimators=600, max_features="sqrt",
                                   bootstrap=True, random_state=SEED, n_jobs=nj)
        lgb = LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                            subsample=1.0, colsample_bytree=1.0, deterministic=True,
                            force_col_wise=True, seed=SEED, feature_fraction_seed=SEED,
                            bagging_seed=SEED, data_random_seed=SEED, num_threads=1, verbosity=-1)
        xgb = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                           subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0,
                           tree_method="exact", random_state=SEED, n_jobs=1, verbosity=0)
    else:
        rf = RandomForestClassifier(n_estimators=600, max_features="sqrt",
                                    bootstrap=True, random_state=SEED, n_jobs=nj)
        lgb = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                             subsample=1.0, colsample_bytree=1.0, deterministic=True,
                             force_col_wise=True, seed=SEED, feature_fraction_seed=SEED,
                             bagging_seed=SEED, data_random_seed=SEED, num_threads=1, verbosity=-1)
        xgb = XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6,
                            subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0,
                            tree_method="exact", random_state=SEED, n_jobs=1,
                            verbosity=0, eval_metric="logloss")
    return {
        "RF":  make_pipeline(SimpleImputer(strategy="median"), rf),
        "LGB": make_pipeline(SimpleImputer(strategy="median"), lgb),
        "XGB": make_pipeline(SimpleImputer(strategy="median"), xgb),
    }

def model_probs(model, X):
    last = list(model.named_steps.keys())[-1]
    est = model.named_steps[last]
    if hasattr(est, "predict_proba"):
        try:
            p = model.predict_proba(X)[:, 1]
            return np.clip(p.astype(float), 1e-12, 1 - 1e-12)
        except Exception:
            pass
    p = model.predict(X)
    return np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)

def weighted_mean(values, weights):
    w = np.asarray(weights, float)
    v = np.asarray(values, float)
    w = np.clip(w, 1e-12, np.inf)
    return float(np.sum(w * v) / np.sum(w))

def brier_weighted(y_true, p, sample_weight=None):
    y = np.asarray(y_true, float); p = np.asarray(p, float)
    loss = (p - y) ** 2
    return float(np.mean(loss)) if sample_weight is None else weighted_mean(loss, sample_weight)

def logloss_weighted_hard(y_true, p, sample_weight=None):
    return log_loss(y_true, p, labels=[0, 1], sample_weight=sample_weight)

def logloss_weighted_soft(y_soft, p, sample_weight=None, eps=1e-12):
    y = np.clip(np.asarray(y_soft, float), eps, 1 - eps)
    p = np.clip(np.asarray(p, float),    eps, 1 - eps)
    loss = -(y*np.log(p) + (1-y)*np.log(1-p))
    return float(np.mean(loss)) if sample_weight is None else weighted_mean(loss, sample_weight)

# ===================== LOAD DATA & DESIGN =====================
df = pd.read_csv(CSV, parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

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

# Targets
y_soft_all = np.clip(df_work["SoftLabel_Occurrence"].astype(float).to_numpy(), 1e-6, 1-1e-6)
y_hard_all = (df_work["InfectionDensity"].astype(float).to_numpy() > 0).astype(int)

# Valid rows only
valid_X = np.isfinite(X_all.to_numpy()).all(axis=1)
mask = valid_X & np.isfinite(y_soft_all) & np.isfinite(y_hard_all)
X_all = X_all.loc[mask].reset_index(drop=True)
y_soft_all = y_soft_all[mask]
y_hard_all = y_hard_all[mask]
df_work = df_work.loc[mask].reset_index(drop=True)

# Weights
w_soft_all = inverse_variance_weights(df_work.get("SoftLabelOccurrence_var", pd.Series(np.ones(len(df_work)))))
w_hard_all = inverse_variance_weights(df_work.get("InfectionDensity_var",     pd.Series(np.ones(len(df_work)))))

# ===================== LOAD PRECOMPUTED TOP-K =====================
def load_topk_features(K):
    path = f"features/topK_global_majority_K{K}.csv"
    if not os.path.exists(path):
        print(f"[WARN] Missing: {path} — skipping K={K}")
        return None
    lst = pd.read_csv(path)["feature"].dropna().astype(str).map(str.strip).tolist()
    return [f for f in lst if f in X_all.columns]

topk_dict = {K: load_topk_features(K) for K in K_GRID}
K_available = [K for K, cols in topk_dict.items() if cols and len(cols) > 0]
if not K_available:
    raise RuntimeError("No usable Top-K lists found under features/topK_global_majority_K*.csv")
print(f"[OK] Using K values: {K_available}")

# ===================== BASELINES =====================
def compute_baselines(y_soft, y_hard, w_soft, w_hard, n_splits=3, use_weights=True):
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=0)
    rows = []
    p_persist_soft = np.r_[np.nan, y_soft[:-1]]
    p_persist_hard = np.r_[np.nan, y_hard[:-1]].astype(float)

    for fold_idx, (tr, te) in enumerate(tscv.split(np.arange(len(y_soft)))):
        wtr_s = w_soft[tr] if (use_weights and w_soft is not None) else None
        wte_s = w_soft[te] if (use_weights and w_soft is not None) else None
        wtr_h = w_hard[tr] if (use_weights and w_hard is not None) else None
        wte_h = w_hard[te] if (use_weights and w_hard is not None) else None

        # SOFT
        ytr_s, yte_s = y_soft[tr], y_soft[te]
        prev_s = float(np.average(ytr_s, weights=wtr_s) if wtr_s is not None else np.mean(ytr_s))
        p_const_s = np.full_like(yte_s, prev_s, dtype=float)
        rows.append(dict(task="soft", kind="constant", fold=fold_idx,
                         Brier=brier_weighted(yte_s, p_const_s, wte_s),
                         LogLoss=logloss_weighted_soft(yte_s, p_const_s, wte_s)))
        ps = p_persist_soft[te]
        m = np.isfinite(ps)
        if m.any():
            rows.append(dict(task="soft", kind="persistence", fold=fold_idx,
                             Brier=brier_weighted(yte_s[m], ps[m], None if wte_s is None else wte_s[m]),
                             LogLoss=logloss_weighted_soft(yte_s[m], ps[m], None if wte_s is None else wte_s[m])))

        # HARD
        ytr_h, yte_h = y_hard[tr], y_hard[te]
        prev_h = float(np.average(ytr_h, weights=wtr_h) if wtr_h is not None else np.mean(ytr_h))
        p_const_h = np.full_like(yte_h, prev_h, dtype=float)
        rows.append(dict(task="hard", kind="constant", fold=fold_idx,
                         Brier=brier_weighted(yte_h, p_const_h, wte_h),
                         LogLoss=logloss_weighted_hard(yte_h, p_const_h, wte_h)))
        ph = p_persist_hard[te]
        m = np.isfinite(ph)
        if m.any():
            rows.append(dict(task="hard", kind="persistence", fold=fold_idx,
                             Brier=brier_weighted(yte_h[m], ph[m], None if wte_h is None else wte_h[m]),
                             LogLoss=logloss_weighted_hard(yte_h[m], ph[m], None if wte_h is None else wte_h[m])))

    dfb = pd.DataFrame(rows)
    base_summary = (dfb.groupby(["task","kind"])
                      .agg(Brier_mean=("Brier","mean"), Brier_sd=("Brier","std"),
                           LogLoss_mean=("LogLoss","mean"), LogLoss_sd=("LogLoss","std"))
                      .reset_index())
    return base_summary

baseline_summary = compute_baselines(y_soft_all, y_hard_all, w_soft_all, w_hard_all,
                                     n_splits=N_OUTER, use_weights=True)
baseline_summary.to_csv("baseline_metrics.csv", index=False)
print("Saved: baseline_metrics.csv")

# ===================== EVALUATE PER-K =====================
outer = TimeSeriesSplit(n_splits=N_OUTER, gap=0)
perf_rows = []

for task in ["soft", "hard"]:
    y_all = y_soft_all if task == "soft" else y_hard_all
    w_all = w_soft_all if task == "soft" else w_hard_all

    for weighted in [False, True]:
        for fold_id, (tr_idx, te_idx) in enumerate(outer.split(X_all)):
            Xtr_all, Xte_all = X_all.iloc[tr_idx], X_all.iloc[te_idx]
            ytr, yte = y_all[tr_idx], y_all[te_idx]
            wtr = (w_all[tr_idx] if weighted else None)
            wte = (w_all[te_idx] if weighted else None)

            for K in K_available:
                cols = topk_dict[K]
                if not cols: 
                    continue
                Xtr, Xte = Xtr_all[cols], Xte_all[cols]

                # Evaluate models
                for name, model in get_models(task).items():
                    last = list(model.named_steps.keys())[-1]
                    fit_kwargs = {f"{last}__sample_weight": wtr} if wtr is not None else {}
                    model.fit(Xtr, ytr, **fit_kwargs)
                    p = model_probs(model, Xte)
                    if task == "soft":
                        b = brier_weighted(yte, p, wte)
                        ll = logloss_weighted_soft(yte, p, wte)
                    else:
                        b = brier_weighted(yte, p, wte)
                        ll = logloss_weighted_hard(yte, p, wte)
                    perf_rows.append(dict(task=task, weighted=weighted, model=name,
                                          K=K, fold=fold_id, Brier=b, LogLoss=ll))

# Save table
perf_df = pd.DataFrame(perf_rows)
if perf_df.empty:
    raise RuntimeError("No evaluation rows produced; check feature files and data alignment.")
summary = (perf_df.groupby(["task","weighted","model","K"])
          .agg(Brier_mean=("Brier","mean"), Brier_sd=("Brier","std"),
               LogLoss_mean=("LogLoss","mean"), LogLoss_sd=("LogLoss","std"))
          .reset_index().sort_values(["task","weighted","model","K"]))
summary.to_csv("results_per_K_precomputed.csv", index=False)
print("Saved: results_per_K_precomputed.csv")

# ===================== PLOTTING (FOUR FIGURES) =====================
# Plot font
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

def plot_scenario_set(task, weighted, perf_summary, baseline_summary,
                      limits_dict, dpi=300):
    label_task = task.capitalize()
    label_weight = "Weighted" if weighted else "Unweighted"
    sub_perf = perf_summary[(perf_summary.task == task) & (perf_summary.weighted == weighted)]
    if sub_perf.empty:
        print(f"[WARN] Nothing to plot for {label_task} – {label_weight}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=False)
    handles_all, labels_all = [], []

    # === Brier ===
    ax = axes[0]
    for name in ["RF", "LGB", "XGB"]:
        s = sub_perf[sub_perf.model == name].sort_values("K")
        (ln,) = ax.plot(s["K"], s["Brier_mean"], marker="o", label=f"{name}")
        ax.fill_between(
            s["K"],
            s["Brier_mean"] - s["Brier_sd"],
            s["Brier_mean"] + s["Brier_sd"],
            alpha=0.10,
        )
    bs_const = baseline_summary[(baseline_summary.task == task) & (baseline_summary.kind == "constant")]
    bs_pers  = baseline_summary[(baseline_summary.task == task) & (baseline_summary.kind == "persistence")]
    if not bs_const.empty:
        ax.axhline(bs_const["Brier_mean"].iloc[0], linestyle=":", label="Baseline: Constant")
    if not bs_pers.empty:
        ax.axhline(bs_pers["Brier_mean"].iloc[0], linestyle="-.", label="Baseline: t−1")

    ax.set_xlabel("K")
    ax.set_ylabel("Brier")
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position("left")
    ax.set_ylim(limits_dict[f"{task}_Brier"])
    ax.grid(alpha=0.3)
    h, l = ax.get_legend_handles_labels()
    handles_all.extend(h); labels_all.extend(l)

    # === LogLoss ===
    ax = axes[1]
    for name in ["RF", "LGB", "XGB"]:
        s = sub_perf[sub_perf.model == name].sort_values("K")
        (ln,) = ax.plot(s["K"], s["LogLoss_mean"], marker="s", linestyle="--", label=f"{name}")
        ax.fill_between(
            s["K"],
            s["LogLoss_mean"] - s["LogLoss_sd"],
            s["LogLoss_mean"] + s["LogLoss_sd"],
            alpha=0.10,
        )
    if not bs_const.empty:
        ax.axhline(bs_const["LogLoss_mean"].iloc[0], linestyle=":", label="Baseline: Constant")
    if not bs_pers.empty:
        ax.axhline(bs_pers["LogLoss_mean"].iloc[0], linestyle="-.", label="Baseline: t−1")

    ax.set_xlabel("K")
    ax.set_ylabel("LogLoss")
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position("left")
    ax.set_ylim(limits_dict[f"{task}_LogLoss"])
    ax.grid(alpha=0.3)
    h, l = ax.get_legend_handles_labels()
    for hh, ll in zip(h, l):
        if ll not in labels_all:
            labels_all.append(ll); handles_all.append(hh)

    # === One legend (left) ===
    fig.legend(handles_all, labels_all,
           loc="center left", bbox_to_anchor=(1.02, 0.5),
           frameon=False)
    fig.suptitle(f"{label_task} – {label_weight}", y=0.98)
    fig.tight_layout(rect=[0.03, 0.0, 1.0, 0.93])
    fname = f"{task}_{'weighted' if weighted else 'unweighted'}_set.png"
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# === Compute per-task-per-metric limits ensuring baselines are visible ===
limits_dict = {}
for task in ["soft", "hard"]:
    sub = summary[summary.task == task]
    base = baseline_summary[baseline_summary.task == task]

    for metric in ["Brier", "LogLoss"]:
        # performance distribution
        mmin = (sub[f"{metric}_mean"] - sub[f"{metric}_sd"]).min()
        mmax = (sub[f"{metric}_mean"] + sub[f"{metric}_sd"]).max()
        # include baselines
        if not base.empty:
            mmax = max(mmax, base[f"{metric}_mean"].max())
            mmin = min(mmin, base[f"{metric}_mean"].min())
        # small padding
        pad = 0.05 * (mmax - mmin)
        limits_dict[f"{task}_{metric}"] = (max(0, mmin - pad), mmax + pad)

# === Plot all four figures ===
for task in ["soft", "hard"]:
    for weighted in [False, True]:
        plot_scenario_set(task, weighted, summary, baseline_summary,
                          limits_dict, dpi=DPI)
