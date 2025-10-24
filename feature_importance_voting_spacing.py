#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 14:52:59 2025

@author: jaanajurvansuu

Feature voting + spacing with tree-model importances (RF/LGBM/XGB).

Purpose
-Identifies the most informative time-lagged features predicting Saprolegnia infection dynamics.
-Combines feature importances from multiple tree-based models.
-Applies temporal spacing to avoid redundant lag features.
-Produces ranked feature lists and visual heatmaps for interpretation.

Main Steps
-Load data: Reads SaprolegniaInfection_LabeledData_2025.csv.
-Feature engineering: Creates lag, rolling mean, and rolling standard-deviation features (1–28 days).
Excludes future data (past-only).
-Target preparation: Binary “hard” label from infection density > 0.
Continuous “soft” label from probabilistic occurrence. Inverse-variance weights handle heteroscedasticity.
- Model fitting: Fits Random Forest (RF), LightGBM (LGB), and XGBoost (XGB) models.
Runs under both hard/soft tasks, with and without weights. Uses 3-fold time-series cross-validation.
-Feature importance extraction: Retrieves native feature importances (Gini/gain).
Normalizes per model to [0, 1]. Aggregates across folds and scenarios.
-Feature ranking & spacing:
Ranks by mean importance and majority votes across folds. Enforces ±15-day spacing across 
lag/rolling families for each base variable.
-Visualization: Builds normalized heatmaps showing relative importance across models/scenarios.

Outputs
-------
- features/topK_{task}_{weighted}_fold{fold}_K{K}.csv
- features/topK_global_majority_K{K}.csv
- feature_votes_heatmap_majority_top50_normalized_UNspaced.png
- feature_votes_heatmap_majority_top50_normalized_SPACED.png
- heatmap_data_majority_top50_normalized_UNspaced.csv
- heatmap_data_majority_top50_normalized_SPACED.csv
"""

import os, re, random, warnings
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

# --------------------- Config ---------------------
CSV = "SaprolegniaInfection_LabeledData_2025.csv"
MAX_LAG = 28
K_GRID = list(range(10, 51, 5))   # 10..50 step 5
N_OUTER = 3
FAMILY_SPACING = 15               # ±14-day exclusion across lag/mean/std for same base var
DPI = 240
SEED = 42

SCENARIOS = [
    ("soft", False),  # (task, weighted)
    ("soft", True),
    ("hard", False),
    ("hard", True),
]
MODEL_NAMES = ["RF", "LGB", "XGB"]

COLS_ORDER = [
    "RF_hard_nw","LGB_hard_nw","XGB_hard_nw",
    "RF_hard_w","LGB_hard_w","XGB_hard_w",
    "RF_soft_nw","LGB_soft_nw","XGB_soft_nw",
    "RF_soft_w","LGB_soft_w","XGB_soft_w"
]

# ===================== PLOT FONTS =====================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

# ----------------- Environment & plotting -----------------
warnings.filterwarnings("ignore", category=UserWarning)
os.makedirs("features", exist_ok=True)
os.makedirs("figs", exist_ok=True)

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)
for var in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(var, "1")

plt.rcParams.update({
    "font.family": "Arial", "font.size": 14,
    "axes.labelsize": 14, "axes.titlesize": 14,
    "legend.fontsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14
})

# ----------------- Feature engineering -----------------
def build_time_features(df: pd.DataFrame, base_cols, max_lag=28, use_roll_mean=True, use_roll_std=True):
    """Past-only features: lags 1..max_lag; rolling mean/std over past (shifted)."""
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
    """Inverse-variance weights (normalized to mean 1)."""
    v = var_series.astype(float).to_numpy()
    v = np.where(np.isfinite(v), v, np.nan)
    v = np.where(np.isnan(v), np.nanmedian(v), v)
    v = np.maximum(v, eps)
    w = 1.0 / (v + eps)
    return w / np.mean(w)

# ----------------- Models & importances -----------------
def get_models(task):
    nj = 1
    if task == "soft":
        return {
            "RF":  make_pipeline(SimpleImputer(strategy="median"),
                                 RandomForestRegressor(n_estimators=600, max_features="sqrt",
                                                       bootstrap=True, random_state=SEED, n_jobs=nj)),
            "LGB": make_pipeline(SimpleImputer(strategy="median"),
                                 LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                                               subsample=1.0, colsample_bytree=1.0,
                                               deterministic=True, force_col_wise=True, seed=SEED,
                                               feature_fraction_seed=SEED, bagging_seed=SEED,
                                               data_random_seed=SEED, num_threads=1, verbosity=-1)),
            "XGB": make_pipeline(SimpleImputer(strategy="median"),
                                 XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                                              subsample=1.0, colsample_bytree=1.0,
                                              reg_lambda=1.0, tree_method="exact",
                                              random_state=SEED, n_jobs=1, verbosity=0))
        }
    else:
        return {
            "RF":  make_pipeline(SimpleImputer(strategy="median"),
                                 RandomForestClassifier(n_estimators=600, max_features="sqrt",
                                                        bootstrap=True, random_state=SEED, n_jobs=nj)),
            "LGB": make_pipeline(SimpleImputer(strategy="median"),
                                 LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                                                subsample=1.0, colsample_bytree=1.0,
                                                deterministic=True, force_col_wise=True, seed=SEED,
                                                feature_fraction_seed=SEED, bagging_seed=SEED,
                                                data_random_seed=SEED, num_threads=1, verbosity=-1)),
            "XGB": make_pipeline(SimpleImputer(strategy="median"),
                                 XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6,
                                               subsample=1.0, colsample_bytree=1.0,
                                               reg_lambda=1.0, tree_method="exact",
                                               random_state=SEED, n_jobs=1, verbosity=0,
                                               eval_metric="logloss"))
        }

def feature_importance_native(fitted_pipeline, X_cols):
    """Return native importance as a Series aligned to X_cols. RF: Gini; LGB/XGB: gain/split proxy."""
    last = list(fitted_pipeline.named_steps.keys())[-1]
    est = fitted_pipeline.named_steps[last]
    try:
        if hasattr(est, "feature_importances_"):
            imp = np.asarray(est.feature_importances_, float)
            if imp.size != len(X_cols):
                imp = np.pad(imp, (0, max(0, len(X_cols)-imp.size)))[:len(X_cols)]
            s = pd.Series(imp, index=X_cols).fillna(0.0)
            return s
    except Exception:
        pass
    return pd.Series(0.0, index=X_cols)

def fit_with_guard(pipe, X, y, sample_weight, task):
    """Fit model; return fitted pipeline or None on failure (e.g., single-class for classifiers)."""
    try:
        last = list(pipe.named_steps.keys())[-1]
        fit_kwargs = {f"{last}__sample_weight": sample_weight} if sample_weight is not None else {}
        if (task == "hard") and np.unique(y).size < 2:
            return None
        pipe.fit(X, y, **fit_kwargs)
        return pipe
    except Exception:
        return None

# ----------------- Spacing helpers -----------------
_FEAT_RX = re.compile(r"^(?P<base>.+?)_(?P<family>lag|roll_mean|roll_std)(?P<day>\d+)$")

def parse_family_day(feat: str):
    m = _FEAT_RX.match(feat)
    if not m:
        return None, None, None
    return m.group("family"), int(m.group("day")), m.group("base")

def select_topK_spaced_across_families(ranked_features, K, spacing=15, relax_if_needed=True):
    """
    Greedy spacing ACROSS families (lag / roll_mean / roll_std) for same base variable.
    Once (base, day) is chosen, forbid any other with same base and |day - day'| < spacing.
    """
    selected = []
    used_days_by_base = defaultdict(list)
    for f in ranked_features:
        if len(selected) >= K:
            break
        fam, day, base = parse_family_day(f)
        if base is None or day is None:
            selected.append(f); continue
        if all(abs(day - d) >= spacing for d in used_days_by_base[base]):
            selected.append(f); used_days_by_base[base].append(day)
    if len(selected) >= K or not relax_if_needed:
        return selected[:K]
    used = set(selected)
    for f in ranked_features:
        if len(selected) >= K: break
        if f in used: continue
        fam, day, base = parse_family_day(f)
        if base is None or day is None:
            selected.append(f); used.add(f); continue
        if all(abs(day - d) >= spacing for d in used_days_by_base[base]):
            selected.append(f); used.add(f); used_days_by_base[base].append(day)
    return selected[:K]

# ----------------- Load & prepare data -----------------
df = pd.read_csv(CSV, parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

EXCLUDE = {"Timestamp", "InfectionDensity", "InfectionDensity_var",
           "SoftLabel_Occurrence", "SoftLabelOccurrence_var"}
numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
base_cols = [c for c in numeric_cols if c not in EXCLUDE]
if not base_cols:
    raise RuntimeError("No numeric base features after exclusions.")

X_feats = build_time_features(df, base_cols, max_lag=MAX_LAG, use_roll_mean=True, use_roll_std=True)

# Drop first MAX_LAG rows (need past history)
df_work = pd.concat([df.reset_index(drop=True), X_feats.reset_index(drop=True)], axis=1)
df_work = df_work.iloc[MAX_LAG:].reset_index(drop=True)
X_all   = X_feats.iloc[MAX_LAG:].reset_index(drop=True)

# Targets
y_soft_all = np.clip(df_work["SoftLabel_Occurrence"].astype(float).to_numpy(), 1e-6, 1 - 1e-6)
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

# ----------------- Outer CV: collect importances -----------------
outer = TimeSeriesSplit(n_splits=N_OUTER, gap=0)

# scenario_imps[(task, weighted)][feature][model] -> list of normalized importances (per fold)
scenario_imps = {
    (task, weighted): {f: {m: [] for m in MODEL_NAMES} for f in X_all.columns}
    for (task, weighted) in SCENARIOS
}
fold_spaced_orders = []

for fold, (tr_idx, te_idx) in enumerate(outer.split(X_all)):
    Xtr = X_all.iloc[tr_idx]
    agg_imps_df = pd.DataFrame(0.0, index=Xtr.columns, columns=MODEL_NAMES)

    # Diagnostics
    print(f"[fold {fold}] hard y var={np.var(y_hard_all[tr_idx]):.4g}, soft y var={np.var(y_soft_all[tr_idx]):.4g}")

    for (task, weighted) in SCENARIOS:
        y_all_cur = y_soft_all if task == "soft" else y_hard_all
        w_all_cur = (w_soft_all if task == "soft" else w_hard_all) if weighted else None

        ytr = y_all_cur[tr_idx]
        wtr = (w_all_cur[tr_idx] if w_all_cur is not None else None)

        imps_dict = {}
        for name, pipe in get_models(task).items():
            fitted = fit_with_guard(pipe, Xtr, ytr, sample_weight=wtr, task=task)
            if fitted is None:
                s = pd.Series(0.0, index=Xtr.columns)
            else:
                s = feature_importance_native(fitted, list(Xtr.columns)).fillna(0.0)
                mx = float(s.max()); s = (s / mx) if mx > 0 else s
            imps_dict[name] = s

            for ftr, val in s.items():
                scenario_imps[(task, weighted)][ftr][name].append(float(val))

        imps_df = pd.concat(imps_dict, axis=1).reindex(columns=MODEL_NAMES).fillna(0.0)
        agg_imps_df = agg_imps_df.add(imps_df, fill_value=0.0)

    # Pooled ranking for this fold
    ranks = agg_imps_df.rank(ascending=False, method="average")
    mean_rank = ranks.mean(axis=1).sort_values(ascending=True)
    ranked_features = list(mean_rank.index)

    # Spaced order for this fold (used later for majority voting)
    spaced_order = select_topK_spaced_across_families(ranked_features, K=max(K_GRID),
                                                      spacing=FAMILY_SPACING, relax_if_needed=True)
    fold_spaced_orders.append(spaced_order)

    # Save per-scenario Top-K feature lists for this fold (same spaced order used for all scenarios)
    for (task, weighted) in SCENARIOS:
        for K in K_GRID:
            topK = spaced_order[:K]
            out_fp = f"features/topK_{task}_{int(weighted)}_fold{fold}_K{K}.csv"
            pd.DataFrame({"feature": topK}).to_csv(out_fp, index=False)

# ----------------- Build per-scenario importance table -----------------
def label(task, weighted, model):
    return f"{model}_{task}_{'w' if weighted else 'nw'}"

# Average normalized importances across folds → per-scenario matrices
rows = []
for (task, weighted) in SCENARIOS:
    store = scenario_imps[(task, weighted)]
    for ftr, mm in store.items():
        for m in MODEL_NAMES:
            vals = mm[m]
            avg = float(np.mean(vals)) if len(vals) > 0 else 0.0
            rows.append({"feature": ftr, "col": label(task, weighted, m), "val": avg})
T = pd.DataFrame(rows)
wide = T.pivot(index="feature", columns="col", values="val").fillna(0.0)

# Ensure all columns present in defined order
for c in COLS_ORDER:
    if c not in wide.columns:
        wide[c] = 0.0
wide = wide[COLS_ORDER].copy()

# Normalize each column to [0,1] for plotting (idempotent if already normalized)
norm = wide.copy()
for c in COLS_ORDER:
    mx = float(norm[c].max())
    if mx > 0:
        norm[c] = norm[c] / mx

# Save the avaraged importances
for c in COLS_ORDER:
    pd.DataFrame({c: wide[c]}).to_csv(f"importance_{c}.csv") 

# ----------------- Majority voting across folds -----------------
maxK = min(max(K_GRID), min(len(lst) for lst in fold_spaced_orders))
votes_all = Counter()
for spaced in fold_spaced_orders:
    votes_all.update(spaced[:maxK])

# Tie-breaker: global mean normalized importance (higher better)
mean_importance = norm.mean(axis=1)

ranked_by_votes = sorted(
    votes_all.keys(),
    key=lambda f: (-votes_all[f], -float(mean_importance.get(f, 0.0)))
)

# Save global Top-K lists with spacing enforced
for K in K_GRID:
    global_spaced = select_topK_spaced_across_families(
        ranked_by_votes, K=K, spacing=FAMILY_SPACING, relax_if_needed=True
    )
    out_fp = f"features/topK_global_majority_K{K}.csv"
    pd.DataFrame({"feature": global_spaced}).to_csv(out_fp, index=False)
    print(f"[OK] Saved majority-vote global Top-{K}: {out_fp}")

# ----------------- Heatmaps -----------------
def _plot_heatmap(df_sub, cols, title, csv_path, png_path):
    df_sub.to_csv(csv_path, index=True); print(f"[OK] Saved CSV: {csv_path}")
    if df_sub.empty:
        print(f"[WARN] Empty heatmap for '{title}'. Skipping.")
        return
    mat = df_sub[cols].to_numpy()
    fig, ax = plt.subplots(figsize=(14, max(6, 0.35 * len(df_sub))))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(df_sub))); ax.set_yticklabels(df_sub.index)
    ax.set_xticks(np.arange(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_xlabel("Model–scenario combination"); ax.set_ylabel("Feature (alphabetical, Top 50)")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax); cbar.set_label("Normalized importance (0–1)")
    fig.tight_layout(); fig.savefig(png_path, dpi=DPI); plt.close(fig)
    print(f"[OK] Saved heatmap: {png_path}")

top50_unspaced_feats = sorted(ranked_by_votes[:50])  # alphabetical for display
top50_unspaced = norm.reindex(top50_unspaced_feats).fillna(0.0)
_plot_heatmap(
    df_sub=top50_unspaced[COLS_ORDER],
    cols=COLS_ORDER,
    title="Feature votes heatmap — Top 50 by votes (no spacing), normalized",
    csv_path="heatmap_data_majority_top50_normalized_UNspaced.csv",
    png_path="figs/feature_votes_heatmap_majority_top50_normalized_UNspaced.png"
)

spaced_top50_list = select_topK_spaced_across_families(
    ranked_by_votes, K=50, spacing=FAMILY_SPACING, relax_if_needed=True
)
top50_spaced_feats = sorted(spaced_top50_list)
top50_spaced = norm.reindex(top50_spaced_feats).fillna(0.0)
_plot_heatmap(
    df_sub=top50_spaced[COLS_ORDER],
    cols=COLS_ORDER,
    title=f"Feature votes heatmap — Top 50 WITH {FAMILY_SPACING}-day spacing, normalized",
    csv_path="heatmap_data_majority_top50_normalized_SPACED.csv",
    png_path="figs/feature_votes_heatmap_majority_top50_normalized_SPACED.png"
)

print("[DONE] Global Top-K CSVs and heatmaps created.")
