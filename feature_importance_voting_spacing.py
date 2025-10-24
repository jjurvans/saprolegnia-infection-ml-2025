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
Ranks by mean importance and majority votes across folds. Enforces ±14-day spacing 
within each family (lag / rolling mean / rolling std) for the same base variable.
-Visualization: Builds normalized heatmaps showing relative importance across models/scenarios,
Generates box plot for global stability by Jaccards and Kendalls. 

Outputs
-------
- features/topK_{task}_{weighted}_fold{fold}_K{K}.csv
- features/topK_global_majority_K{K}.csv
- figs/feature_votes_heatmap_majority_top50_normalized_UNspaced.png
- figs/feature_votes_heatmap_majority_top50_normalized_SPACED.png
- heatmap_data_majority_top50_normalized_UNspaced.csv
- heatmap_data_majority_top50_normalized_spaced.csv
- stability_global_long.csv
- global_stability_boxplots.png
"""

import os, re, random, warnings
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import combinations
from scipy.stats import kendalltau
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
FAMILY_SPACING = 14               # ±day exclusion across lag/mean/std for same base var
DPI = 300
SEED = 42
K_POOL = 100

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

def parse_family_day(feat):
    """Return (family, day:int, base) or (None,None,None) if not a parsable temporal feature."""
    if not isinstance(feat, str):
        return None, None, None
    m = _FEAT_RX.match(feat)
    if not m:
        return None, None, None
    return m.group("family"), int(m.group("day")), m.group("base")

def select_topK_spaced_within_families(ranked_features, K, spacing=14, relax_if_needed=True):
    """
    Greedy spacing WITHIN families (lag / roll_mean / roll_std) for the SAME base variable.
    After selecting (base, family, day), forbid any other feature with the SAME (base, family)
    whose |day - day'| < spacing. Different families for the same base do NOT block each other.
    """
    selected = []
    used_days_by_key = defaultdict(list)  # key = (base, family) -> list of chosen days

    # first pass (strict)
    for f in ranked_features:
        if len(selected) >= K:
            break
        fam, day, base = parse_family_day(f)
        if base is None or day is None or fam is None:
            selected.append(f)
            continue
        key = (base, fam)
        if all(abs(day - d) >= spacing for d in used_days_by_key[key]):
            selected.append(f)
            used_days_by_key[key].append(day)

    if len(selected) >= K or not relax_if_needed:
        return selected[:K]

    # relaxation pass (fill up to K while preserving within-family spacing)
    used = set(selected)
    for f in ranked_features:
        if len(selected) >= K:
            break
        if f in used:
            continue
        fam, day, base = parse_family_day(f)
        if base is None or day is None or fam is None:
            selected.append(f); used.add(f)
            continue
        key = (base, fam)
        if all(abs(day - d) >= spacing for d in used_days_by_key[key]):
            selected.append(f); used.add(f)
            used_days_by_key[key].append(day)

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
fold_mean_ranks = []
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
    spaced_order = select_topK_spaced_within_families(ranked_features, K=K_POOL,
                                                      spacing=FAMILY_SPACING, relax_if_needed=True)
    fold_mean_ranks.append(ranks.mean(axis=1))
    fold_spaced_orders.append(spaced_order)

    # Save per-scenario Top-K feature lists for this fold (same spaced order used for all scenarios)
    for (task, weighted) in SCENARIOS:
        for K in K_GRID:
            topK = spaced_order[:K]
            out_fp = f"features/topK_{task}_{int(weighted)}_fold{fold}_K{K}.csv"
            pd.DataFrame({"feature": topK}).to_csv(out_fp, index=False)

# ----------------- Build per-scenario importance table (fresh, no cache) -----------------
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

# Optionally persist raw averaged per-scenario columns (not used in this run)
for c in COLS_ORDER:
    pd.DataFrame({c: wide[c]}).to_csv(f"importance_{c}.csv")  # raw, not reloaded here

# ----------------- Majority voting across folds -----------------
votes_all = Counter()
for spaced in fold_spaced_orders:
    votes_all.update(spaced[:K_POOL])
print("[diag] spaced lengths per fold:", [len(lst) for lst in fold_spaced_orders])

# Tie-breaker: global mean normalized importance (higher better)
mean_importance = norm.mean(axis=1)

ranked_by_votes = sorted(
    votes_all.keys(),
    key=lambda f: (-votes_all[f], -float(mean_importance.get(f, 0.0)))
)

# Save global Top-K lists with spacing enforced
for K in K_GRID:
    global_spaced = select_topK_spaced_within_families(
        ranked_by_votes, K=K, spacing=FAMILY_SPACING, relax_if_needed=True
    )
    out_fp = f"features/topK_global_majority_K{K}.csv"
    pd.DataFrame({"feature": global_spaced}).to_csv(out_fp, index=False)
    print(f"[OK] Saved majority-vote global Top-{K}: {out_fp}")

# ----------------- Heatmaps -----------------
def _plot_heatmap(df_sub, cols, y_label, csv_path, png_path):
    # Save CSV used in the plot
    df_sub.to_csv(csv_path, index=True)
    print(f"[OK] Saved CSV: {csv_path}")

    if df_sub.empty:
        print(f"[WARN] Empty heatmap for y-axis '{y_label}'. Skipping.")
        return

    mat = df_sub[cols].to_numpy()
    fig, ax = plt.subplots(figsize=(14, max(6, 0.35 * len(df_sub))))

    # Half-width color range around 0.5 (0.25–0.75) for better contrast
    im = ax.imshow(mat, aspect="auto", vmin=0.25, vmax=0.75)

    ax.set_yticks(np.arange(len(df_sub)))
    ax.set_yticklabels(df_sub.index)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_xlabel("Model–scenario combination")
    ax.set_ylabel(y_label)  # custom y-axis label; no title

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized importance (0–1)")

    fig.tight_layout()
    fig.savefig(png_path, dpi=DPI)
    plt.close(fig)
    print(f"[OK] Saved heatmap: {png_path}")

# --- Unspaced Top-50
top50_unspaced_feats = sorted(ranked_by_votes[:50])  # alphabetical for display
top50_unspaced = norm.reindex(top50_unspaced_feats).fillna(0.0)
_plot_heatmap(
    df_sub=top50_unspaced[COLS_ORDER],
    cols=COLS_ORDER,
    y_label="Top 50 labels unspaced",
    csv_path="heatmap_data_majority_top50_normalized_UNspaced.csv",
    png_path="figs/feature_votes_heatmap_majority_top50_normalized_UNspaced.png"
)

# --- Spaced Top-50
spaced_top50_list = select_topK_spaced_within_families(
    ranked_by_votes, K=50, spacing=FAMILY_SPACING, relax_if_needed=True
)
top50_spaced_feats = sorted(spaced_top50_list)
top50_spaced = norm.reindex(top50_spaced_feats).fillna(0.0)
_plot_heatmap(
    df_sub=top50_spaced[COLS_ORDER],
    cols=COLS_ORDER,
    y_label="Top 50 labels spaced",
    csv_path="heatmap_data_majority_top50_normalized_SPACED.csv",
    png_path="figs/feature_votes_heatmap_majority_top50_normalized_SPACED.png"
)
# ----- Global stability across outer folds (Jaccard & Kendall) + box-plots -----

# Clamp K so all folds have at least K spaced features
if len(fold_spaced_orders) >= 2:
    maxK = min(max(K_GRID), min(len(lst) for lst in fold_spaced_orders))
    K_eff = [K for K in K_GRID if K <= maxK]
else:
    K_eff = []

def _pairwise_jaccards(list_of_sets):
    vals = []
    for a, b in combinations(list_of_sets, 2):
        u = len(a | b); i = len(a & b)
        vals.append(i / u if u > 0 else 1.0)
    return np.asarray(vals, float)

def _pairwise_kendalls(rank_series_list, idx):
    vals = []
    fillv = len(idx) + 1
    for a, b in combinations(rank_series_list, 2):
        r1 = a.reindex(idx).fillna(fillv).to_numpy()
        r2 = b.reindex(idx).fillna(fillv).to_numpy()
        tau, _ = kendalltau(r1, r2, nan_policy="omit")
        vals.append(tau)
    return np.asarray(vals, float)

stability_long = []
if len(K_eff) > 0:
    for K in K_eff:
        # Jaccard on Top-K sets from each fold
        sets_K = [set(lst[:K]) for lst in fold_spaced_orders]
        j_vals = _pairwise_jaccards(sets_K)

        # Kendall on ranks restricted to union of Top-K across folds
        union_topK = sorted(set().union(*sets_K))
        idx_use = pd.Index(union_topK) if union_topK else pd.Index([])
        t_vals = _pairwise_kendalls(fold_mean_ranks, idx_use) if len(idx_use) > 0 else np.array([])

        for v in j_vals:
            stability_long.append(dict(metric="Jaccard", K=K, value=float(v)))
        for v in t_vals:
            stability_long.append(dict(metric="KendallTau", K=K, value=float(v)))

stab_long_df = pd.DataFrame(stability_long)
stab_long_df.to_csv("stability_global_long.csv", index=False)
print("[OK] Saved stability_global_long.csv")

# ----- Box-plots by metric (global, not per-scenario) -----
if not stab_long_df.empty:
    os.makedirs("figs", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Jaccard
    j = stab_long_df[stab_long_df.metric == "Jaccard"]
    axes[0].boxplot(
        [j[j.K == K]["value"].to_numpy() for K in K_eff],
        tick_labels=[str(K) for K in K_eff],  # <- renamed
        showmeans=True
        ) 
    axes[0].set_title("Jaccard (pairwise across outer folds)")
    axes[0].set_xlabel("Top-K"); axes[0].set_ylabel("Jaccard index"); axes[0].grid(alpha=0.3)

    # Right: Kendall τ
    k = stab_long_df[stab_long_df.metric == "KendallTau"]
    if k.empty:
        axes[1].text(0.5, 0.5, "Kendall τ not available", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].axis("off")
    else:
        axes[1].boxplot(
            [k[k.K == K]["value"].to_numpy() for K in K_eff],
            tick_labels=[str(K) for K in K_eff],  # <- renamed
            showmeans=True
            )  
        axes[1].set_title("Kendall τ (pairwise across outer folds)")
        axes[1].set_xlabel("Top-K"); axes[1].set_ylabel("Kendall τ"); axes[1].grid(alpha=0.3)

    fig.suptitle("Global selection stability across outer folds")
    fig.tight_layout()
    fig.savefig("figs/global_stability_boxplots.png", dpi=DPI)
    plt.close(fig)
    print("[OK] Saved figs/global_stability_boxplots.png")
else:
    print("[WARN] Not enough folds or K values for stability.")

print("[DONE] Global Top-K CSVs and heatmaps created.")
