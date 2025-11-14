#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 08:55:40 2025

@author: jaanajurvansuu

Final Random Forest modeling pipeline for Saprolegnia infection prediction.

Performs deterministic training, time-lag and rolling feature engineering, sample weighting
(inverse-variance), SHAP-based interpretation, block permutation importance, probability calibration,
and calibrated forecasting on new time-series data.

Required input files:
---------------------
- SaprolegniaInfection_LabeledData_2025.csv: historical labeled dataset with columns: 
     Timestamp, InfectionDensity, InfectionDensity_var, and environmental predictors.
- Features/topK_global_majority_K50.csv: list of selected engineered features (one per line).
- TestFinal.csv: new unlabeled data for prediction (same structure as historical data).
- NewInfectionDensity.csv: observed infection densities for new data (semicolon-separated, European dates, '%' allowed).

Main outputs (written to `final_rf_outputs/`):
----------------------------------------------
- shap_plots/shap_beeswarm.png – global SHAP summary (feature importance).
- shap_dependence/*.png – top feature SHAP dependence plots.
- perm_importance/perm_importance_brier.{csv,png} – block permutation importance results.
- predictions_new_calibrated.csv – calibrated infection probabilities (+ truth if provided).
- prediction_line_new_calibrated_truth_zeros_marked.png – time series plot of predicted probabilities
   vs. observed infection density (zero-density days highlighted).

"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- determinism ----------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(k, "1")

# -------------- config --------------
EXCLUDE_COLS = {
    "Timestamp","InfectionDensity","InfectionDensity_var",
    "SoftLabel","SoftLabel_Occurrence","SoftLabelOccurrence_var"
}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 14,
    "axes.labelsize": 14, "axes.titlesize": 14,
    "legend.fontsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14
})

OUT_DIR = Path("final_rf_outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("SaprolegniaInfection_LabeledData_2025.csv", parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

# -------------- feature engineering (strict past-only) --------------
def build_time_features(df, base_cols, max_lag=28, use_roll_mean=True, use_roll_std=True):
    frames = []
    for col in base_cols:
        lag_df = pd.concat({f"{col}_lag{k}": df[col].shift(k) for k in range(1, max_lag+1)}, axis=1)
        frames.append(lag_df)
        past = df[col].shift(1)
        if use_roll_mean:
            frames.append(pd.concat({f"{col}_roll_mean{w}": past.rolling(w, min_periods=1).mean()
                                     for w in range(1, max_lag+1)}, axis=1))
        if use_roll_std:
            frames.append(pd.concat({f"{col}_roll_std{w}": past.rolling(w, min_periods=1).std(ddof=0)
                                     for w in range(1, max_lag+1)}, axis=1))
    return pd.concat(frames, axis=1)

# -------------- weights (stabilized) --------------
def inverse_variance_weights(var_series, eps=1e-6):
    v = np.asarray(pd.Series(var_series).astype(float))
    v = np.where(np.isfinite(v), v, np.nanmedian(v))
    v = np.maximum(v, eps)
    w = 1.0 / (v + eps)
    return w / np.mean(w)

# -------------- prepare design --------------
def prepare_design(df: pd.DataFrame, selected_features_csv= "features/topK_global_majority_K50.csv", max_lag=28):
    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    base_cols = [c for c in numeric_cols if c not in EXCLUDE_COLS]
    if not base_cols: raise RuntimeError("No numeric base features after exclusions.")
    X_time = build_time_features(df, base_cols, max_lag=max_lag, use_roll_mean=True, use_roll_std=True)
    df_work = pd.concat([df.reset_index(drop=True), X_time.reset_index(drop=True)], axis=1)
    df_work = df_work.iloc[max_lag:].reset_index(drop=True)
    X_time  = X_time.iloc[max_lag:].reset_index(drop=True)

    y_hard = (df_work["InfectionDensity"].astype(float).to_numpy() > 0).astype(int)
    w_arr = inverse_variance_weights(df_work["InfectionDensity_var"].astype(float).to_numpy()) \
            if "InfectionDensity_var" in df_work.columns else None

    sel = pd.read_csv(selected_features_csv).iloc[:,0].tolist()
    X_full = X_time[sel].copy()

    mask = np.isfinite(X_full.to_numpy()).all(axis=1) & np.isfinite(y_hard)
    X = X_full.loc[mask].reset_index(drop=True)
    y = y_hard[mask]
    w = (w_arr[mask] if w_arr is not None else None)

    tstamp = None
    if "Timestamp" in df.columns:
        tstamp = pd.to_datetime(df["Timestamp"]).iloc[max_lag:].reset_index(drop=True)
        tstamp = tstamp.loc[mask].reset_index(drop=True)

    return X, y, w, tstamp, base_cols


# ---------- SHAP: always explain the post-imputation matrix as an Explanation ----------
def compute_shap_explanation(pipe, X_df: pd.DataFrame):
    imp = pipe.named_steps["simpleimputer"]
    X_imp = pd.DataFrame(imp.transform(X_df), columns=X_df.columns, index=X_df.index)

    rf = pipe.named_steps["randomforestclassifier"]
    expl = shap.TreeExplainer(rf, feature_names=list(X_imp.columns))

    sv = expl.shap_values(X_imp, check_additivity=False)  # can be list, 2D, or 3D

    def _as_2d_values_and_base(sv_any, X_imp, expl):
        # list case: pick positive class if present
        if isinstance(sv_any, list):
            cls_idx = 1 if len(sv_any) > 1 else 0
            vals = np.asarray(sv_any[cls_idx])
            base = expl.expected_value[cls_idx] if isinstance(expl.expected_value, (list, tuple, np.ndarray)) else expl.expected_value
            return vals, float(base)

        arr = np.asarray(sv_any)
        if arr.ndim == 2:
            base = expl.expected_value
            if isinstance(base, (list, tuple, np.ndarray)):
                base = float(base[-1])
            return arr, float(base)

        if arr.ndim == 3:
            n, a, b = arr.shape
            d = X_imp.shape[1]
            # common: (n_samples, n_classes, n_features)
            if b == d:
                cls_idx = 1 if a > 1 else 0
                vals = arr[:, cls_idx, :]
                base = expl.expected_value
                if isinstance(base, (list, tuple, np.ndarray)):
                    base = float(base[cls_idx if a > 1 else 0])
                return vals, float(base)
            # less common: (n_samples, n_features, n_outputs)
            if a == d:
                out_idx = 0 if b == 1 else (b - 1)
                vals = arr[:, :, out_idx]
                base = expl.expected_value
                if isinstance(base, (list, tuple, np.ndarray)):
                    base = float(base[out_idx])
                return vals, float(base)

        raise ValueError(f"Unexpected shap_values shape {arr.shape}; X has {X_imp.shape[1]} features.")

    values, base = _as_2d_values_and_base(sv, X_imp, expl)
    if values.shape[1] != X_imp.shape[1]:
        raise ValueError(f"SHAP values have {values.shape[1]} features but X has {X_imp.shape[1]}.")

    n = values.shape[0]
    exp_all = shap.Explanation(
        values=values,
        base_values=np.full(n, base, dtype=float),
        data=X_imp.to_numpy(),
        feature_names=list(X_imp.columns),
    )
    exp_one = shap.Explanation(
        values=values[-1],
        base_values=float(base),
        data=X_imp.iloc[-1, :].to_numpy(),
        feature_names=list(X_imp.columns),
    )
    return exp_all, exp_one, X_imp

def save_shap_plots(pipe, X_df: pd.DataFrame, out_dir: Path, topn_waterfall: int = 15):
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_all, exp_one, X_imp = compute_shap_explanation(pipe, X_df)

    # Beeswarm (classical)
    plt.figure(figsize=(9,6))
    shap.summary_plot(exp_all, show=False)        # pass Explanation directly
    plt.tight_layout(); plt.savefig(out_dir/"shap_beeswarm.png", dpi=300); plt.close()

# ---------- SHAP dependence plots (top-K) ----------
def plot_shap_dependence_quantile_x(
    feature: str,
    exp_all,                   # shap.Explanation or np.ndarray
    X_imp,                     # pd.DataFrame (post-impute, aligned to exp_all)
    color_feature: str | None = "auto",
    K: int = 20,
    show_mean: bool = True
):
    # --- arrays ---
    j = X_imp.columns.get_loc(feature)
    S = exp_all.values if hasattr(exp_all, "values") else exp_all  # (n, p)
    shap_vals_f = S[:, j]
    x_raw = X_imp.iloc[:, j].to_numpy()

    # --- auto interaction feature (like shap.dependence_plot) ---
    if color_feature == "auto":
        try:
            order = shap.utils.approximate_interactions(j, S, X_imp.values)
            k = next((idx for idx in order if idx != j), None)
            color_feature = X_imp.columns[k] if k is not None else None
        except Exception:
            color_feature = None

    # --- color values ---
    cvals = None; cat_labels = None
    if isinstance(color_feature, str) and color_feature in X_imp.columns:
        c = X_imp[color_feature].to_numpy()
        # simple categorical detection (few uniques)
        if np.isfinite(c).sum() and (np.unique(c[~np.isnan(c)]).size <= 8 or X_imp[color_feature].dtype == "object"):
            # map categories to 0..K and show legend
            cats, inv = np.unique(c.astype(str), return_inverse=True)
            cvals = inv
            cat_labels = cats
        else:
            cvals = c

    # --- quantile bins for x-axis ---
    finite = np.isfinite(x_raw)
    edges = np.unique(np.quantile(x_raw[finite], np.linspace(0, 1, K + 1))) if finite.any() else np.linspace(0, 1, K + 1)
    if len(edges) < 3:
        lo, hi = np.nanmin(x_raw), np.nanmax(x_raw)
        edges = np.unique(np.linspace(lo, hi, num=K + 1))
    mids = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.searchsorted(edges, x_raw, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)
    qx = bin_idx + 0.5

    # --- plot ---
    fig, ax = plt.subplots(figsize=(6, 4))
    # custom colormap: blue → violet → red
    cmap_bvr = LinearSegmentedColormap.from_list(
        "blue_violet_red", ["#0088ff", "#6a29a6", "#ff084a"], N=256
    )

    # robust normalization for vivid color contrast
    norm = None
    if cvals is not None and np.issubdtype(type(cvals[0]), np.number):
        finite = np.isfinite(cvals)
        if finite.any():
            p5, p95 = np.nanpercentile(cvals[finite], [5, 95])
            norm = Normalize(vmin=p5, vmax=p95, clip=True)

    # main scatter
    sc = ax.scatter(
        qx, shap_vals_f, s=10, alpha=1.0, c=cvals, cmap=cmap_bvr, norm=norm
    )

    # colorbar or categorical legend
    if cvals is not None and cat_labels is None:
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(color_feature)
    elif cat_labels is not None:
        # categorical legend
        handles = []
        for ci, lab in enumerate(cat_labels):
            handles.append(plt.Line2D([0],[0], marker='o', lw=0, color=cmap_bvr(ci / len(cat_labels)), label=str(lab)))
        ax.legend(handles=handles, title=color_feature, loc="best", frameon=False)

    # mean line
    if show_mean:
        means = np.full(len(edges) - 1, np.nan)
        for k in range(len(edges) - 1):
            m = shap_vals_f[bin_idx == k]
            if m.size:
                means[k] = np.mean(m)
        ax.plot(np.arange(0.5, len(edges) - 0.5), means, lw=1.8, color="black")

    ax.axhline(0, ls="--", alpha=0.6, lw=0.8)
    ax.set_xlim(0, len(edges) - 1)
    ttl = f"SHAP dependence – {feature} (quantile x-axis)"
    if isinstance(color_feature, str):
        ttl += f" | color: {color_feature}"
    #ax.set_title(ttl)
    ax.set_ylabel("SHAP value")


    # secondary axis in real units
    n_bins = len(edges) - 1
    tick_q = np.linspace(0.5, n_bins - 0.5, num=min(6, n_bins))
    mids_idx = np.clip(np.round(tick_q - 0.5).astype(int), 0, n_bins - 1)
    tick_vals = mids[mids_idx]
    secax = ax.secondary_xaxis("bottom")
    secax.set_xticks(tick_q); secax.set_xticklabels([f"{v:.3g}" for v in tick_vals])
    secax.set_xlabel(feature)
    ax.set_xticks(tick_q); ax.set_xticklabels([])

    fig.tight_layout()
    return fig, ax

def save_shap_dependence_topk(pipe, X: pd.DataFrame, out_dir: Path, topk: int = 8, K_quant: int = 20):
    out_dir = Path(out_dir) / "shap_dependence"; out_dir.mkdir(parents=True, exist_ok=True)
    exp_all, _, X_imp = compute_shap_explanation(pipe, X)

    mean_abs = np.abs(getattr(exp_all, "values", exp_all)).mean(axis=0)
    feats = [X_imp.columns[i] for i in np.argsort(-mean_abs)[:min(topk, X_imp.shape[1])]]

    for f in feats:
        # quantile-x variant
        fig, ax = plot_shap_dependence_quantile_x(f, exp_all, X_imp, K=K_quant, show_mean=True)
        fig.savefig(out_dir / f"shap_dependence_quantile_{f}.png", dpi=300); plt.close(fig)

    pd.Series(feats, name="feature").to_csv(out_dir / "dependence_features.csv", index=False)

# ---------- Block permutation importance ----------
def _metric_delta(y_true, p_base, p_perm, which="brier"):
    if which == "brier":
        return float(brier_score_loss(y_true, p_perm) - brier_score_loss(y_true, p_base))
    if which == "logloss":
        return float(log_loss(y_true, np.clip(p_perm, 1e-12, 1-1e-12), labels=[0,1]) -
                     log_loss(y_true, np.clip(p_base, 1e-12, 1-1e-12), labels=[0,1]))
    raise ValueError("metric must be 'brier' or 'logloss'")

def block_permutation_importance(pipe, X: pd.DataFrame, y: np.ndarray,
                                 block_size: int = 14, n_perm: int = 200, metric: str = "brier", seed: int = 42):
    rng = np.random.default_rng(seed)
    p_base = pipe.predict_proba(X)[:, 1]
    n = len(X); n_blocks = max(1, n // block_size)
    starts = np.arange(0, n_blocks * block_size, block_size)
    blocks = [slice(s, min(s + block_size, n)) for s in starts]
    if blocks[-1].stop < n: blocks.append(slice(blocks[-1].stop, n))
    rows = []
    for feat in X.columns:
        deltas = []
        for _ in range(n_perm):
            order = rng.permutation(len(blocks))
            Xp = X.copy()
            col = np.concatenate([X[feat].values[blocks[j]] for j in order], axis=0)
            Xp[feat] = col
            p_perm = pipe.predict_proba(Xp)[:, 1]
            deltas.append(_metric_delta(y, p_base, p_perm, which=metric))
        deltas = np.array(deltas, float)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        rows.append({"feature": feat, f"delta_{metric}": float(np.mean(deltas)),
                     "ci_low": float(lo), "ci_high": float(hi)})
    return pd.DataFrame(rows).sort_values(f"delta_{metric}", ascending=False)

def save_block_perm_plots(pipe, X: pd.DataFrame, y: np.ndarray, out_dir: Path,
                          block_size: int = 14, n_perm: int = 200, metric: str = "brier", seed: int = 42):
    out_dir = Path(out_dir) / "perm_importance"; out_dir.mkdir(parents=True, exist_ok=True)
    df_imp = block_permutation_importance(pipe, X, y, block_size=block_size, n_perm=n_perm, metric=metric, seed=seed)
    df_imp.to_csv(out_dir / f"perm_importance_{metric}.csv", index=False)
    show = df_imp.head(20)
    plt.figure(figsize=(8.5, max(3.0, 0.35*len(show))))
    y_pos = np.arange(len(show))
    plt.barh(y_pos, show[f"delta_{metric}"], xerr=[show[f"delta_{metric}"]-show["ci_low"],
                                                   show["ci_high"]-show[f"delta_{metric}"]])
    plt.gca().invert_yaxis()
    plt.xlabel(f"Δ {metric} (perm - base) ↑ worse")
    plt.title(f"Block permutation importance ({metric})")
    plt.yticks(y_pos, show["feature"])
    plt.tight_layout(); plt.savefig(out_dir / f"perm_importance_{metric}.png", dpi=300); plt.close()

# -------------- fit final RF, metrics, SHAP --------------

def run_final_rf_with_df(
    df: pd.DataFrame,
    selected_features_csv: str = "features/topK_global_majority_K50.csv",
    max_lag: int = 28,
    rf_params_user: dict | None = None,
    df_new: pd.DataFrame | None = None
):
    # 1) Prepare train design
    X, y, w, tstamp, base_cols = prepare_design(df, selected_features_csv, max_lag=max_lag)

    # 2) RF params (strict: user must supply)
    if not rf_params_user:
        raise ValueError(
        "No RF parameters provided. Please pass rf_params_user to run_final_rf_with_df()."
    )

    rf_params = rf_params_user.copy()

    # 3) Fit model
    pipe = Pipeline([
        ("simpleimputer", SimpleImputer(strategy="median")),
        ("randomforestclassifier", RandomForestClassifier(**rf_params)),
    ])
    fit_kwargs = {}
    if w is not None:
        fit_kwargs["randomforestclassifier__sample_weight"] = w
    pipe.fit(X, y, **fit_kwargs)
    pipe._train_X = X.copy(); pipe._train_y = y.copy()

    # 4) SHAP dependence (top-K) and beeswarm
    save_shap_dependence_topk(pipe, X, OUT_DIR, topk=8)
    print(f"SHAP dependence -> {(OUT_DIR/'shap_dependence').resolve()}")

    shap_dir = OUT_DIR / "shap_plots"
    save_shap_plots(pipe, X, shap_dir)
    print(f"[SHAP] saved -> {shap_dir}")
    
    # 5) Block permutation importance
    save_block_perm_plots(pipe, X, y, OUT_DIR, block_size=14, n_perm=200, metric="brier", seed=42)
    print(f"Permutation importance -> {(OUT_DIR/'perm_importance').resolve()}")

    # 6) predict on new data (time-aware sigmoid calibration; optional separate InfectionDensity CSV)
    if df_new is not None:


        # ---------- A) build calibrated predictor from historical X,y (no leakage) ----------
        calib_holdout_frac = 0.20
        n = len(X)
        split = int(round(n * (1.0 - calib_holdout_frac)))
        if split <= 0 or split >= n:
            raise ValueError(f"Bad calibration split: split={split}, n={n}")

        X_fit, y_fit = X.iloc[:split], y[:split]
        X_cal, y_cal = X.iloc[split:], y[split:]

        # ALWAYS define w_fit safely
        w_fit = None
        try:
            if w is not None:
                w_fit = np.asarray(w)[:split]
        except NameError:
            # 'w' not defined in scope -> keep w_fit=None
            pass

        base_for_cal = clone(pipe)  # keep SHAP/perm 'pipe' untouched
        
        fit_kwargs_base = {}
        if w_fit is not None:
            fit_kwargs_base["randomforestclassifier__sample_weight"] = w_fit
    
        base_for_cal.fit(X_fit, y_fit, **fit_kwargs_base)
        
        # Calibrator (version-agnostic)
        try:
            cal = CalibratedClassifierCV(estimator=base_for_cal, cv="prefit", method="sigmoid")
        except TypeError: 
            cal = CalibratedClassifierCV(base_estimator=base_for_cal, cv="prefit", method="sigmoid")
        cal.fit(X_cal, y_cal)  # sklearn ignores sample_weight here

        # ---------- B) prepare NEW data features (ensure X_new is defined) ----------
        # load df_new
        if isinstance(df_new, (str, Path)):
            if not Path(df_new).exists():
                raise FileNotFoundError(f"df_new file not found: {df_new}")
                # try to parse Timestamp if present
            try:
                df_new = pd.read_csv(df_new, sep=';', parse_dates=["Timestamp"])
            except Exception:
                df_new = pd.read_csv(df_new)

        # build engineered features
        X_time_new = build_time_features(
            df_new, base_cols, max_lag=max_lag, use_roll_mean=True, use_roll_std=True
            )
        if len(X_time_new) <= max_lag:
            raise ValueError(f"df_new needs ≥ {max_lag+1} rows to create lagged features (has {len(df_new)}).")
        X_time_new = X_time_new.iloc[max_lag:].reset_index(drop=True)
    
        # feature selection intersection
        sel = pd.read_csv(selected_features_csv).iloc[:, 0].astype(str).tolist()
        missing = [c for c in sel if c not in X_time_new.columns]
        if missing:
            raise RuntimeError(
                f"New data missing engineered features (first): {missing[:6]}"
                )
        X_new = X_time_new[sel].copy()  # <-- X_new DEFINITELY defined here

        # timestamps for plotting/export
        if "Timestamp" in df_new.columns:
            tstamp_new = pd.to_datetime(df_new["Timestamp"], errors="coerce").iloc[max_lag:].reset_index(drop=True)
        else:
            tstamp_new = pd.RangeIndex(len(X_new))

        # ---------- C) calibrated prediction on new data ----------
        p_new = cal.predict_proba(X_new)[:, 1]

        # ------- OPTIONAL: load separate InfectionDensity CSV and align by date -------
        infection_csv = "NewInfectionDensity.csv"
        y_true_new = None

        if isinstance(tstamp_new, (pd.Series, pd.DatetimeIndex)):
            # base plotting frame (always defined)
            df_plot = pd.DataFrame({"Timestamp": pd.to_datetime(tstamp_new, errors="coerce")})
            df_plot["Date"] = df_plot["Timestamp"].dt.floor("D")

            if Path(infection_csv).exists():
                # 1) read with semicolon; robust cleaning
                df_inf = pd.read_csv(infection_csv, sep=";")

                # standardize column names (trim spaces)
                df_inf.columns = [c.strip() for c in df_inf.columns]
                if "Timestamp" not in df_inf.columns or "InfectionDensity" not in df_inf.columns:
                    raise ValueError("NewInfectionDensity.csv must have 'Timestamp' and 'InfectionDensity' columns.")

                # 2) parse European date format (dd.mm.yyyy), coerce errors -> NaT
                df_inf["Timestamp"] = pd.to_datetime(df_inf["Timestamp"].astype(str).str.strip(),
                                             dayfirst=True, errors="coerce")
                # align to date for merging
                df_inf["Date"] = df_inf["Timestamp"].dt.floor("D")

                # 3) clean density: strip %, replace comma decimal (if any), to numeric
                dens = (df_inf["InfectionDensity"].astype(str).str.strip()
                    .str.replace("%", "", regex=False)
                    .str.replace(",", ".", regex=False))
                df_inf["InfectionDensity"] = pd.to_numeric(dens, errors="coerce")

                # 4) handle duplicate dates (pick max; change to 'mean' if preferred)
                df_inf_daily = (df_inf.dropna(subset=["Date"])
                              .groupby("Date", as_index=False)["InfectionDensity"].max())

                # 5) merge left on Date (keeps prediction order)
                df_plot = df_plot.merge(df_inf_daily, on="Date", how="left")

                # final series for plotting/CSV (can be NaN on unobserved days)
                y_true_new = df_plot["InfectionDensity"]
            else:
                # no infection CSV -> keep only timeline
                df_plot = df_plot[["Timestamp", "Date"]]
        else:
            # predictions are index-based (no timestamps)
            df_plot = pd.DataFrame({"Index": np.arange(len(p_new))})

        # ---------- E) save predictions (+ truth if present) ----------
        if "Timestamp" in df_plot.columns:
            pred_df = pd.DataFrame({"Timestamp": df_plot["Timestamp"], "p_infection_calibrated": p_new})
        else:
            pred_df = pd.DataFrame({"Index": df_plot["Index"], "p_infection_calibrated": p_new})
        if y_true_new is not None:
                pred_df["infection_density_true"] = y_true_new
        pred_df.to_csv(OUT_DIR / "predictions_new_calibrated.csv", index=False)

        # ---------- F) plot ----------
        plt.figure(figsize=(10, 4))

        # time axis (Timestamp or Index)
        if "Timestamp" in df_plot.columns and pd.api.types.is_datetime64_any_dtype(df_plot["Timestamp"]):
            x_axis = df_plot["Timestamp"]
        else:
            x_axis = np.arange(len(p_new))

        # --- predicted probabilities ---
        plt.plot(x_axis, p_new, lw=2, label="Predicted P(infection) (calibrated)", color="tab:blue")

        # --- observed infection density (if available) ---
        if "InfectionDensity" in df_plot.columns:
            y_arr = df_plot["InfectionDensity"].to_numpy(dtype=float)
            finite_mask = np.isfinite(y_arr)
            if finite_mask.any():
                ymax = float(np.nanmax(y_arr))
                if ymax > 0:
                    y_scaled = (y_arr / ymax)
                    plt.bar(np.array(x_axis)[finite_mask], y_scaled[finite_mask], width=0.8,
                            alpha=0.35, label="Observed infection density (scaled)",
                            color="tab:orange")

                    # --- mark observed zeros with red circle outline ---
                    zero_mask = np.isclose(y_arr, 0.0, atol=1e-12)
                    if np.any(zero_mask):
                        plt.scatter(np.array(x_axis)[zero_mask],
                            [0.02]*np.sum(zero_mask),
                            facecolors='none', edgecolors='red', s=60,
                            label="Observed zero density", zorder=4)
                else:
                    zero_mask = np.isclose(y_arr, 0.0, atol=1e-12)
                    if np.any(zero_mask):
                        plt.scatter(np.array(x_axis)[zero_mask],
                            [0.02]*np.sum(zero_mask),
                            facecolors='none', edgecolors='red', s=60,
                            label="Observed zero density", zorder=4)

        # --- threshold line ---
        plt.axhline(0.5, ls="--", lw=1, color="gray", label="0.5 threshold")

        plt.xlabel("Date" if "Timestamp" in df_plot.columns else "Index")
        plt.ylabel("Probability / normalized density")
        plt.title("Predicted infection probability (calibrated) vs. observed density")
        plt.ylim(0, 1.05)
        plt.legend(frameon=False)

        # --- rotate x-axis labels 90° ---
        if "Timestamp" in df_plot.columns and pd.api.types.is_datetime64_any_dtype(df_plot["Timestamp"]):
            plt.xticks(rotation=90, ha='center')

        plt.tight_layout()
        plt.savefig(OUT_DIR / "prediction_line_new_calibrated_truth_zeros_marked.png", dpi=300)
        plt.close()

        print(f"[new] calibrated + truth plot with zero markers -> {OUT_DIR/'prediction_line_new_calibrated_truth_zeros_marked.png'}")
        print(f"[new] calibrated predictions -> {OUT_DIR/'predictions_new_calibrated.csv'}")
        print(f"[new] calibrated + truth plot -> {OUT_DIR/'prediction_line_new_calibrated_truth_separateCSV.png'}")

# ---------------- run ----------------
pipe = run_final_rf_with_df(df, selected_features_csv="features/topK_global_majority_K50.csv", max_lag=28,
                            rf_params_user={
                                "n_estimators": 1000, "max_depth": 3, "min_samples_leaf": 4,
                                "min_samples_split": 2, "max_features": 0.5,
                                "bootstrap": True, "max_samples": None,
                                "criterion": "log_loss", "class_weight": None,
                                "min_impurity_decrease": 0.0, "ccp_alpha": 0.0,
                                "random_state": SEED, "n_jobs": -1
                            },
                            df_new="TestFinal.csv")
