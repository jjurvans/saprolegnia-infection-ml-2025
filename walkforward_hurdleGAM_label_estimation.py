#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 12:37:20 2025
@author: jaanajurvansuu
Title: Daily infection labels (hurdle-GAM) + soft labels from occurrence GAM

Overall:
    This script takes:
      - Water feature data aggregated to daily resolution
          -> file: WaterAnalysisAggregatedData.csv
      - Daily removal/effort log
          -> file: "SaprolengiaInfections24_25.csv"  (columns: Timestamp, Removed, Volume)

    It estimates daily infection *density* using a two-part hurdle model:
      1) Occurrence model  π_t = P(Removed > 0 | day)           [LogisticGAM]
      2) Intensity model   r_t = E[Removed/Volume | Removed>0]  [GammaGAM]

    Then it:
      - Combines parts: μ_t = π_t * r_t  (expected daily density)
      - Provides uncertainty via coefficient draws (mean, var, 95% CI)
      - Constructs daily counts:
          • observed counts on Effort=1 days (truth)
          • model-based counts elsewhere using Volume (with configurable fill)
      - Applies a conservative LOD + hysteresis rule to zero sustained very-low densities 
      which is based on real detection limit (less than one infected fish in any pool).
      - Exports “soft labels” = probability of positive (occurrence GAM)
      - Produces figures and a final merged CSV for ML use

Inputs:
    WaterAnalysisAggregatedData.csv   (daily covariates/features)
    EnonkoskiSaph24_25.csv            (Timestamp, Removed, Volume)

Outputs:
    SaprolegniaInfection_LabeledData_2025.csv
    soft_labels_timeseries.png
    occurrence_softlabel_timeseries.png

Licensing (example):
    Code: MIT
"""

# ----------------------------
# Imports
# ----------------------------
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.special import expit
import scipy.sparse as sp
from pygam import LogisticGAM, GammaGAM, s

# ----------------------------
# Config
# ----------------------------
RANDOM_SEED = 123
DEFAULT_VOLUME_FILL = "median"  # "median" or numeric e.g., 100.0
TIME_SPLINES = 24               # smoothness over time
TIME_LAM     = 50               # penalty over time (↑ smoother)
WDAY_SPLINES = 7                # cyclic weekday smooth
WDAY_LAM     = 200              # penalty for weekday effect

FIG  ="figures"
OUT  ="outputs"

FEATURES_FILE = "WaterAnalysisAggregatedData.csv"
EFFORT_FILE   = "SaprolengiaInfections24_25.csv"
FINAL_FILE    = "SaprolegniaInfection_LabeledData_2025.csv"

rng = default_rng(RANDOM_SEED)

# Restore old .A alias on sparse matrices (pyGAM compatibility on some SciPy versions)
for cls in (sp.csr_matrix, sp.csc_matrix, sp.coo_matrix):
    if not hasattr(cls, "A"):
        cls.A = property(cls.toarray)


# ----------------------------
# 0) Load data (parse Timestamp)  — Explanation:
#    • Ensure Timestamp is datetime
#    • Sort, drop invalid rows
#    • Optional timezone handling stub
# ----------------------------
df_feat = pd.read_csv(FEATURES_FILE, parse_dates=["Timestamp"])
df_eff  = pd.read_csv(EFFORT_FILE,   parse_dates=["Timestamp"])

# Drop invalid timestamps and sort
df_feat = df_feat.dropna(subset=["Timestamp"]).sort_values("Timestamp")
df_eff  = df_eff.dropna(subset=["Timestamp"]).sort_values("Timestamp")

# Normalize effort dates to midnight (daily key)
df_eff["Timestamp"] = df_eff["Timestamp"].dt.normalize()

# Sanity check
for c in ["Removed", "Volume"]:
    if c not in df_eff.columns:
        raise ValueError(f"Column '{c}' missing in {EFFORT_FILE.name}")


# ----------------------------
# 1) Build daily calendar and effort — Explanation:
#    • Effort day = day appears in df_eff (even if Removed=0)
#    • Aggregate Removed, Volume by day
#    • Merge to full daily calendar to cover gaps
# ----------------------------
all_days = pd.date_range(df_eff["Timestamp"].min(), df_eff["Timestamp"].max(), freq="D")
cal = pd.DataFrame({"Timestamp": all_days})
cal["Effort"] = cal["Timestamp"].isin(df_eff["Timestamp"].unique()).astype(int)

daily = (
    df_eff.groupby("Timestamp", as_index=False)
          .agg(Removed=("Removed","sum"),
               Volume=("Volume","sum"))
)

data = cal.merge(daily, on="Timestamp", how="left")
data["Removed"] = data["Removed"].fillna(0.0)  # distinguish with Effort


# ----------------------------
# 2) Covariates (time & weekday) — Explanation:
#    • t = days since start (smooth trend)
#    • w = weekday index (0..6) for cyclic weekly pattern
# ----------------------------
data["t"] = (data["Timestamp"] - data["Timestamp"].min()).dt.days.astype(float)
data["w"] = data["Timestamp"].dt.dayofweek.astype(int)
X_all = np.c_[data["t"].values, data["w"].values]


# ----------------------------
# 3) Occurrence model (π_t) — Explanation:
#    • LogisticGAM on Effort=1 days
#    • Target is 1{Removed>0}
#    • Predict probability of a positive day for ALL days
# ----------------------------
mask_occ = data["Effort"] == 1
y_occ = (data.loc[mask_occ, "Removed"] > 0).astype(int).values
X_occ = np.c_[data.loc[mask_occ, "t"].values, data.loc[mask_occ, "w"].values]

occ = LogisticGAM(
    s(0, n_splines=TIME_SPLINES, lam=TIME_LAM) +
    s(1, n_splines=WDAY_SPLINES, lam=WDAY_LAM, basis='cp')
).gridsearch(X_occ, y_occ)

pi_hat = occ.predict_mu(X_all)  # shape [n_days], in [0,1]


# ----------------------------
# 4) Intensity model (r_t) — Explanation:
#    • On Effort=1 & Removed>0 days, model r = Removed/Volume (strictly >0)
#    • GammaGAM with log link for positive rates
#    • Predict r_t for ALL days (no exposure needed)
# ----------------------------
pos = (data["Effort"] == 1) & (data["Removed"] > 0)
if pos.sum() == 0:
    inten = None
    rate_pos_hat = np.zeros(len(data))
else:
    y_pos   = data.loc[pos, "Removed"].astype(float).values
    vol_pos = data.loc[pos, "Volume"].astype(float).values
    if (vol_pos <= 0).any() or np.isnan(vol_pos).any():
        raise ValueError("Positive Effort days must have Volume > 0 and not NA.")
    r_pos = y_pos / vol_pos

    X_pos = np.c_[data.loc[pos, "t"].values, data.loc[pos, "w"].values]
    inten = GammaGAM(
        s(0, n_splines=TIME_SPLINES, lam=TIME_LAM) +
        s(1, n_splines=WDAY_SPLINES, lam=WDAY_LAM, basis='cp')
    ).gridsearch(X_pos, r_pos)

    rate_pos_hat = inten.predict_mu(X_all)


# ----------------------------
# 5) Combine parts → InfectionDensity mean and uncertainty — Explanation:
#    • μ_t = π_t * r_t
#    • Draw from coefficient posteriors (approx.) to get var & 95% CI
# ----------------------------
infect_mean = pi_hat * rate_pos_hat

def _psd(cov, eps=1e-10):
    cov = (cov + cov.T) / 2.0
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, eps, None)
    return (V * w) @ V.T

# design matrices (private API in pyGAM; works consistently in practice)
M_occ_all = occ._modelmat(X_all).toarray()
cov_occ   = _psd(occ.statistics_['cov'])
beta_occ  = rng.multivariate_normal(mean=occ.coef_, cov=cov_occ, size=200)
eta_occ   = M_occ_all @ beta_occ.T
pi_draws  = expit(eta_occ)

if inten is not None:
    M_int_all = inten._modelmat(X_all).toarray()
    cov_int   = _psd(inten.statistics_['cov'])
    beta_int  = rng.multivariate_normal(mean=inten.coef_, cov=cov_int, size=200)
    eta_int   = M_int_all @ beta_int.T
    rate_draws = np.exp(eta_int)  # log link
else:
    rate_draws = np.zeros_like(pi_draws)

infect_draws = pi_draws * rate_draws
infect_var   = infect_draws.var(axis=1, ddof=1)
infect_lo, infect_hi = np.percentile(infect_draws, [2.5, 97.5], axis=1)

# ----------------------------
# 6) Counts — Explanation:
#    • Use observed counts on Effort=1 days
#    • Else predict Count = μ_t * Volume (with filled Volume when missing)
# ----------------------------
vol_all = data["Volume"].astype(float).copy()
if vol_all.isna().any():
    fill_val = (np.nanmedian(vol_all) if DEFAULT_VOLUME_FILL == "median"
                else float(DEFAULT_VOLUME_FILL))
    vol_all = vol_all.fillna(fill_val)

count_model = infect_mean * vol_all.values
count_final = count_model.copy()
mask_eff1 = data["Effort"] == 1
count_final[mask_eff1] = data.loc[mask_eff1, "Removed"].values

# ----------------------------
# 7) Assemble labels table — Explanation:
#    • InfectionDensity mean/var/CI
#    • Count (observed or modeled)
#    • ObservedDensity on Effort=1 days
# ----------------------------
labels = pd.DataFrame({
    "Timestamp": data["Timestamp"],
    "InfectionDensity": infect_mean,
    "InfectionDensity_var": infect_var,
    "InfectionDensity_lo": infect_lo,
    "InfectionDensity_hi": infect_hi,
    "Count": np.round(count_final).astype(int)
})
labels["ObservedDensity"] = (
    (data["Removed"]/data["Volume"].replace(0, np.nan))
      .where(data["Effort"]==1)
)

# ----------------------------
# 8) LOD + hysteresis (stability rule) — Explanation:
#    • Compute a detection threshold τ from data
#    • Zero μ_t for runs ≥ K days below τ (but never zero observed positives)
#    • Keep variance with a floor (avoid overconfident zeros)
# ----------------------------
K = 2  # require ≥K consecutive days below τ

eff     = (data["Effort"] == 1).to_numpy()
removed = data["Removed"].to_numpy(dtype=float)
vol     = data["Volume"].to_numpy(dtype=float)
obs_pos = eff & (removed > 0)

valid_eff_vol = eff & np.isfinite(vol) & (vol > 0)
lod_vol = 1.0 / np.median(vol[valid_eff_vol]) if valid_eff_vol.any() else 0.0

pos_mask = eff & (removed > 0) & np.isfinite(vol) & (vol > 0)
min_pos = float((removed[pos_mask] / vol[pos_mask]).min()) if pos_mask.any() else 0.0
tau = max(lod_vol, 0.5 * min_pos)

dens = labels["InfectionDensity"].to_numpy(dtype=float)
cand = np.isfinite(dens) & (dens < tau)
cand_s  = pd.Series(cand, index=labels.index)
run_id  = (cand_s != cand_s.shift(fill_value=False)).cumsum()
run_len = cand_s.groupby(run_id, observed=True).transform("size")
to_zero = (cand_s & (run_len >= K)).to_numpy()
to_zero &= ~obs_pos  # protect observed positives

labels.loc[to_zero, "InfectionDensity"] = 0.0
labels.loc[to_zero, "Count"] = 0

# variance floor for zeroed days
eps = 1e-4
k   = 20
y_bin = (labels["InfectionDensity"] > 0).astype(int)
pi_roll = (y_bin.rolling(30, min_periods=5, center=True).mean().fillna(y_bin.mean()))
v_floor = (pi_roll * (1 - pi_roll)) / (k + 1)

v = pd.to_numeric(labels["InfectionDensity_var"], errors="coerce").fillna(0.0)
v = np.maximum(v, v_floor)
v = np.clip(v, eps, np.inf)
labels.loc[to_zero, "InfectionDensity_var"] = v.loc[to_zero].to_numpy()

# fix CIs on zeroed days: lo=0, hi from sd
sd0 = np.sqrt(labels.loc[to_zero, "InfectionDensity_var"].astype(float).to_numpy())
labels.loc[to_zero, "InfectionDensity_lo"] = 0.0
labels.loc[to_zero, "InfectionDensity_hi"] = 1.96 * sd0

# ----------------------------
# 9) Soft labels from occurrence model — Explanation:
#    • SoftLabel_Occurrence := P(Removed>0 | day)
#    • For provenance: observed positives=1, observed zeros=0, no-sampling=π̂_t
#    • Also compute Var via delta method on logit scale
# ----------------------------
ts_all = labels["Timestamp"].reset_index(drop=True)
t_all  = (ts_all - ts_all.min()).dt.days.astype(float).to_numpy()
w_all  = ts_all.dt.dayofweek.astype(int).to_numpy()
X_all2 = np.c_[t_all, w_all]

pi_hat_all = occ.predict_mu(X_all2)  # [0,1]

eff2      = (data["Effort"] == 1).to_numpy()
removed2  = data["Removed"].to_numpy(dtype=float)
obs_pos2  = eff2 & (removed2 > 0)
obs_zero2 = eff2 & ~obs_pos2
est2      = ~eff2

Xd = occ._modelmat(X_all2)
Xd = Xd.toarray() if hasattr(Xd, "toarray") else np.asarray(Xd)
beta = np.asarray(occ.coef_)
Cov  = np.asarray(occ.statistics_["cov"])
eta     = Xd @ beta
var_eta = np.einsum("ij,jk,ik->i", Xd, Cov, Xd)
p      = np.clip(expit(eta), 1e-6, 1-1e-6)
var_p  = (p * (1 - p))**2 * var_eta
var_p  = np.clip(var_p, 1e-4, np.inf)

soft = np.where(obs_pos2, 1.0, np.where(obs_zero2, 0.0, p))
soft = np.clip(soft, 1e-6, 1-1e-6)

soft_df = pd.DataFrame({
    "Timestamp": ts_all,
    "SoftLabel_Occurrence": soft,
    "SoftLabelOccurrence_var": var_p,
    "SoftSource": np.where(obs_pos2, "observed_pos",
                    np.where(obs_zero2, "observed_zero", "estimated"))
})

# ----------------------------
# 10) Merge features + labels — Explanation:
#     • Attach InfectionDensity stats + Soft labels to daily features
#     • Save final dataset for ML
# ----------------------------
lab = labels.set_index("Timestamp")[["InfectionDensity", "InfectionDensity_var"]]
soft_lab = soft_df.set_index("Timestamp")[["SoftLabel_Occurrence", "SoftLabelOccurrence_var", "SoftSource"]]

df_final = (df_feat.merge(lab, left_on="Timestamp", right_index=True, how="left")
                  .merge(soft_lab, left_on="Timestamp", right_index=True, how="left"))
df_final.to_csv(FINAL_FILE, index=False)

print(f"[OK] Wrote final dataset: {FINAL_FILE}")


# ----------------------------
# 11) Plots — Explanation:
#     • Probability (soft labels) timeseries with observed markers
#     • Infection density mean & CI, with observed zeros highlighted
# ----------------------------
# Global font and size settings
plt.rcParams.update({
    "font.family": "Arial",     
    "font.size": 12,            
    "axes.labelsize": 12,              
    "legend.fontsize": 12,      
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# ----- Soft labels plot -----
eff2      = (data["Effort"] == 1).to_numpy()
obs_pos2  = eff2 & (data["Removed"].to_numpy(dtype=float) > 0)
obs_zero2 = eff2 & ~obs_pos2
est2      = ~eff2  # no-sampling (estimated) days

# y-values for observed labels (1 for pos, 0 for zero), only on Effort=1 days
y_obs = np.where(obs_pos2, 1.0, 0.0)

fig, ax = plt.subplots(figsize=(14,6))

# main probability line
ax.plot(soft_df["Timestamp"], soft_df["SoftLabel_Occurrence"], '--', linewidth=2,
        label="P(positive) from occurrence GAM")

# distinct markers for clarity 
ax.scatter(soft_df.loc[obs_pos2, "Timestamp"],  np.ones(obs_pos2.sum()),
           s=22, marker='^', label="Observed positive", zorder=5, color="orange")
ax.scatter(soft_df.loc[obs_zero2,"Timestamp"], np.zeros(obs_zero2.sum()),
           s=22, facecolors="none", edgecolors="red", marker='o',
           label="Observed zero", zorder=7)

# light shading for estimated/no-sampling periods
if est2.any():
    est_s = pd.Series(est2, index=soft_df.index)
    run_id = (est_s != est_s.shift(fill_value=False)).cumsum()
    first = True
    for rid, grp in est_s.groupby(run_id):
        if grp.iloc[0]:  # this run is 'estimated'
            start = soft_df["Timestamp"].iloc[grp.index[0]]
            end   = soft_df["Timestamp"].iloc[grp.index[-1]]
            ax.axvspan(start, end, alpha=0.08,
                       label="No sampling (estimated)" if first else None, zorder=0)
            first = False

ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("Date"); ax.set_ylabel("Probability of positive")
ax.legend()
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig("soft_labels_timeseries.png", dpi=200)
plt.close()

# ----- Infection density plot: add shading + orange observed points -----
eff2      = (data["Effort"] == 1).to_numpy()
obs_dens  = labels["ObservedDensity"]            # observed density on Effort=1 days
has_obs   = eff2 & obs_dens.notna().to_numpy()
est2      = ~eff2                                # no-sampling (estimated) days

fig, ax = plt.subplots(figsize=(14,6))

# main model line + CI
ax.plot(labels["Timestamp"], labels["InfectionDensity"], '--', linewidth=2,
        label="Infection density (hurdle-GAM)")
ax.fill_between(labels["Timestamp"],
                labels["InfectionDensity_lo"], labels["InfectionDensity_hi"],
                alpha=0.20, label="95% CI")

# orange observed points (actual density from Removed/Volume)
ax.scatter(labels.loc[has_obs, "Timestamp"], obs_dens.loc[has_obs],
           s=26, color="orange", edgecolors="none",
           label="Observed infection density", zorder=6)

# mark observed zeros 
zero_obs = eff2 & labels["ObservedDensity"].fillna(0).eq(0).to_numpy()
ax.scatter(labels.loc[zero_obs, "Timestamp"], np.zeros(zero_obs.sum()),
           s=22, facecolors="none", edgecolors="red", marker="o",
           label="Observed zero", zorder=7)

# light shading for estimated no-sampling periods
if est2.any():
    est_s = pd.Series(est2, index=labels.index)
    run_id = (est_s != est_s.shift(fill_value=False)).cumsum()
    first = True
    for rid, grp in est_s.groupby(run_id):
        if grp.iloc[0]:  # this run is 'estimated'
            start = labels["Timestamp"].iloc[grp.index[0]]
            end   = labels["Timestamp"].iloc[grp.index[-1]]
            ax.axvspan(start, end, alpha=0.08,
                       label="No sampling (estimated)" if first else None, zorder=0)
            first = False

ax.legend(loc="upper left")   # move legend to top-left corner
ax.set_ylim(0, 12)            # limit y-axis from 0 to 12
ax.set_xlabel("Date"); ax.set_ylabel("Infection density (saprolengia fish per volume)")
ax.legend()
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig("infection_density_timeseries.png", dpi=200)
plt.close()
