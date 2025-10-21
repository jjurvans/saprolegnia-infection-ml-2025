#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 09:44:36 2025

@author: jaanajurvansuu

Computes daily means for all features (10-min raw to daily), 
except bottom turbidity which shows circadian variation. 
Bottom turbidity is aggregated into mean night, morning, afternoon, 
and evening. Timezone: Europe/Helsinki. Missing daily values from 
incomplete day bins were imputed using a multivariate iterative imputer. 
The algorithm used chained regression (Bayesian Ridge by default) 
with 20 iterations and random seed = 0. Versions used: pandas 2.3.1, numpy 2.2.5, and
scikit-learn 1.7.1.

"""
import pandas as pd
import numpy as np
from sklearn.impute import IterativeImputer

# --- I/O
RAW = "WaterAnalysisRawData.csv"
OUT = "WaterAnalysisAggregatedData.csv"

# --- Load & time handling
df = pd.read_csv(RAW, parse_dates=["Timestamp"])
df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")

col_turb = "turbidity_bottom_FTUeq" 

# --- Time-of-day bins
bins   = [0, 6, 12, 18, 24]
labels = ["night", "morning", "afternoon", "evening"]
df["period"] = pd.cut(df["Timestamp"].dt.hour, bins=bins, labels=labels, right=False, include_lowest=True)

# --- Daily means for all numeric features EXCEPT bottom turbidity
num_cols = df.select_dtypes(include="number").columns.tolist()
cols_mean = [c for c in num_cols if c != col_turb]

daily = (df
         .groupby(pd.Grouper(key="Timestamp", freq="D"))[cols_mean]
         .mean())

# --- Bottom turbidity as day-part means, pivoted to columns
turb_daily = (
    df.groupby([pd.Grouper(key="Timestamp", freq="D"), "period"], observed=True)[col_turb]
      .mean()
      .unstack("period")
)

# --- Ensure stable column order and names
turb_daily = turb_daily.reindex(columns=labels)
turb_daily.columns = [f"{col_turb}_{p}" for p in turb_daily.columns]

# --- Join and save
result = (daily
          .join(turb_daily)
          .reset_index()            
          .sort_values("Timestamp"))

# --- Imputation: fill missing numeric values
num_cols = result.select_dtypes(include=[np.number]).columns
miss_mask = result[num_cols].isna() 

imp = IterativeImputer(random_state=0, max_iter=20, skip_complete=True)
result[num_cols] = imp.fit_transform(result[num_cols])

# --- quality check
print("Missing before:", miss_mask.sum().sum())
print("Missing after:", result[num_cols].isna().sum().sum())

result.to_csv(OUT, index=False)
