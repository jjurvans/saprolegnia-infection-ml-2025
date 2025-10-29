# saprolegnia-infection-ml-2025

**Data and machine learning models for studying *Saprolegnia* infection dynamics in the Enonkoski fish farm, Finland (2024–2025).**

---

## Repository structure

### **Data**

- **`WaterAnalysisRawData.csv`**  
  Contains 10-minute interval water-quality data collected from the surface and bottom layers of the Enonkoski fish farm between March 2024 and August 2025. Parameters include temperature, pH, pH-mV, dissolved oxygen, UV254 (filtered and total), nitrate (NO₃eq, NO₃_Neq), and turbidity.

- **`WaterAnalysisAggregatedData.csv`**  
  Provides daily means of all features from `WaterAnalysisRawData.csv`, except for bottom turbidity. Bottom turbidity shows statistically significant circadian variation and was therefore divided into morning, afternoon, evening, and night means. Missing daily values from incomplete day bins were imputed using a multivariate iterative imputer. *Script used:* `aggregate-water-features-daily.py`

- **`SaprolengiaInfections24_25.csv`**  
  Contains timestamped data on fish removal due to extensive *Saprolegnia* mould infection or mortality caused by *Saprolegnia*. Columns include fish species, explanation (“saprolengia” or “other” for working days when no *Saprolegnia* was visible), total fish per pool, number of fish removed due to *Saprolegnia*, pool volume, and timestamp.

- **`SaprolegniaInfection_LabeledData_2025.csv`**  
  Combines feature data from `WaterAnalysisAggregatedData.csv` with daily infection density (visibly infected fish per pool volume) estimated under a walk-forward, causal scheme using a two-part hurdle Generalised Additive Model (GAM) derived from `SaprolengiaInfections24_25.csv`. *Script used:* `walkforward_hurdleGAM_label_estimation.py`

- **`topK_global_majority_K50.csv`**  
  Lists the 50 most predictive lagged, rolling mean, and standard deviation features selected using `evaluate_topk_temporal_features.py`.

- **`TestData.csv`**  
  Contains water-quality data from September–October 2025, which were not used in the development of the Random Forest model.

- **`NewInfectionDensity.csv`**  
  Provides infection density data from September–October 2025, not used during model development.

---

### **Scripts**

- **`aggregate-water-features-daily.py`**  
  Computes daily means for all features (10-minute raw to daily), except bottom turbidity, which shows circadian variation. Bottom turbidity is aggregated into mean night, morning, afternoon, and evening values. Missing daily values from incomplete bins are imputed using a multivariate iterative imputer.

- **`walkforward_hurdleGAM_label_estimation.py`**  
  Estimates daily *Saprolegnia* infection density and occurrence probability using a causal, walk-forward hurdle Generalised Additive Model (GAM). Reconstructs missing infection records by combining logistic and Gamma components, applies uncertainty propagation and detection-limit filtering, and produces `SaprolegniaInfection_LabeledData_2025.csv` together with figures `infection_density_timeseries` and `soft_label_timeseries`.

- **`feature_importance_voting_spacing.py`**  
  Selects predictive time-lagged features from `SaprolegniaInfection_LabeledData_2025.csv` using Random Forest, LightGBM, and XGBoost. Builds 1–28-day lag and rolling features, applies a 14-day within-family spacing rule to avoid redundancy, and identifies the most stable predictors across time-series folds using model-averaged importances and stability metrics.

- **`evaluate_topk_temporal_features.py`**  
  Evaluates Top-K temporal feature sets from `feature_importance_voting_spacing.py` using time-series cross-validation with ensemble models (RF, LGBM, XGB).Produces baseline metrics and comparative Brier/Log-Loss performance plots under weighted and unweighted evaluation schemes.

- **`RF_weighted_optimisation_with_top50.py`**  
  Performs Random Forest model optimisation on `SaprolegniaInfection_LabeledData_2025.csv` with `topK_global_majority_K50.csv`, using weighted hard-label classification and inverse-variance weighting. Conducts nested time-series cross-validation with a broad deterministic hyperparameter grid, evaluates against constant and persistence (t−1) baselines, and reports probabilistic performance metrics (Brier, Log-Loss, ROC-AUC, PR-AUC) with diagnostic plots.

- **`saprolegnia-rf-pipeline.py`**  
  Generates lagged and rolling time-series features from `SaprolegniaInfection_LabeledData_2025.csv`, applies inverse-variance weighting, retains the selected predictors from `topK_global_majority_K50.csv`, and fits a calibrated Random Forest model. Produces SHAP-based interpretability plots, block permutation importance analyses, and calibrated infection-probability forecasts using `TestData.csv` and `NewInfectionDensity.csv`.

---

### **Note**

The original script was generated with assistance from **ChatGPT (OpenAI)** and has been subsequently **reviewed, modified, and tested by the repository author**.
