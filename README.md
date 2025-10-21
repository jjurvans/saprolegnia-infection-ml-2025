# saprolegnia-infection-ml-2025
**Data and machine learning models for studying *Saprolegnia* infection dynamics in Enonkoski fish farm, Finland (2024–2025).**

## Repository structure
**data**
- `WaterAnalysisRawData.csv` contains 10-minute interval water quality data collected from Enonkoski fish farm surface and bottom layers between March 2024 and August 2025.Parameters include temperature, pH, pH-mV, dissolved oxygen, UV254 (filtered and total), nitrate (NO₃eq, NO₃_Neq), and turbidity. 
- `WaterAnalysisAggregatedData.csv` is daily mean of all the features from WaterAnalysisRawData.csv except bottom turbidity. Bottom turbidity has statistically significant circadian differences and was thus divided into morning, afternoon, evening and night mean values. The script used is aggregate-water-features-daily.py.
- `SaprolengiaInfections24_25.csv` fish farm data on timestamped data on fish removal due to extensive mold or death. Columns are fish species, explanation ("saprolengia" or "other" for other treatment), quantity of all fish in the pool, number of removed fish due to saprolengia, pool volume, and timestamp. 

**Scripts**
- `aggregate-water-features-daily.py` Computes daily means for all features (10-min raw to daily), except bottom turbidity which shows circadian variation. Bottom turbidity is aggregated into mean night, morning, afternoon, and evening. Timezone: Europe/Helsinki. Missing daily values from 
incomplete day bins were imputed using a multivariate iterative imputer. The algorithm used chained regression (Bayesian Ridge by default) 
with 20 iterations and random seed = 0. Versions used: pandas 2.3.1, numpy 2.2.5, and scikit-learn 1.7.1.
 
