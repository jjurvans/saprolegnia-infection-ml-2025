# saprolegnia-infection-ml-2025
Data and machine learning models for studying Saprolegnia infection dynamics in Finland (2024–2025).

Data:
1. WaterAnalysisRawData.csv contains 10-minute interval water quality data collected from Enonkoski fish farm surface and bottom layers between March 2024 and August 2025.Parameters include temperature, pH, pH-mV, dissolved oxygen, UV254 (filtered and total), nitrate (NO₃eq, NO₃_Neq), and turbidity. 
2. WaterAnalysisAggregatedData.csv is daily mean of all the features from WaterAnalysisRawData.csv except bottom turbidity. Bottom turbidity has statistically significant circadian differences and was thus divided into morning, afternoon, evening and night mean values. The script used is aggregate-water-features-daily.py.

Scripts:
1. aggregate-water-features-daily.py aggregates the 10-minute interwal water quality data from WaterAnalysisRawData.csv into daily WaterAnalysisAggregatedData.csv.
2. 
