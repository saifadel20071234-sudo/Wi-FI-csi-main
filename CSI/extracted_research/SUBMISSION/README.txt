WIFI CSI OCCUPANCY ESTIMATION - SUBMISSION

This folder contains the ESP32 CSI data analysis and model evaluation work.

Data:
Cleaned ESP32 recordings used in the experiments.

Features:
Causal and advanced temporal feature datasets.

Results:
Model comparisons, error analysis, dataset summary and signal plots.

Scripts:
Python scripts used for preprocessing, feature engineering,
model evaluation and analysis.

Important:
The third recording contains occupancy labels only, without
corresponding CSI measurements. It was not used for CSI training.

The best within-recording MAE was 0.723.
Cross-recording evaluation did not outperform the baseline.
The current model is not yet reliable for general deployment.

See FINAL_RESULTS_REPORT.txt for full findings.
