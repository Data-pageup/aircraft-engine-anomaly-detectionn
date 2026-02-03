# Aircraft Engine Anomaly Detection

This project identifies aircraft engines operating at
abnormally high Turbine Gas Temperatures (TGT) during
cruise phase using unsupervised learning techniques.

## Problem Statement
An issue during engine development can cause engines
to run hotter than expected. The objective is to
identify affected engines before inspection.

## Approach
- Exploratory data analysis of cruise-phase sensor data
- Feature scaling and normalization
- Isolation Forest for anomaly detection
- Engine-level aggregation of anomaly scores

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
