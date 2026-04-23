# Energy Consumption Predictor (Machine Learning)

## Project Title
Household Energy Consumption Prediction

## What This Project Is
This project predicts household energy usage using machine learning with a script-driven implementation, dataset artifact, and report deliverables.

## Problem Statement
Energy demand forecasting supports better planning and efficiency decisions. The objective is to model household power consumption patterns and estimate future usage.

## Implemented Solution
- Implemented the core ML pipeline in a Python script.
- Used a household power consumption dataset for model training and evaluation.
- Documented methodology and outcomes in report formats.

## Tech Stack
- Python
- CSV data processing
- Machine learning libraries used in script

## Results and Outputs
- The implementation script builds a full time-series feature pipeline on household power data, including lag features (1, 6, 24), rolling statistics (6, 24), cyclical time encoding, PCA, and K-Means-assisted analysis.
- The report documents evaluation across 19 regression models and identifies Gradient Boosting as best performer with R2 = 0.9992 and RMSE = 0.0348 kW on the project evaluation setup.
- Model selection and optimization are explicitly tracked through n_estimators sweeps and tuned tree-based variants (Random Forest, Gradient Boosting, XGBoost) before final comparison.

## Local Setup and Run Steps
Some setup details are not explicitly defined in this folder; inferred minimal steps are provided.

1. Create a Python environment.
2. Install dependencies imported in the script.
3. Ensure the dataset file remains in this folder.
4. Run the script from this directory.

Example command:
python mlcode_ca02_energyconsumptionpredictor_bhargavakoya_20075511.py

## Project Structure
- mlcode_ca02_energyconsumptionpredictor_bhargavakoya_20075511.py: Main implementation script.
- household_power_consumption.csv: Input dataset.
- EnergyConsumptionPredictor_MLAndPatternRecognition_CA02_Report_BhargavaKoya_20075511.docx: Report.
- EnergyConsumptionPredictor_MLAndPatternRecognition_CA02_Report_BhargavaKoya_20075511.pdf: Report (PDF).

## Limitations and Future Improvements
- The report states a compute-bound training constraint: experiments were run on a 50K-row subset instead of the full dataset, which may limit external validity.
- Multiple models are very close in R2, so future work should include stronger stress tests (out-of-period validation and robustness checks) rather than relying only on headline fit.
- Recommended next steps in the report include broader hyperparameter optimization on full-scale data to verify whether current gains persist beyond sampled training.

## Source Evidence Used
- mlcode_ca02_energyconsumptionpredictor_bhargavakoya_20075511.py
- household_power_consumption.csv
- EnergyConsumptionPredictor_MLAndPatternRecognition_CA02_Report_BhargavaKoya_20075511.docx
- EnergyConsumptionPredictor_MLAndPatternRecognition_CA02_Report_BhargavaKoya_20075511.pdf
