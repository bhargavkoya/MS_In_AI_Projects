# Client Subscription Classifier (Machine Learning)

## Project Title
Bank Client Subscription Classification

## What This Project Is
This project develops a machine learning classifier to predict whether a bank client is likely to subscribe, implemented through notebook-based experimentation and report documentation.

## Problem Statement
Marketing resources are limited, so targeting likely subscribers is critical. The objective is to build a predictive model that improves campaign focus and conversion efficiency.

## Implemented Solution
- Built an end-to-end ML notebook for data preparation, modeling, and evaluation.
- Documented modeling approach and findings in a report.

## Tech Stack
- Python
- Jupyter Notebook
- Classical machine learning tools used in notebook

## Results and Outputs
- The notebook reports full train/test metrics per model; Random Forest test performance is Accuracy 0.8873, F1 0.3704, and ROC-AUC 0.89.
- Cross-validation results are included for model comparison (for example, Random Forest CV Accuracy 0.8985 and CV ROC-AUC 0.9016), plus top-feature ranking output led by duration, balance, and age.
- Hyperparameter search outputs are logged (for example, tuned Random Forest with max_depth 15, min_samples_leaf 2, min_samples_split 5, n_estimators 100), enabling reproducible model-selection rationale.

## Local Setup and Run Steps
Some setup details are not explicitly defined in this folder; inferred minimal steps are provided.

1. Create a Python environment.
2. Install libraries imported in the notebook.
3. Run the notebook sequentially to reproduce training and evaluation outputs.

## Project Structure
- MLAndPatternRecognition_Banks_Bhargava_Koya.ipynb: Main classification notebook.
- Report_Bhargava_Koya.docx: Project report.

## Limitations and Future Improvements
- The notebook shows a large train-vs-test gap for Random Forest (perfect training scores vs materially lower test recall), indicating overfitting risk.
- Recall remains comparatively low on the positive class across tested models, so future work should prioritize class-imbalance handling, threshold tuning, and cost-sensitive optimization.
- Next iteration should add calibration and business-oriented decision thresholds (not only overall accuracy) to reduce missed-subscriber errors in campaign targeting.

## Source Evidence Used
- MLAndPatternRecognition_Banks_Bhargava_Koya.ipynb
- Report_Bhargava_Koya.docx
