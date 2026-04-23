# Online Stock Forecasting (Deep Learning)

## Project Title
Online Stock Forecasting

## What This Project Is
This project explores stock price forecasting using a deep learning workflow implemented in notebook format, supported by a written report.

## Problem Statement
Financial time-series forecasting is challenging due to non-stationarity and noisy patterns. The objective is to model historical market behavior and generate short-term predictive insights.

## Implemented Solution
- Built notebook-driven forecasting experiments for stock time-series data.
- Captured methodology, assumptions, and analysis in a companion report.

## Tech Stack
- Python
- Jupyter Notebook
- Time-series and deep learning tools used in the notebook

## Results and Outputs
- The notebook reports a full test-set comparison across three models (Vanilla LSTM, Online LSTM, IL-ETransformer) using RMSE, MAE, MSE, MAPE, and directional accuracy.
- Best error performance is achieved by Online LSTM (RMSE 8.6174, MAE 6.9194, MAPE 3.5344), while best directional accuracy is from Vanilla LSTM (52.78%).
- The experiment uses an AAPL pipeline with 11 engineered features and a processed dataset shape of 2,466 rows x 15 columns, with final metrics printed in the CA02 results summary block.

## Local Setup and Run Steps
Some setup details are not explicitly defined in this folder; inferred minimal steps are provided.

1. Create a Python environment.
2. Install libraries referenced in the notebook.
3. Open the notebook and run cells sequentially.

## Project Structure
- CA02_Project1_Online_Stock_Forecasting (1).ipynb: Forecasting notebook.
- CA02_Report.docx: Project report.

## Limitations and Future Improvements
- The report's limitations section states that findings are from a single-stock setup (AAPL), so generalization to other tickers/market regimes is not guaranteed.
- The report recommends a layered update strategy (periodic batch retraining plus daily mini-batch updates) to balance stability and responsiveness in online settings.
- Future iterations should extend multi-asset evaluation and regime-robust validation, since directional gains and absolute-error gains do not align uniformly across all models.

## Source Evidence Used
- CA02_Project1_Online_Stock_Forecasting (1).ipynb
- CA02_Report.docx
