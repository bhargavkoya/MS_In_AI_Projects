# Fraud Detection and Profitability Analysis (Graph and AI)

## Project Title
Graph-Based Fraud Risk and Profitability Analysis

## What This Project Is
This project applies graph analytics to customer and transaction data to evaluate fraud risk patterns and profitability signals.

## Problem Statement
Fraud often emerges through network behavior rather than isolated records. The objective is to detect suspicious structures and influential actors while also estimating profitability-related indicators.

## Implemented Solution
- Prepared customer and transaction graph data.
- Implemented Cypher query scripts for graph projection, centrality, pattern detection, and feature generation.
- Built Python analysis code to process graph-derived features.
- Documented outcomes in report and presentation artifacts.

## Tech Stack
- Python
- CSV data processing
- Cypher query workflows for graph analytics

## Results and Outputs
- Feature outputs include customer embeddings and engineered fraud-risk attributes.
- Query scripts capture analytical logic for reciprocity, communities, circular patterns, centrality, and profitability scoring.
- Report and presentation files summarize findings.

## Local Setup and Run Steps
Some setup details are not explicitly defined in this folder; inferred minimal steps are provided.

1. Load data from the data folder into the graph environment used by the Cypher scripts.
2. Execute Cypher scripts in a logical sequence: load data, project graph, compute algorithms, and detect patterns.
3. Run the Python analysis script to process exported graph features.

## Project Structure
- Fraud risk profitability code/cypher/: Graph query scripts for analytics.
- Fraud risk profitability code/data/: Input CSV files.
- Fraud risk profitability code/python/fraud_risk_analysis.py: Python analysis workflow.
- FraudDetection_GraphAndAI_CA02_Report_BhargavaKoya_20075511.docx: Report.
- Fraud Risk & Profitability Analysis.pptx: Presentation.

## Limitations and Future Improvements
- Environment provisioning for graph database execution is not fully specified here.
- A reproducible end-to-end runner script would improve usability.
- Validation with larger or external datasets would strengthen robustness.

## Source Evidence Used
- Fraud risk profitability code/python/fraud_risk_analysis.py
- Fraud risk profitability code/python/customer_features_with_embeddings.csv
- Fraud risk profitability code/data/customers.csv
- Fraud risk profitability code/data/transactions.csv
- Fraud risk profitability code/cypher/Projecting new in memory graph for algos.txt
- FraudDetection_GraphAndAI_CA02_Report_BhargavaKoya_20075511.docx
- Fraud Risk & Profitability Analysis.pptx
