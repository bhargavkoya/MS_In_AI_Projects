# GAT and LLM Hybrid Movie Recommender System

## Project Title
Hybrid Movie Recommender using Graph Attention Networks and LLM Components

## What This Project Is
This project explores a hybrid recommendation approach that combines graph-based representation learning with language-model-driven reasoning for movie recommendations.

## Problem Statement
Traditional recommenders may miss complex user-item graph relationships or nuanced semantic preferences. The objective is to combine graph intelligence and language understanding to improve recommendation quality and explainability.

## Implemented Solution
- Implemented and documented the hybrid workflow in a notebook.
- Produced supporting analysis and QA/report artifacts.
- Submitted multiple report variants for detailed and simplified communication.

## Tech Stack
- Python
- Jupyter Notebook
- Graph-based recommendation methods
- LLM-assisted recommendation reasoning

## Results and Outputs
- The notebook reports ranking metrics for the hybrid model, including NDCG@10 = 0.0478, Precision@10 = 0.0097, and HitRate@10 = 0.0965 in the ablation summary.
- Ablation output compares three baselines (Popularity, Content-based SBERT, LightGCN+SBERT), with the hybrid setup showing the strongest overall Top-K ranking profile in notebook logs.
- The report further documents an extended evaluation setting where LightGCN + SBERT achieves NDCG@10 = 0.2847, and ties these gains to graph-plus-semantic feature fusion.

## Local Setup and Run Steps
Some setup details are not explicitly defined in this folder; inferred minimal steps are provided.

1. Create a Python environment.
2. Install dependencies used by the notebook.
3. Open and run the notebook sequentially to reproduce outputs.

## Project Structure
- CA2_—_GAT_+_LLM_Hybrid_Movie_Recommender_System_BhargavaKoya_20075511 (1).ipynb: Main implementation notebook.
- CA2_Report_BhargavaKoya_20075511.docx and CA2_Report_Simplified_BhargavaKoya_20075511.docx: Reports.
- CA2_CodeAnalysis_BhargavaKoya_20075511.docx: Code analysis narrative.
- CA2_DemoQA_BhargavaKoya_20075511.docx: Demo QA support material.

## Limitations and Future Improvements
- Report analysis identifies popularity bias as a persistent ethical/technical issue even in hybrid recommenders, with risk of over-exposing already popular items.
- Explainability methods used (for example, LIME/SHAP discussion in report material) are useful but local/approximate, so explanations are not globally faithful for all recommendation contexts.
- Future work documented in project materials includes calibrated re-ranking, adversarial debiasing, and stronger fairness-aware evaluation beyond standard Top-K utility metrics.

## Source Evidence Used
- CA2_—_GAT_+_LLM_Hybrid_Movie_Recommender_System_BhargavaKoya_20075511 (1).ipynb
- CA2_Report_BhargavaKoya_20075511.docx
- CA2_Report_Simplified_BhargavaKoya_20075511.docx
- CA2_CodeAnalysis_BhargavaKoya_20075511.docx
- CA2_DemoQA_BhargavaKoya_20075511.docx
