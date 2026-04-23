# Blog Gender Classification and Transformer Paper Analysis (NLP)

## Project Title
Blog Gender Classification and Transformer Paper Critical Analysis

## What This Project Is
This project combines two NLP-focused assignment components: a critical review of the Transformer paper and a practical blog gender classification workflow.

## Problem Statement
The project addresses both conceptual and practical NLP objectives: understanding modern sequence modeling foundations and applying text classification techniques to author-gender prediction tasks.

## Implemented Solution
- Produced a critical review report for the Transformer paper.
- Implemented a blog gender classification pipeline in notebook format.
- Generated supporting evaluation and feature-importance visual outputs.

## Tech Stack
- Python
- Jupyter Notebook
- NLP and ML libraries used in notebook workflow

## Results and Outputs
- Task 2 executes 9 experiment combinations (3 representations x 3 algorithms), and the results are persisted in `task2_results.csv` with CV and test metrics.
- Notebook outputs include model-wise metrics such as TF-IDF + SVM (CV Acc 0.713 +/- 0.028), Word2Vec + SVM (Test Acc 0.708), and TF-IDF + Naive Bayes (Test AUC 0.781), showing comparable performance across representation choices.
- Visual diagnostics are captured in `evaluation.png` and `feature_importance.png`, while the Transformer critical-review report records conceptual findings separately from the classification pipeline.

## Local Setup and Run Steps
Some setup details are not explicitly defined in this folder; inferred minimal steps are provided.

1. Create a Python environment.
2. Install notebook dependencies referenced in the Task 2 notebook.
3. Open and run the notebook in Task 2 sequentially.

## Project Structure
- CA1_Task1_AttentionIsAllYouNeed_CriticalReview_Report_BhargavaKoya_20075511.docx: Transformer paper review.
- Task 2/BlogGenderClassification_NLP_BhargavaKoya_20075511.ipynb: Classification notebook.
- Task 2/task2_results.csv: Result data.
- Task 2/evaluation.png and Task 2/feature_importance.png: Visual result artifacts.

## Limitations and Future Improvements
- The Transformer review explicitly highlights the quadratic self-attention cost as a major limitation for long-sequence scaling.
- Classification scores cluster around ~0.66 to ~0.71 test accuracy across setups, indicating room for stronger feature engineering and richer model families beyond the current baseline trio.
- Future work should add deeper error analysis (topic/style-specific failure cases), larger contextual embeddings, and stronger robustness checks across domain-shifted blog data.

## Source Evidence Used
- CA1_Task1_AttentionIsAllYouNeed_CriticalReview_Report_BhargavaKoya_20075511.docx
- Task 2/BlogGenderClassification_NLP_BhargavaKoya_20075511.ipynb
- Task 2/task2_results.csv
- Task 2/evaluation.png
- Task 2/feature_importance.png
