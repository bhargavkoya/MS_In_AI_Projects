# Centrality Measures Calculation Tasks (Graph and AI)

## Project Title
Centrality Measures and Graph Analysis Tasks

## What This Project Is
This project combines graph-theory assignment work with practical centrality computation artifacts, including notebook analysis, CSV exports, and Cypher-style query notes.

## Problem Statement
Understanding influential nodes and structural behavior in a graph requires multiple centrality perspectives. The objective is to compute and compare degree, closeness, betweenness, PageRank, and eigenvector-style indicators.

## Implemented Solution
- Completed graph assignment analysis in a notebook.
- Produced detailed per-metric outputs in CSV files.
- Stored procedural calculation/query notes in text files.

## Tech Stack
- Jupyter Notebook
- Graph analytics workflows
- CSV-based metric exports

## Results and Outputs
- The notebook produces a concrete shortest-path result: node 1 to node 10 via [1, 2, 4, 6, 5, 8, 10] with total path length 24.
- Centrality outputs are exported as separate CSV files for 16 entities per metric (betweenness, closeness, degree, eigenvector, and PageRank), enabling direct cross-metric ranking.
- Graph topology evidence is preserved in both script-style construction files and the NodesAndTheirRelations visualization image.

## Local Setup and Run Steps
Some setup details are not explicitly defined in this folder; inferred minimal steps are provided.

1. Open the main notebook to review or rerun analysis.
2. Inspect CSV files inside the centrality task folder for metric details.
3. Use text artifacts as step references for graph construction and metric calculations.

## Project Structure
- GraphAndAICA01_Dijkastra_Assignment.ipynb: Core assignment notebook.
- CA01_GraphAndAI_HandCalculationAssignment.pdf: Assignment context.
- CentralityMeasuresCalculationTask/: Metric CSVs, graph setup scripts, and visualization.

## Limitations and Future Improvements
- The workflow is split across multiple TXT scripts and CSV exports, so reproducibility depends on manual execution order rather than a single orchestrated pipeline.
- Analysis is snapshot-based on one fixed graph instance; temporal dynamics, weighted-edge uncertainty, and sensitivity checks are not part of the current outputs.
- Naming and artifact consistency issues (for example, mixed spellings and separate metric files) should be standardized into one comparative report/table for easier interpretation.

## Source Evidence Used
- GraphAndAICA01_Dijkastra_Assignment.ipynb
- CA01_GraphAndAI_HandCalculationAssignment.pdf
- CentralityMeasuresCalculationTask/betweenesscentrality_details.csv
- CentralityMeasuresCalculationTask/closenesscentrality_details.csv
- CentralityMeasuresCalculationTask/degreecentrality_details.csv
- CentralityMeasuresCalculationTask/eigenvectorcentrality_details.csv
- CentralityMeasuresCalculationTask/pagerankcentrality_details.csv
- CentralityMeasuresCalculationTask/NodesAndTheirRelations.png
