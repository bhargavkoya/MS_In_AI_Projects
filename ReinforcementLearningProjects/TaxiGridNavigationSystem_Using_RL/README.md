# Taxi Grid Navigation System Using RL

## Project Title
Taxi Grid Navigation using Reinforcement Learning

## What This Project Is
This project implements a reinforcement learning assignment for taxi grid navigation, centered around notebook experimentation with supporting report and data artifacts.

## Problem Statement
The challenge is to learn an effective navigation policy for taxi movement in a constrained grid environment. The objective is to train and evaluate RL behavior for task completion in simulated conditions.

## Implemented Solution
- Implemented RL workflow in notebook format.
- Included supporting dataset/text artifact and report.

## Tech Stack
- Python
- Jupyter Notebook
- Reinforcement learning methods implemented in notebook

## Results and Outputs
- Notebook logs report 5,000-episode training for both Q-Learning and Monte Carlo, with checkpointed success-rate traces (for example, both approaches reach ~100% around later checkpoints under tuned settings).
- Output cells include comparative runs under parameter changes (for example epsilon decay and alpha variants), making convergence behavior directly inspectable.
- The report complements notebook outputs with assignment-level interpretation of policy-learning behavior in the taxi-grid environment.

## Local Setup and Run Steps
Some setup details are not explicitly defined in this folder; inferred minimal steps are provided.

1. Create a Python environment.
2. Install libraries imported by the notebook.
3. Open RL_Assignment.ipynb and execute cells in order.

## Project Structure
- RL_Assignment.ipynb: Main RL notebook.
- data.txt: Supporting input/reference artifact.
- report.pdf: Project report.

## Limitations and Future Improvements
- Current results are tied to the single Taxi-grid setup and fixed reward structure; transferability to larger or stochastic environments is not validated in this folder.
- The notebook demonstrates strong convergence but limited baseline diversity (tabular Q-Learning vs Monte Carlo only), so broader algorithmic comparison remains open.
- Future work should include policy generalization tests, richer environment variants, and reproducibility controls (seed management + pinned dependency/runtime metadata).

## Source Evidence Used
- RL_Assignment.ipynb
- data.txt
- report.pdf
