# Taxi Grid Navigation System Prototype (Reinforcement Learning)

## Project Title
Taxi Grid Navigation System Prototype with RL and Interactive Simulation

## What This Project Is
This project combines reinforcement learning assignment work with an interactive taxi simulation prototype that includes a backend service and a frontend workflow design.

## Problem Statement
Grid-based taxi navigation requires efficient route planning under dynamic conditions. The objective is to model RL-driven navigation behavior and provide a simulation interface for scenario exploration.

## Implemented Solution
- Developed RL assignment notebook and analysis notes.
- Implemented backend logic for simulation and RL behavior.
- Built a frontend simulation workflow design with modern web tooling.

## Tech Stack
- Python
- Jupyter Notebook
- Frontend stack in the simulation design subproject
- JSON data artifacts

## Results and Outputs
- Notebook training logs show both Q-Learning and Monte Carlo reaching near-saturated success on the configured task (for example, Q-Learning hits 100% success by around episode 1000 in the printed checkpoints).
- Hyperparameter sweeps are included in outputs (for example alpha and epsilon-decay variants), with comparative training traces and smoothed reward/success visualizations.
- The prototype extends notebook logic into a runnable system: FastAPI backend (`main.py`, `rl_logic.py`) plus React/Vite frontend simulation workflow, documented in `PROJECT_ANALYSIS.md`.

## Local Setup and Run Steps
1. Review RL_Assignment.ipynb for algorithmic workflow and outputs.
2. For backend:
   - Navigate to backend.
   - Install dependencies from backend/requirements.txt.
   - Run main.py.
3. For frontend simulation workflow:
   - Navigate to Taxi Simulation Workflow Design.
   - Install package dependencies and run the dev server using package scripts.

## Project Structure
- RL_Assignment.ipynb: Core RL notebook.
- PROJECT_ANALYSIS.md: Analytical summary.
- cells_extract.json: Grid/cell artifact.
- backend/: Python backend service files including main.py and rl_logic.py.
- Taxi Simulation Workflow Design/: Frontend simulation app with its own README and package files.

## Limitations and Future Improvements
- Current backend architecture uses a global in-memory state model (documented in `PROJECT_ANALYSIS.md`), which is practical for demos but limited for multi-user or distributed deployment.
- Evaluation is focused on one Taxi-style grid environment/reward design; broader policy robustness across alternative maps and reward schemes is still pending.
- Future direction should include persistent experiment tracking, concurrent-session support, and expanded algorithm benchmarks (for example policy-gradient or value-based deep RL baselines).

## Source Evidence Used
- RL_Assignment.ipynb
- PROJECT_ANALYSIS.md
- backend/main.py
- backend/rl_logic.py
- backend/requirements.txt
- Taxi Simulation Workflow Design/README.md
- Taxi Simulation Workflow Design/package.json
