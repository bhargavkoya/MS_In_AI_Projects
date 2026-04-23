# LinkedIn Jobs Recommender System

## Project Title
LinkedIn Jobs Recommendation System

## What This Project Is
This project presents a recommender systems assignment focused on job recommendation scenarios using LinkedIn-related context, implemented in notebook format with supporting presentation and report artifacts.

## Problem Statement
Users need relevant job recommendations aligned with their profiles and preferences. The objective is to model recommendation logic that prioritizes relevance and usability.

## Implemented Solution
- Developed recommendation workflow in a Jupyter notebook.
- Prepared presentation and supporting script/report materials for communication.

## Tech Stack
- Python
- Jupyter Notebook
- Recommender systems techniques used in notebook

## Results and Outputs
- The notebook includes Top-10 evaluation over 200 random queries with reported proxy precision/title-match rate 0.585 +/- 0.324, intra-list diversity 0.550 +/- 0.270, and catalogue coverage 100% (50,000/50,000 items queryable).
- Ranking quality analysis sections in the notebook explicitly discuss NDCG/MAP framing and system-level trade-offs between relevance and diversity.
- Presentation/report artifacts document dataset scale and implementation framing used for the CA1 recommender deliverable.

## Local Setup and Run Steps
Some setup details are not explicitly defined in this folder; inferred minimal steps are provided.

1. Create a Python environment.
2. Install notebook dependencies.
3. Open and run the notebook sequentially.

## Project Structure
- LinkedIn_Jobs_Recommender_System_CA1.ipynb: Main recommender notebook.
- LinkedIn_Jobs_Recommender_System.pptx: Presentation.
- LinkedIn_Recommender_Presentation_Script.docx: Presentation script.
- Recommender Systems Group CA One.pdf: Assignment/report document.

## Limitations and Future Improvements
- Slide deck limitations explicitly call out cold-start behavior for new users/new jobs and reduced recommendation quality when profile/content signals are sparse.
- The current approach is primarily content-driven; future work in slides recommends adding collaborative filtering from user interaction logs (click/apply signals).
- Additional future direction in slides includes richer semantic encoders (for example BERT-based text representations) and stronger online feedback loops.

## Source Evidence Used
- LinkedIn_Jobs_Recommender_System_CA1.ipynb
- LinkedIn_Jobs_Recommender_System.pptx
- LinkedIn_Recommender_Presentation_Script.docx
- Recommender Systems Group CA One.pdf
