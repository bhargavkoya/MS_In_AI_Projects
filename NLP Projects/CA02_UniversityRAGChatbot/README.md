# University RAG Chatbot (NLP)

## Project Title
University Document Question Answering with Retrieval-Augmented Generation

## What This Project Is
This project builds a Retrieval-Augmented Generation chatbot for university-related documents, with ingestion, retrieval pipeline logic, and evaluation components.

## Problem Statement
Users need reliable answers grounded in institutional documents, but raw language models can hallucinate without retrieval grounding. The objective is to combine document retrieval with generation for context-aware, source-grounded responses.

## Implemented Solution
- Implemented application entrypoint, ingestion pipeline, retrieval/generation pipeline, and evaluation script.
- Prepared a document corpus in the data folder.
- Added evaluation assets to measure answer quality.
- Documented system design and outcomes in report files.

## Tech Stack
- Python
- RAG architecture
- JSON-based evaluation assets
- Requirements-based dependency management

## Results and Outputs
- Quantitative evaluation is recorded in `evaluation_results.json` over 51 questions, with average keyword coverage 0.6029, average response time 1.3466s, and source citation rate 1.0.
- Per-question outputs include response text, keyword-score, latency, and source-count fields, allowing traceable analysis of answer quality and grounding behavior.
- The report documents the implemented retrieval/generation stack (chunk_size 800, overlap 80, FAISS index, Claude 3 Haiku generation) and aligns these design choices with evaluation outcomes.

## Local Setup and Run Steps
1. Navigate to university_rag_chatbot.
2. Create and activate a Python environment.
3. Install dependencies from requirements.txt.
4. Run ingest.py to build retrieval-ready artifacts if needed.
5. Start the app using app.py.
6. Use evaluation.py for quality assessment.

Example commands:
cd university_rag_chatbot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python ingest.py
python app.py

## Project Structure
- university_rag_chatbot/app.py: Application entrypoint.
- university_rag_chatbot/rag_pipeline.py: Retrieval and generation pipeline.
- university_rag_chatbot/ingest.py: Data ingestion workflow.
- university_rag_chatbot/evaluation.py: Evaluation workflow.
- university_rag_chatbot/evaluation_results.json and test_set.json: Evaluation assets.
- university_rag_chatbot/data/: Source documents.
- CA02_Koya_Bhargava_20075511_Individual_Report.pdf and NLP_CA2_RAG_Report.docx: Reports.

## Limitations and Future Improvements
- The report notes interpretability limits at generation stage (LLM internals remain a black box even with retrieval grounding), so explainability is only partial.
- Current FAISS setup is in-memory and can become RAM-bound for larger corpora; the report recommends scaling strategies such as hybrid search and alternative vector-store options.
- Report recommendations include expanding knowledge-base coverage and adding RAGAS-style ground-truth evaluation to improve beyond the current baseline (60.3% keyword coverage).

## Source Evidence Used
- university_rag_chatbot/app.py
- university_rag_chatbot/rag_pipeline.py
- university_rag_chatbot/ingest.py
- university_rag_chatbot/evaluation.py
- university_rag_chatbot/requirements.txt
- university_rag_chatbot/evaluation_results.json
- CA02_Koya_Bhargava_20075511_Individual_Report.pdf
- university_rag_chatbot/NLP_CA2_RAG_Report.docx
