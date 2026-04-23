# LinkedIn Jobs Scraper and Data Analysis

## Project Title
LinkedIn Jobs Web Scraper and Analysis

## What This Project Is
This project focuses on collecting job-market signals from LinkedIn listings and analyzing role trends in notebook format, with report and presentation support.

## Problem Statement
Job-seekers and analysts need structured insights from large volumes of job posts. The objective is to scrape relevant listings and convert raw postings into actionable trends.

## Implemented Solution
- Built a notebook workflow for scraping and exploratory analysis.
- Captured methodology and findings in report and slide formats.
- Added sample problem statements to frame analytical tasks.

## Tech Stack
- Python
- Jupyter Notebook
- Web scraping and data analysis libraries used in notebook

## Results and Outputs
- The notebook run logs show a completed scrape and persistence flow: 19 jobs scraped, feature extraction completed for 19 jobs, and 19 rows inserted into the jobs table.
- Outputs include structured job records and downstream analysis-ready fields, documented in the notebook and report narrative.
- The presentation/report pair provides implementation-level evidence for scraping workflow, parsing strategy, and practical operational constraints.

## Local Setup and Run Steps
Some setup details are not explicitly defined in this folder; inferred minimal steps are provided.

1. Create a Python environment.
2. Install libraries imported by the notebook.
3. Open and run the notebook in sequence.

## Project Structure
- LinkedinJobsWebscrapper_CA02_BhargavaKoya_20075511.ipynb: Main scraper and analysis notebook.
- LinkedinJobs_WebScraper_CA02_Bhargava_Koya.docx: Report.
- LinkedIn Jobs Web Scraper.pptx: Presentation.
- Sample problems.txt: Problem prompts and scope notes.

## Limitations and Future Improvements
- Report and slide material explicitly identify LinkedIn rate limiting and anti-bot thresholds as a current failure mode for longer scraping runs.
- HTML structure changes are a recurring maintenance risk; the project notes fallback parsing logic as necessary hardening work.
- Future improvement direction includes production-grade throttling/backoff, stronger session persistence/caching, and scheduled ingestion for trend monitoring.

## Source Evidence Used
- LinkedinJobsWebscrapper_CA02_BhargavaKoya_20075511.ipynb
- LinkedinJobs_WebScraper_CA02_Bhargava_Koya.docx
- LinkedIn Jobs Web Scraper.pptx
- Sample problems.txt
