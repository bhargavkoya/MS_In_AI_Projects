# Crop Disease Detection (Deep Learning)

## Project Title
Crop Disease Detection using Deep Learning

## What This Project Is
This project builds a computer-vision workflow to identify crop diseases from plant images using deep learning. It is structured as a coursework project with notebook-based experimentation and a report.

## Problem Statement
Manual crop disease identification is slow and can be error-prone at scale. The goal is to automate disease recognition from image inputs to support faster and more consistent diagnosis.

## Implemented Solution
- Developed notebook-based deep learning experiments for crop disease classification.
- Used an interactive demo notebook for inference workflows.
- Documented approach and findings in a project report.

## Tech Stack
- Python
- Jupyter Notebook
- Deep Learning libraries used inside notebooks

## Results and Outputs
- The training pipeline was executed on PlantVillage apple classes with explicit split counts in notebook output: 7,771 training images, 1,747 validation images, and 196 held-out test images.
- The benchmark section compares four architectures (Custom CNN, EfficientNetB0, ResNet50, VGG16) on the same 196-image test set, and the notebook includes comparison tables/plots for test accuracy and loss.
- End-to-end data preparation and inference workflow is reproducible in the demo notebook, including dataset download/extraction and class-wise split logging.

## Local Setup and Run Steps
Some setup details are not explicitly defined in this folder; inferred minimal steps are provided.

1. Create a Python environment.
2. Install notebook dependencies used in the project notebooks.
3. Launch Jupyter and open the notebooks in this folder.
4. Run cells in order to reproduce preprocessing, training, and inference outputs.

## Project Structure
- CropDiseaseDetection_DL_BhargavaKoya_20075511_Code.ipynb: Main implementation notebook.
- crop_disease_demo.ipynb: Demo and inference notebook.
- CropDiseaseDetection_DL_BhargavKoya_20075511_Report.docx: Project report.

## Limitations and Future Improvements
- The report explicitly notes a domain-gap risk: PlantVillage images are captured in controlled conditions, so real-field performance can degrade under variable lighting/backgrounds.
- Disease-class overlap is a stated challenge (for example, Apple Scab vs Black Rot visual similarity), so further error-analysis and class-specific augmentation are needed.
- The report also highlights compute constraints (VGG16 scale and training cost without strong GPU support); future work should focus on lighter deployment models and production-oriented inference packaging.

## Source Evidence Used
- CropDiseaseDetection_DL_BhargavaKoya_20075511_Code.ipynb
- crop_disease_demo.ipynb
- CropDiseaseDetection_DL_BhargavKoya_20075511_Report.docx
