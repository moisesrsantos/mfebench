# Empirical Comparison of Meta-feature Extractors for Time Series Forecasting

---

## Overview
This repository contains the implementation for the paper **"Empirical Comparison of Meta-feature Extractors for Time Series Forecasting"** by Moisés Santos, Vitor Cerqueira, and Carlos Soares. The paper evaluates the effectiveness of different meta-feature extraction methods for time series forecasting algorithm selection.

## Experiment Description
The study investigates how meta-features derived from time series influence the selection of forecasting algorithms. Four meta-feature extraction frameworks were compared: TSFRESH, TSFEATURES, TSFEL, and Catch22. The experiments utilized data from the M4 Competition, applying a range of forecasting models, including neural network-based methods, and analyzed the meta-level performance of these feature extractors.

---

## Repository Structure

- **`base_models.py`**  
  Implements time series forecasting using the base learners, such as Multi-Layer Perceptron (MLP), DeepAR, NHITS, and TCN.

- **`base_performance.py`**  
  Extracts the forecasting performance metrics (e.g., MAE) for each algorithm and dataset.

- **`mfe.py`**  
  Handles the extraction of meta-features from the time series data using the selected frameworks.

- **`metadata.py`**  
  Combines the extracted meta-features with the target values (based on forecasting performance) to generate the metadata required for meta-learning.

- **`analysis.ipynb`**  
  A Jupyter notebook containing all meta-analyses, visualizations, and statistical evaluations performed during the study.

---
