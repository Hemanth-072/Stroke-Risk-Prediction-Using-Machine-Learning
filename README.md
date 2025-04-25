# Stroke-Risk-Prediction-Using-Machine-Learning

Predictive analysis of brain stroke risk factors and model development using Python, SQL, and ML techniques.

## Table of Contents

1. [Overview](#overview)  
2. [Dataset](#dataset)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Methodology](#methodology)  
   - [Data Cleaning & Preparation](#data-cleaning--preparation)  
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
   - [Statistical Testing](#statistical-testing)  
   - [Feature Engineering](#feature-engineering)  
   - [Modeling](#modeling)  
6. [Results](#results)  
7. [Contributing](#contributing)  
8. [License](#license)  

---

## Overview

This project analyzes a publicly available stroke dataset to identify key risk factors and build predictive models for 30-day stroke occurrence. We apply data cleaning, statistical tests, visualization, and machine learning pipelines (Logistic Regression, Decision Tree, Random Forest, XGBoost) to evaluate model performance and interpretability.

## Dataset

- **Source:** [Kaggle: Brain Stroke Dataset](https://www.kaggle.com/datasets/jillanisofttech/brain-stroke-dataset)  
- **Records:** 4,981  
- **Features:**  
  - Demographics: `gender`, `age`, `ever_married`, `residence_type`, `work_type`, `smoking_status`  
  - Health: `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`  
  - Target: `stroke` (0 = no stroke, 1 = stroke)

## Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/Hemanth-072/Stroke-Risk-Prediction-ML.git
   cd Stroke-Risk-Prediction-ML

    Create a virtual environment and install dependencies:

    python3 -m venv venv
    source venv/bin/activate        # Linux/macOS
    # .\venv\Scripts\activate      # Windows
    pip install -r requirements.txt

    Place brain_stroke.csv in the project root (or modify the path in the notebooks/scripts).

Usage

    Jupyter Notebooks:

        01_Data_Cleaning_EDA.ipynb – Data loading, cleaning, EDA, outlier handling

        02_Statistical_Analysis.ipynb – Normality tests, chi-square, Mann–Whitney U, Kruskal–Wallis

        03_Modeling.ipynb – Preprocessing pipelines, model training, evaluation, and comparison

    Scripts (optional):

    python src/train.py         # Runs end-to-end preprocessing and model training
    python src/predict.py       # Serves a saved model for inference

Methodology
Data Cleaning & Preparation

    Standardize column names and data types

    Confirm and handle missing values (none in this dataset)

    Winsorize outliers (1st–99th percentile) for continuous features

Exploratory Data Analysis (EDA)

    Histograms & box plots for age, avg_glucose_level, bmi

    Count plots for categorical variables by stroke outcome

    Correlation matrix for numeric and binary features

Statistical Testing

    Normality: Shapiro–Wilk on capped continuous features

    Non-parametric: Mann–Whitney U (continuous vs. stroke)

    Categorical: Chi-square tests for independence

    Multi-group: Kruskal–Wallis for distributions across stroke groups

Feature Engineering

    One-hot encoding for categorical variables

    Train-test split (80/20 stratified)

    SMOTE to address class imbalance

Modeling

    Baseline Logistic Regression (with class_weight='balanced')

    Decision Tree, Random Forest, and XGBoost classifiers

    Hyperparameter tuning via RandomizedSearchCV

    Evaluation metrics: ROC-AUC, PR-AUC, accuracy, precision, recall, F1-score, confusion matrices

    Model interpretability: SHAP / feature‐importance plots

Results

    Key Risk Factors: Age, hypertension, heart disease, marital status, work type, and smoking status all showed statistically significant associations with stroke risk (p < 0.05).

    Best Model: Random Forest and XGBoost achieved ROC-AUC ≈ 0.85 on the held-out test set.

    Interpretability: Feature importance analyses highlighted age and hypertension as top predictors.

For full metric tables, plots, and model outputs see the notebooks in this repo.
