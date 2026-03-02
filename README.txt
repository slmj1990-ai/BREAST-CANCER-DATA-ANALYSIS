Breast Cancer Classification Project
 Project Overview

This project aims to build and evaluate multiple machine learning models to predict whether a breast tumor is benign or malignant using the Breast Cancer Wisconsin dataset.

Given the medical context, minimizing false negatives is critical, making Recall for the malignant class the most important metric.

Dataset

Source: Breast Cancer Wisconsin Dataset

Observations: 569

Features: 30 numeric predictors

Target:

0 → Benign

1 → Malignant

No missing values were found in the final working dataset.

 Exploratory Data Analysis

Data type verification

Target distribution analysis

Feature correlation analysis

Outlier inspection

Feature importance exploration

Key Findings:

Slight class imbalance

Strong correlation among some radius/area/perimeter variables

Dataset well structured for classification tasks

Preprocessing

Train/Test Split (Stratified)

Feature Scaling (Standardization / Normalization)

Feature Selection

Class balancing:

Oversampling (SMOTE)

Undersampling

 Models Implemented

K-Nearest Neighbors (KNN)

Logistic Regression

Decision Tree

Random Forest

Ensemble Methods:

Bagging

Pasting

AdaBoost

Gradient Boosting

 Model Optimization

Grid Search

Random Search

Cross Validation

Hyperparameters were tuned to improve recall and overall performance.

 Evaluation Metrics

Accuracy

Precision

Recall (Malignant class priority)

F1-score

ROC-AUC

Confusion Matrix

Final Model

The best performing model was:

 Random Forest

It achieved:

High recall for malignant cases

Strong ROC-AUC

Balanced precision and F1-score

This model minimizes the risk of undetected malignant tumors.

 Class Imbalance Strategy

Comparison of:

Original dataset

SMOTE oversampling

Undersampling

SMOTE provided better recall without significant loss in precision.

Future Improvements

SHAP for model interpretability

Deployment with Streamlit

External dataset validation

Pipeline automation

Tech Stack

Python

Pandas

NumPy

Matplotlib / Seaborn

Scikit-learn

Imbalanced-learn

Author

Sara Leticia Marín Jáuregui
Machine Learning Project – 2026