# Breast Cancer Prediction using Logistic Regression

This project implements a machine learning model to predict breast cancer diagnosis based on features extracted from a dataset. The model uses logistic regression for classification.

---

## Dataset

- The dataset `breast_cancer.csv` contains feature columns and a target diagnosis column.
- Features are used as input variables (X).
- The diagnosis column is the target variable (y).

---

## Project Overview

- Load and preprocess the dataset using `pandas`.
- Split the dataset into training and testing sets (80% train, 20% test).
- Train a logistic regression classifier on the training data.
- Predict on the test data and evaluate model performance.
- Perform 10-fold cross-validation to validate the accuracy and robustness of the model.

---

## Features

- Data preprocessing and splitting using `scikit-learn`.
- Logistic Regression model training and prediction.
- Evaluation using confusion matrix.
- Cross-validation to estimate model accuracy and variance.

---

## How to Run

1. Ensure you have the dataset file `breast_cancer.csv` in the project directory.
2. Install necessary Python libraries:
    ```bash
    pip install pandas scikit-learn
    ```
3. Run the Python script:
    ```bash
    python breast_cancer_prediction.py
    ```

---

## Output

- Predicted labels for the test dataset will be printed.
- Confusion matrix showing true vs predicted classifications.
- Average accuracy and standard deviation from 10-fold cross-validation.

---

## Dependencies

- Python 3.x
- pandas
- scikit-learn

---
