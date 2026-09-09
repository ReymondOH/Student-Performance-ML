# Student Performance Machine Learning Analysis

## Overview

This project analyzes student performance data using supervised
and unsupervised machine learning techniques. The project includes
data preprocessing, regression, classification, neural networks,
and clustering.

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- PyTorch
- Scikit-learn
- SciPy

## Data Preprocessing

The dataset was cleaned by:

- Detecting duplicate observations while excluding student IDs
- Removing rows containing two or more missing values
- Using KNN imputation for remaining missing values
- Removing impossible values and outliers
- Normalizing features for model training

## Machine Learning Methods

### Linear Regression
Implemented linear regression from scratch using NumPy and
compared the results with PyTorch.

### Logistic Regression
Implemented logistic regression for student pass/fail prediction.

### Neural Network
Implemented a 2-2-1 neural network including forward propagation,
binary cross-entropy loss, backpropagation, and gradient descent.

### K-Means
Implemented K-Means clustering from scratch using NumPy and
compared the implementation against Scikit-learn.

### Additional Clustering
Explored student performance patterns using:
- Hierarchical clustering
- Agglomerative clustering
- DBSCAN

## Results

The models produced consistent results across the custom NumPy
implementations and their library-based counterparts.

| Model / Analysis | Result |
|---|---|
| Linear Regression (NumPy) | Validation MSE: 38.18 |
| Linear Regression (PyTorch) | Validation MSE: 38.18 |
| Logistic Regression (NumPy) | Validation Accuracy: 90.16% |
| Logistic Regression (PyTorch) | Validation Accuracy: 90.16% |
| Neural Network | 10/10 toy examples correctly classified |
| K-Means | 2 clusters (127 and 113 students) |
| K-Means vs. Pass/Fail | 82.08% alignment |
| DBSCAN | Best ε = 0.8 |

### Key Findings

- The NumPy implementations produced results nearly identical to their
  PyTorch equivalents, validating the from-scratch implementations.
- Logistic regression achieved approximately **90.16% validation accuracy**
  for predicting whether a student passed.
- Linear regression achieved a validation MSE of approximately **38.18**
  when predicting exam scores.
- The elbow method identified **2 clusters** as the appropriate choice
  for K-Means.
- Without using pass/fail labels during clustering, the K-Means clusters
  achieved **82.08% alignment** with the actual pass/fail groups.
