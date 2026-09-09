# Student Performance Machine Learning Analysis

An end-to-end machine learning project analyzing student performance using supervised and unsupervised learning techniques. Several algorithms are implemented from scratch with NumPy and compared with PyTorch and scikit-learn implementations.

## Project Overview

This project explores how student characteristics such as study time, attendance, sleep, and previous exam performance can be used to predict academic outcomes and identify patterns within student data.

The project covers the complete machine learning workflow, including data preprocessing, feature selection, normalization, regression, classification, neural networks, and clustering.

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- PyTorch
- scikit-learn
- SciPy
- Jupyter Notebook

## Machine Learning Techniques

### Supervised Learning
- Linear Regression from scratch with NumPy
- Linear Regression verification with PyTorch
- Polynomial Regression
- Logistic Regression from scratch with NumPy
- Logistic Regression verification with PyTorch
- Decision threshold analysis
- Neural Network from scratch using forward propagation and backpropagation

### Unsupervised Learning
- K-Means from scratch with NumPy
- K-Means verification with scikit-learn
- Elbow Method
- Hierarchical Clustering
- DBSCAN

## Data Preprocessing

The preprocessing pipeline includes:

- Exploratory data analysis
- Missing-value detection and KNN imputation
- Duplicate removal
- Invalid-value filtering
- Outlier detection using IQR
- Feature selection
- Z-score normalization
- 80/20 training and validation split

The final model features were:

- `hours_studied`
- `attendance_rate`
- `sleep_hours`
- `prev_exam_score`

## Results

| Model / Analysis | Result |
|---|---|
| Linear Regression (NumPy) | Validation MSE: **38.18** |
| Linear Regression (PyTorch) | Validation MSE: **38.18** |
| Best Polynomial Regression | **Degree 2** |
| Logistic Regression (NumPy) | Validation Accuracy: **90.16%** |
| Logistic Regression (PyTorch) | Validation Accuracy: **90.16%** |
| Neural Network | **10/10 training examples classified correctly** |
| K-Means | **2 clusters** |
| Elbow Method | **K = 2** |
| DBSCAN | Selected **ε = 0.8** |
| K-Means vs. Pass/Fail | **82.08% alignment** |

The close agreement between the custom NumPy implementations and their PyTorch counterparts helped validate the from-scratch implementations.

K-Means also identified two groups that aligned with actual pass/fail outcomes at approximately 82.08%, despite the labels not being used during clustering.

## Key Takeaways

- Implementing machine learning algorithms from scratch provided a deeper understanding of gradient descent, loss functions, backpropagation, and clustering.
- Feature normalization was important for both gradient-based models and distance-based clustering.
- Increasing model complexity did not always improve validation performance, demonstrating the bias-variance tradeoff.
- Classification thresholds can significantly affect model performance.
- Unsupervised learning discovered meaningful structure in the student data without using target labels.

## Repository Structure

```text
student-performance-ml/
├── README.md
├── student_performance_analysis.ipynb
└── requirements.txt
```

## Running the Project

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Then open the Jupyter Notebook:

```bash
jupyter notebook student_performance_analysis.ipynb
```

# Dataset

The dataset used in this project was provided as part of a university course and is not included in this repository.

To run the notebook, place the required `train.csv` and `test.csv` files in this directory.

## Future Improvements

Future improvements could include testing additional machine learning models, performing more extensive hyperparameter tuning, evaluating additional classification metrics, and testing the models on larger datasets.
