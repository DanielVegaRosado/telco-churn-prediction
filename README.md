[README.md](https://github.com/user-attachments/files/26853701/README.md)
# Telco Customer Churn Prediction

**Subject:** Development of Machine Learning Applications  
**Author:** Daniel Vega Rosado  
**University:** UEMC — Computer Engineering (3rd year)

## Description

Supervised classification project on the **Telco Customer Churn** dataset by IBM. The goal is to predict whether a customer will cancel their contract (`Churn = Yes/No`) using a **Random Forest** with hyperparameter optimization via **Optuna**.

## Dataset

`WA_Fn-UseC_-Telco-Customer-Churn.csv` — Public IBM dataset containing information on ~7,000 telecom customers: demographic data, contracted services, and whether they churned.

## Repository Structure

```
├── Tarea_1.py                            # Main script
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── Trabajo1_Daniel_Vega_Rosado.pdf       # Project report
├── practica_bloque_1.pdf                 # Assignment statement
└── README.md
```

## Methodology

1. **Loading and preprocessing**: dropping irrelevant columns, type conversion, one-hot encoding of categorical variables.
2. **Train/test split** (80/20) with `random_state=42`.
3. **Hyperparameter optimization** with Optuna (50 trials, maximizing accuracy via 5-fold cross-validation):
   - `n_estimators` ∈ [50, 300]
   - `max_depth` ∈ [2, 20]
   - `min_samples_split` ∈ [2, 10]
   - `min_samples_leaf` ∈ [1, 5]
4. **Final model training** with best hyperparameters (Pipeline with StandardScaler).
5. **Evaluation** on the test set: accuracy, precision, recall, F1-score (macro) and confusion matrix.
6. **Feature importance**: top 10 most relevant features.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
python Tarea_1.py
```

The script prints the model metrics and the top 10 most important features to the console.

## Technologies

- Python 3.x
- pandas
- scikit-learn
- optuna
