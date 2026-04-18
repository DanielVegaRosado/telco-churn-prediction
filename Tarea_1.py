# -*- coding: utf-8 -*-
"""

@author: dvega
"""

import pandas as pd
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def load_churn():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop("Churn", axis=1)
    X = pd.get_dummies(X)
    
    return X, y

X, y = load_churn()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

best_model = RandomForestClassifier(**study.best_params, random_state=42)
pipeline_final = Pipeline([
    ("scaler", StandardScaler()),
    ("model", best_model)
])
pipeline_final.fit(X_train, y_train)

y_pred = pipeline_final.predict(X_test)
accuracy = pipeline_final.score(X_test, y_test)

precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n--- RESULTADOS DEL MODELO ---")
print("Accuracy:", accuracy)
print("Precision (macro):", precision)
print("Recall (macro):", recall)
print("F1-score (macro):", f1)
print("Matriz de confusión:\n", conf_matrix)

importances = best_model.feature_importances_
feature_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

print("\n--- LAS 10 VARIABLES MÁS IMPORTANTES ---")
print(feature_imp.head(10))