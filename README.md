# Trabajo 1 — Clasificación de Churn en Telco

**Asignatura:** Desarrollo de Aplicaciones de Aprendizaje Automático  
**Autor:** Daniel Vega Rosado  
**Universidad:** UEMC — Ingeniería Informática (3º curso)

## Descripción

Práctica de clasificación supervisada sobre el dataset **Telco Customer Churn** de IBM. El objetivo es predecir si un cliente cancelará su contrato (`Churn = Yes/No`) usando un **Random Forest** con optimización de hiperparámetros mediante **Optuna**.

## Dataset

`WA_Fn-UseC_-Telco-Customer-Churn.csv` — Dataset público de IBM que recoge información de ~7000 clientes de una empresa de telecomunicaciones: datos demográficos, servicios contratados y si causaron baja.

## Estructura del repositorio

```
├── Tarea_1.py                          # Script principal
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── Trabajo1_Daniel_Vega_Rosado.pdf     # Memoria del trabajo
├── practica_bloque_1.pdf               # Enunciado de la práctica
└── README.md
```

## Metodología

1. **Carga y preprocesado** del dataset: eliminación de columnas irrelevantes, conversión de tipos, codificación one-hot de variables categóricas.
2. **División** train/test (80/20) con `random_state=42`.
3. **Optimización de hiperparámetros** con Optuna (50 trials, maximizando accuracy en validación cruzada 5-fold):
   - `n_estimators` ∈ [50, 300]
   - `max_depth` ∈ [2, 20]
   - `min_samples_split` ∈ [2, 10]
   - `min_samples_leaf` ∈ [1, 5]
4. **Entrenamiento** del modelo final con los mejores hiperparámetros (Pipeline con StandardScaler).
5. **Evaluación** sobre el conjunto de test: accuracy, precision, recall, F1-score (macro) y matriz de confusión.
6. **Importancia de variables**: top 10 features más relevantes del modelo.

## Requisitos

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python Tarea_1.py
```

El script imprime en consola las métricas del modelo y las 10 variables más importantes.

## Tecnologías

- Python 3.x
- pandas
- scikit-learn
- optuna
