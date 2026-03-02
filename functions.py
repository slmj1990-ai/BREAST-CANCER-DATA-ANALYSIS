 
#BREAST CANCER DATA FUNCTIONS  


# values/functions.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==============================
# SPLIT
# ==============================

def split_data(df, target="diagnosis", test_size=0.2, random_state=42):
    X = df.drop(target, axis=1)
    y = df[target]
    
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


# ==============================
# METRICAS CLASIFICACION
# ==============================

def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }


# ==============================
# ENTRENAMIENTO GENERICO
# ==============================

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


# ==============================
# EVALUACION COMPLETA
# ==============================

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    metrics = classification_metrics(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return metrics, cm


# ==============================
# COMPARAR MODELOS
# ==============================

def compare_models(results_dict):
    """
    results_dict ejemplo:
    {
        "RandomForest": {"f1_score": 0.95, ...},
        "DecisionTree": {"f1_score": 0.91, ...}
    }
    """
    comparison = {}
    
    for model_name, metrics in results_dict.items():
        comparison[model_name] = metrics["f1_score"]
    
    return comparison