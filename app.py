import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_breast_cancer

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Configuración de página
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color: #f8f9fa;
}

h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
}

.metric-container {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
}

.stButton>button {
    background-color: #4ecdc4;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #38b2ac;
    color: white;
}

</style>
""", unsafe_allow_html=True)
# Ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Ruta imagen
image_path = BASE_DIR / "img.png"

# Mostrar imagen centrada
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image(str(image_path), use_container_width=True)

# Título
st.markdown("<h1 style='text-align: center;'>Breast Cancer Diagnostic Analysis</h1>", unsafe_allow_html=True)

# Subtítulo
st.markdown("<h3 style='text-align: center; color: gray;'>Exploratory Data Analysis & Machine Learning Model</h3>", unsafe_allow_html=True)

st.markdown("---")

# Ruta base del proyecto (subimos desde notebooks/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Ruta al dataset
data_path = BASE_DIR / "data" / "breast+cancer+wisconsin+diagnostic" / "wdbc.data"

st.markdown("##  Acerca del Proyecto")

st.write("""
Este proyecto analiza el **Conjunto de Datos de Diagnóstico de Cáncer de Mama de Wisconsin**
para explorar patrones en las características tumorales y construir un modelo de aprendizaje automático capaz de predecir si un tumor es **maligno** o **benigno**.

El objetivo es comprender qué características son las más influyentes y evaluar el rendimiento de un modelo de clasificación.
""")

st.markdown("---")

# ==========================
# LOAD DATASET (PROFESIONAL)
# ==========================

@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df

df = load_data()

# ==========================
# DESCRIPCION DE DATOS DEL DATASET 
# ==========================

st.markdown("## Datos Dataset")

col1, col2, col3 = st.columns(3)

col1.metric("Number of Samples", df.shape[0])
col2.metric("Number of Features", df.shape[1])
col3.metric("Target Classes", df["target"].nunique())

st.markdown("### Target Distribution")

df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})

st.markdown("### Target Distribution")

fig, ax = plt.subplots(figsize=(6,4))

sns.countplot(
    data=df,
    x="diagnosis",
    palette=["#ff6b6b", "#4ecdc4"],
    ax=ax
)

ax.set_title("Distribution of Tumor Diagnosis", fontsize=14)
ax.set_xlabel("Diagnosis")
ax.set_ylabel("Count")

st.pyplot(fig)

st.markdown("---")

# ==========================
# CORRELATION HEATMAP
# ==========================

st.markdown("## 🔥 Correlation Heatmap")

# Remove non-numeric columns
corr = df.drop(columns=["diagnosis"]).corr()

fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    ax=ax
)

ax.set_title("Feature Correlation Matrix", fontsize=16)

st.pyplot(fig)

st.markdown("---")

st.write("### Vista Preliminar Dataset")
st.dataframe(df.head())

# ==========================
# RANDOM FOREST MODEL
# ==========================

st.markdown("## 🤖 Random Forest Model")

# Features and target
X = df.drop(columns=["target", "diagnosis"])
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

colA, colB = st.columns(2)

colA.metric("Model Accuracy", f"{accuracy:.4f}")

colB.metric("Test Samples", len(y_test))

st.markdown("### Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Malignant", "Benign"],
    yticklabels=["Malignant", "Benign"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)

st.markdown("---")

# ==========================
# FEATURE IMPORTANCE
# ==========================

st.markdown("## Feature Importance")

importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Mostrar top 10
top_features = importance_df.head(10)

fig, ax = plt.subplots(figsize=(8,5))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature",
    palette="viridis",
    ax=ax
)

ax.set_title("Top 10 Most Important Features")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Feature")

st.pyplot(fig)

st.markdown("---")

# ==========================
# INTERACTIVE PREDICTION
# ==========================

st.markdown("## 🎯 Make a Prediction")

st.write("Adjust the values below to simulate a tumor profile:")

input_data = {}

selected_features = top_features["Feature"].values[:5]

for feature in selected_features:
    input_data[feature] = st.slider(
        feature,
        float(df[feature].min()),
        float(df[feature].max()),
        float(df[feature].mean())
    )

input_df = pd.DataFrame([input_data])

for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = df[col].mean()

input_df = input_df[X.columns]

if st.button("Predict Diagnosis"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    if prediction == 0:
        st.markdown("### 🔴 Prediction: Malignant")
        st.progress(float(probability[0]))
    else:
        st.markdown("### 🟢 Prediction: Benign")
        st.progress(float(probability[1]))

    st.write(f"Probability Malignant: {probability[0]:.4f}")
    st.write(f"Probability Benign: {probability[1]:.4f}")