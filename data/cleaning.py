# BREAST CANCER DATASET 
#CLEANING OF DATASET WDBC

# values/cleaning.py

import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import LabelEncoder

def drop_id_column(df):
    if "id" in df.columns:
        df = df.drop("id", axis=1)
    return df

def encode_target(df, target_column="diagnosis"):
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])
    return df, le

def prepare_data(df):
    df = drop_id_column(df)
    df, le = encode_target(df)

    y = df["diagnosis"]
    X = df.drop("diagnosis", axis=1)

    return X, y


def load_data(path="data/breast+cancer+wisconsin+diagnostic/wdbc.data"):
    columns = ["id", "diagnosis"] + [f"feature_{i}" for i in range(30)]
    df = pd.read_csv(path, header=None, names=columns)
    return df


def clean_wdbc(df):
    """
    Clean WDBC dataset
    """

    # ✅ Guardamos shape inicial
    initial_shape = df.shape

    # ----------------------------
    # 1️⃣ Normalizar nombres columnas
    # ----------------------------
    df.columns = (
        df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
    )

    df.columns = [re.sub(r'[^a-z0-9_]', '', col) for col in df.columns]

    # ----------------------------
    # 2️⃣ Eliminar columna ID
    # ----------------------------
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    # ----------------------------
    # 3️⃣ Eliminar filas completamente vacías
    # ----------------------------
    df.dropna(how="all", inplace=True)

    # ----------------------------
    # 4️⃣ Eliminar duplicados
    # ----------------------------
    df.drop_duplicates(inplace=True)

    # ----------------------------
    # 5️⃣ Manejo robusto de diagnosis
    # ----------------------------
    if "diagnosis" not in df.columns:
        raise ValueError("La columna 'diagnosis' no existe")

    # Si viene como M/B
    if df["diagnosis"].dtype == "object":
        df["diagnosis"] = (
            df["diagnosis"]
            .astype(str)
            .str.strip()
            .str.upper()
        )
        df = df[df["diagnosis"].isin(["M", "B"])]
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    # Si ya viene como 0/1
    elif set(df["diagnosis"].unique()).issubset({0, 1}):
        df["diagnosis"] = df["diagnosis"].astype(int)

    else:
        raise ValueError("Formato inesperado en 'diagnosis'")

    # ----------------------------
    # 6️⃣ Eliminar filas con nulos restantes
    # ----------------------------
    if df.isnull().sum().sum() > 0:
        df.dropna(inplace=True)

    # ----------------------------
    # 📊 Reporte final
    # ----------------------------
    print("✅ Dataset limpio correctamente")
    print(f"📊 Shape inicial: {initial_shape}")
    print(f"📊 Shape final: {df.shape}")
    print(f"🗑 Filas eliminadas totales: {initial_shape[0] - df.shape[0]}")
   
    return df
