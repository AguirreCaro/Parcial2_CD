"""
Módulo de Preprocesamiento de Datos (Versión Multiclase)
-----------------------------------
Este script contiene las funciones para cargar y preparar el dataset
de Wine Quality White en 3 categorías: Bajo, Medio y Alto.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """Carga el dataset especificando ';' como separador."""
    return pd.read_csv(path, sep=';')

def categorize_quality(df, target_col='quality'):
    """
    Convierte la calidad (3-9) en 3 categorías discretas:
    0: Bajo (calidad <= 5)
    1: Medio (calidad == 6)
    2: Alto (calidad >= 7)
    
    Argumentos:
        df (pd.DataFrame): Dataset original.
    Retorna:
        pd.DataFrame: Copia del dataset con la columna objetivo categorizada.
    """
    # OPERACIÓN SEGURA: Creamos una copia para proteger los datos originales
    df_copy = df.copy()
    
    def aplicar_reglas(val):
        if val <= 5:
            return 0  # Bajo (Vinos más básicos)
        elif val == 6:
            return 1  # Medio (Vino estándar/bueno)
        else:
            return 2  # Alto (Vinos premium/excelentes)
            
    df_copy[target_col] = df_copy[target_col].apply(aplicar_reglas)
    return df_copy

def prepare_data(df, target_col='quality', test_size=0.2, random_state=42):
    """
    Realiza el split y la estandarización de los datos.
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Es VITAL usar stratify=y en problemas multiclase para que los 3 grupos
    # estén representados proporcionalmente en train y test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    scaler = StandardScaler()
    # Fit solo en entrenamiento para evitar Data Leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler