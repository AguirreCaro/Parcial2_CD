"""
Módulo de Preprocesamiento de Datos
-----------------------------------
Este script contiene las funciones necesarias para cargar, limpiar 
y preparar el dataset de Wine Quality para el modelado.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """
    Carga el dataset desde un archivo CSV.
    
    Argumentos:
        path (str): Ruta relativa o absoluta del archivo .csv
    Retorna:
        pd.DataFrame: Datos cargados.
    """
    # Se especifica ";" como separador según la naturaleza del dataset original
    return pd.read_csv(path, sep=';')

def binarize_quality(df, target_col='quality', threshold=6):
    """
    Convierte la variable objetivo 'quality' en categorías binarias.
    
    Argumentos:
        df (pd.DataFrame): Dataset original.
        target_col (str): Nombre de la columna objetivo.
        threshold (int): Valor de corte para definir alta calidad.
        
    Retorna:
        pd.DataFrame: Una copia del dataset con la columna objetivo binarizada.
    """
    # OPERACIÓN SEGURA: Creamos una copia para no alterar el DataFrame original en memoria
    df_copy = df.copy()
    
    # 0 para calidad < 6 (3, 4, 5) y 1 para >= 6 (6, 7, 8, 9)
    # Esto ayuda a balancear mejor el problema ante el solapamiento detectado en el PCA.
    df_copy[target_col] = (df_copy[target_col] >= threshold).astype(int)
    
    return df_copy

def prepare_data(df, target_col='quality', test_size=0.2, random_state=42):
    """
    Realiza la partición de datos (Train/Test) y la estandarización de características.
    
    Argumentos:
        df (pd.DataFrame): Dataset ya binarizado o procesado.
        target_col (str): Nombre de la columna objetivo.
        test_size (float): Proporción de datos para el set de prueba.
        random_state (int): Semilla para asegurar reproducibilidad.
        
    Retorna:
        X_train, X_test, y_train, y_test (arrays): Datos listos para el modelo.
        scaler (StandardScaler): El objeto escalador entrenado para uso futuro.
    """
    # Separación de características (X) y etiqueta (y)
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Split con estratificación para mantener la proporción de clases (Indicador 2.2.1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    # Estandarización: Crucial para modelos sensibles a la escala como SVM o KNN
    scaler = StandardScaler()
    
    # Entrenamos el escalador solo con X_train para evitar "Data Leakage"
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler