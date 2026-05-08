#Funciones para optimización de hiperparámetros.
"""
Módulo de Optimización de Hiperparámetros
----------------------------------------
Funciones para búsqueda de mejores parámetros usando GridSearch.
"""
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def run_rf_grid_search(X_train, y_train):
    """
    Ejecuta una búsqueda en rejilla para Random Forest.
    """
    # Definimos el espacio de búsqueda
    param_grid = {
        'n_estimators': [100, 200],         # Cantidad de árboles
        'max_depth': [None, 10, 20],        # Profundidad máxima
        'min_samples_split': [2, 5],        # Mínimo de muestras para dividir un nodo
        'criterion': ['gini', 'entropy']    # Calidad de la división
    }
    
    # Configuramos el modelo base
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    # Configuramos la búsqueda (Usamos f1_macro por el desbalance)
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=3, 
        scoring='f1_macro', 
        n_jobs=-1, # Usa todos los núcleos del procesador
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_