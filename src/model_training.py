"""
Módulo de Entrenamiento de Modelos
----------------------------------
Contiene la configuración de los algoritmos de clasificación.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def get_models(random_state=42):
    """
    Define y retorna los modelos a evaluar.
    Se usa 'class_weight=balanced' para compensar que la clase 2 (Alto) 
    tiene menos datos que la clase 1 (Medio).
    """
    models = {
        'Random_Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight='balanced'
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            random_state=random_state,
            class_weight='balanced'
        )
    }
    return models

def train_model(model, X_train, y_train):
    """Entrena un modelo específico."""
    return model.fit(X_train, y_train)