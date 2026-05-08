"""
Módulo de Evaluación de Modelos
------------------------------
Funciones para calcular métricas y realizar validación cruzada.
"""
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_with_cv(model, X, y, cv=5):
    """Realiza validación cruzada y retorna los scores."""
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
    return scores

def plot_confusion_matrix(y_true, y_pred, title="Matriz de Confusión"):
    """Genera un mapa de calor de la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Bajo', 'Medio', 'Alto'],
                yticklabels=['Bajo', 'Medio', 'Alto'])
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    plt.title(title)
    plt.show()