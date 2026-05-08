# Proyecto de Clasificación: Predicción de Calidad en Vinos Blancos

Este repositorio contiene el desarrollo integral de un pipeline de **Machine Learning** para la clasificación de calidad vinícola. El proyecto se basa en el dataset "Wine Quality - White" y busca categorizar los vinos en tres niveles de calidad (**Bajo, Medio y Alto**) mediante modelos supervisados y optimización avanzada.

## Resumen del Proyecto

El objetivo principal es identificar las propiedades fisicoquímicas que determinan la excelencia de un vino blanco. Para ello, se ha estructurado el trabajo siguiendo una metodología modular, separando la lógica de procesamiento, el entrenamiento de modelos y la evaluación final.

---

## Estructura del Repositorio

Para facilitar la revisión y asegurar la escalabilidad del código, el proyecto se organiza de la siguiente manera:

### Notebooks de Trabajo (Flujo de Ejecución)

Los notebooks están numerados secuencialmente. Se recomienda seguirlos en este orden para replicar los resultados:

1. **`01_exploratory_analysis.ipynb`**: Análisis descriptivo (EDA), detección de correlaciones y visualización mediante **PCA** (Análisis de Componentes Principales).
2. **`02_supervised_modeling.ipynb`**: Fase de construcción. Aquí se cargan los datos, se entrenan los modelos base (**Random Forest** y **SVM**) y se exportan los archivos `.pkl`.
3. **`03_model_evaluation.ipynb`**: Comparativa técnica. Análisis de matrices de confusión, reportes de métricas (Precision, Recall, F1) y **Validación Cruzada** para medir robustez.
4. **`04_hyperparameter_optimization.ipynb`**: Refinamiento del modelo ganador mediante `GridSearchCV`, optimizando parámetros para mejorar el desempeño en clases minoritarias.
5. **`05_final_analysis.ipynb`**: Conclusiones de ingeniería. Análisis de **importancia de variables** y síntesis de hallazgos para la toma de decisiones.

### Código Fuente (`/src`)

Este directorio contiene los módulos de Python que encapsulan la lógica reutilizable del proyecto:

* **`data_preprocessing.py`**: Funciones de ingesta, normalización (0-1), categorización triple y limpieza.
* **`model_training.py`**: Definición de arquitecturas de modelos y lógica de entrenamiento.
* **`model_evaluation.py`**: Herramientas visuales de evaluación y cálculo de métricas estadísticas.
* **`hyperparameter_tuning.py`**: Scripts para la búsqueda sistemática de hiperparámetros óptimos.

### Modelos y Resultados

* **`/models/trained_models/`**: Almacena los modelos serializados (`.pkl`). Incluye el modelo optimizado final y el escalador de datos (`scaler.pkl`).
* **`/results/`**: Contiene las métricas de rendimiento y los gráficos exportados (`/plots/`) para la documentación del informe técnico.
* **`/data/`**: Repositorio del dataset original `winequality-white.csv`.

---

## Guía para Colaboradores y Revisores

### Para Modificar la Lógica

Si se desea alterar el comportamiento del pipeline (ej. cambiar el método de escalado o añadir nuevos parámetros de búsqueda), se debe editar directamente el archivo correspondiente en `/src`. Esto asegura que los cambios se propaguen automáticamente a todos los notebooks gracias al sistema de recarga de módulos (`autoreload`).

### Ejecución del Proyecto

1. Asegurarse de tener instaladas las dependencias: `pandas`, `scikit-learn`, `matplotlib`, `seaborn` y `joblib`. Puedes ejecutar el siguiente comando en la consola y se instalarán `pip install -r requirements.txt`.
2. Mantener la estructura de carpetas actual para evitar errores de rutas relativas (`../data/`, `../models/`, etc.).
3. Al ejecutar el **Notebook 04**, el modelo final se guardará automáticamente, actualizando el análisis del **Notebook 05**.

