# Car Price Prediction Pipeline

Proyecto de Machine Learning basado en el capítulo 2 del libro "Hands-On Machine Learning" de Aurélien Géron.

## Estructura del Proyecto
- src/data_split.py: Split estratificado para evitar sesgos.
- src/num_data_clean.py: Limpieza de datos (luego queda "absorbido" por transformers.py).
- src/transformers.py: Transformadores personalizados.
- src/pipeline.py: Construcción del ColumnTransformer y Pipeline.
- notebooks/cars.ipynb: Análisis exploratorio y entrenamiento de modelos.
- notebooks/aed.ipynb: Análisis exploratorio de datos

## Modelos Evaluados
1. Linear Regression (Baseline)
2. Decision Tree (Overfitting detectado con Cross-Validation)
3. Random Forest (Modelo actual)