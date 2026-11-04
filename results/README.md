# Resultados experimentales

`imported_B1/` reúne el historial de 100 épocas, las curvas de entrenamiento, la matriz de confusión y `summary.json` con las métricas de validación de B1. `imported_M1/` conserva el historial, las curvas y el resumen de M1. `imported_M2/summary.json` resume M2 a partir del notebook actualizado.

`final_summary/` reúne la comparación B1/M1/M2 en validación, el diagnóstico de errores, el costo computacional y las métricas TEST del modelo seleccionado M2. `b1_m1_validation/` reúne las tablas globales, el diagnóstico por clase y la figura comparativa extraídos de sus salidas. `dataset_summary.json` contiene la distribución del dataset experimental.

Las fases de evaluación generan las carpetas `val/` y `test/`:

| Archivo | Contenido |
|---|---|
| `comparison.csv` | Métricas globales, clase gun y costo por ejecución |
| `per_class.csv` | Precision, Recall, F1 y AP para cada clase |
| `comparison_summary.json` | Media y desviación estándar por experimento |
| `comparison.tex` | Tabla para el artículo |
| `gun_pr.png` y `gun_pr.pdf` | Curvas Precision–Recall de gun |
| `error_examples/` | Imágenes y selección de errores |

Los checkpoints, las predicciones completas y los registros de entrenamiento se guardan en la carpeta de trabajo configurada para cada equipo.

Las tablas finales se regeneran con `scripts/import_final_results.py`. Los archivos de `b1_m1_validation/` corresponden a la comparación inicial de dos modelos.
