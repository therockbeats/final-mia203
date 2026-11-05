# Resumen de evaluación en workstation

Modelos y semillas: B1/seed1, M2/seed1.

Se exportaron 6 filas del validador YOLOv5. Las métricas complementarias se encuentran en comparison.csv y per_class.csv.

## Umbral de demostración

Criterio: mayor F1 de gun entre los modelos, semillas y umbrales evaluados en Validation; desempate por recall.

Mejor combinación: B1, semilla 1, confianza 0.40; Precision 0.902, Recall 0.683, F1 0.777.

## Rendimiento

Tiempo de procesamiento secuencial a batch 1 y FP32, con lectura, preparación, inferencia y posprocesamiento. rendimiento.csv conserva cada medición; rendimiento_resumen.csv indica cuántas se realizaron y su dispersión. La memoria representa el pico asignado por PyTorch en MiB.

## Revisión visual

revision_errores.csv contiene los ejemplos seleccionados. Las columnas occlusion, scene y observation se completan al revisar las imágenes. Las observaciones existentes se conservan al regenerar estas tablas.

## Validación YOLOv5

| Modelo | Semilla | Clase | P | R | mAP50 | mAP50:95 |
|---|---:|---|---:|---:|---:|---:|
| B1 | 1 | all | 0.933 | 0.802 | 0.863 | 0.688 |
| B1 | 1 | person | 0.975 | 0.922 | 0.964 | 0.836 |
| B1 | 1 | gun | 0.892 | 0.683 | 0.762 | 0.539 |
| M2 | 1 | all | 0.921 | 0.800 | 0.862 | 0.676 |
| M2 | 1 | person | 0.945 | 0.912 | 0.962 | 0.827 |
| M2 | 1 | gun | 0.897 | 0.688 | 0.762 | 0.524 |

## Diagnóstico complementario de gun

P/R a confianza 0,25; armas pequeñas con área relativa ≤1 %.

| Modelo | Semilla | TP | FP | FN | Recall pequeño | Objetos pequeños |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 1 | 503 | 97 | 209 | 0.165 | 133 |
| M2 | 1 | 500 | 105 | 212 | 0.173 | 133 |
