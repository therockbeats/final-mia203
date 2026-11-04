# Resumen de evaluación en workstation

Modelos y semillas: B1/seed0, M1/seed0, M2/seed0.

Se exportaron 9 filas del validador YOLOv5. Las métricas complementarias se encuentran en comparison.csv y per_class.csv.

## Umbral de demostración

Criterio: mayor F1 de gun entre los modelos, semillas y umbrales evaluados en Validation; desempate por recall.

Mejor combinación: B1, semilla 0, confianza 0.40; Precision 0.892, Recall 0.671, F1 0.766.

## Rendimiento

Tiempo de procesamiento secuencial a batch 1 y FP32, con lectura, preparación, inferencia y posprocesamiento. rendimiento.csv conserva cada medición; rendimiento_resumen.csv indica cuántas se realizaron y su dispersión. La memoria representa el pico asignado por PyTorch en MiB.

## Revisión visual

revision_errores.csv contiene los ejemplos seleccionados. Las columnas occlusion, scene y observation se completan al revisar las imágenes. Las observaciones existentes se conservan al regenerar estas tablas.

## Validación YOLOv5

| Modelo | Semilla | Clase | P | R | mAP50 | mAP50:95 |
|---|---:|---|---:|---:|---:|---:|
| B1 | 0 | all | 0.931 | 0.797 | 0.852 | 0.670 |
| B1 | 0 | person | 0.959 | 0.920 | 0.957 | 0.822 |
| B1 | 0 | gun | 0.902 | 0.674 | 0.746 | 0.518 |
| M1 | 0 | all | 0.777 | 0.763 | 0.789 | 0.524 |
| M1 | 0 | person | 0.813 | 0.890 | 0.907 | 0.656 |
| M1 | 0 | gun | 0.742 | 0.635 | 0.672 | 0.393 |
| M2 | 0 | all | 0.910 | 0.815 | 0.861 | 0.680 |
| M2 | 0 | person | 0.955 | 0.931 | 0.963 | 0.831 |
| M2 | 0 | gun | 0.865 | 0.698 | 0.760 | 0.529 |

## Diagnóstico complementario de gun

P/R a confianza 0,25; armas pequeñas con área relativa ≤1 %.

| Modelo | Semilla | TP | FP | FN | Recall pequeño | Objetos pequeños |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 0 | 497 | 96 | 215 | 0.143 | 133 |
| M1 | 0 | 355 | 94 | 357 | 0.128 | 133 |
| M2 | 0 | 498 | 104 | 214 | 0.195 | 133 |
