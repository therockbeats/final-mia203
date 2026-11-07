# Detección de personas y armas con YOLOv5s

MIA-203 · Redes Neuronales y Aprendizaje Profundo · UNI · Grupo 4

El proyecto compara tres configuraciones de YOLOv5s para detectar personas y armas en YouTube-GDD: una referencia a 640, entrenamiento a 1280 y entrenamiento multiescala. Los experimentos comparten las particiones, la inicialización COCO, el lote, el optimizador y la semilla. Para medir la variabilidad entre entrenamientos, B1 y M2 se repitieron además con las semillas 0, 1 y 2 en una RTX 4090.

## Experimentos

| ID | Entrenamiento | Inferencia | Estado |
|---|---|---|---|
| B1 | 640, 100 épocas, lote 16, SGD, semilla 0 | Imagen completa a 640 | Entrenamiento completo |
| M1 | 1280, 100 épocas, lote 16, SGD, semilla 0 | Imagen completa a 1280 | Entrenamiento completo |
| M2 | Base 640, entrenamiento multiescala, 100 épocas | Imagen completa a 640 | Entrenamiento completo; modelo seleccionado |

Los tres modelos parten del mismo checkpoint COCO. M1 utiliza 1280 durante entrenamiento e inferencia. M2 activa `multi_scale=True`: varía la resolución del lote aproximadamente entre 320 y 960, en múltiplos de 32, y evalúa a 640. Las configuraciones están en `configs/`.

YOLOv5 ajusta el peso de presencia de objeto con el factor `(imgsz/640)^2`. El repositorio incluye la adaptación de `random.randrange` para el entrenamiento multiescala con Python moderno.

### Recorrido experimental

B1 tuvo buen desempeño global, pero sus errores se concentraron en armas pequeñas, periféricas y en escenas con varias armas. M1 (1280) no mejoró: recuperó 22 armas que B1 omitía, perdió 37 que B1 detectaba y aumentó el tiempo de entrenamiento en 76 %. M2 (multiescala) obtuvo el mejor fitness de validación en la corrida única del grupo y fue el modelo seleccionado y evaluado en TEST. Las réplicas con tres semillas no confirman una ventaja de M2 sobre B1 (ver [Variabilidad entre semillas](#variabilidad-entre-semillas)). El prototipo de video con motor temporal (Streamlit, versiones V3 a V5) que describe el artículo se desarrolló fuera de este repositorio.

## Resultados disponibles

Los tres entrenamientos del grupo (semilla 0) completaron 100 épocas en A100 de 40 GB. Las métricas del checkpoint seleccionado de cada experimento en validación son:

| Métrica global | B1 | M1 | M2 · multiescala |
|---|---:|---:|---:|
| Mejor época, desde cero | 84 | 99 | 98 |
| Precision | 0,928 | 0,779 | 0,913 |
| Recall | 0,798 | 0,764 | 0,813 |
| mAP@0.5 | 0,852 | 0,790 | 0,862 |
| mAP@0.5:0.95 | 0,669 | 0,524 | 0,680 |
| Fitness | 0,688 | 0,551 | 0,698 |
| Horas de entrenamiento principal | 1,82 | 3,21 | 1,84 |

En esta corrida única, M2 superó a B1 en 1,08 puntos porcentuales de mAP@0.5:0.95. En el diagnóstico de gun a confianza 0,001, NMS IoU 0,60 y matching IoU 0,50, las omisiones pasaron de 132 a 122; los falsos positivos pasaron de 3.232 a 3.978. M2 recuperó 15 de las 110 armas que B1 y M1 omitían conjuntamente. Al umbral operativo 0,25, en cambio, las omisiones de gun son prácticamente iguales (215 en B1 y 214 en M2). Las réplicas con más semillas muestran que la diferencia entre B1 y M2 está dentro de la variación entre corridas; la caída de M1 sí es muy superior a esa variación.

El grupo seleccionó M2 con validación y lo evaluó en las 508 imágenes de TEST:

| Clase | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Promedio | 0,875 | 0,780 | 0,845 | 0,659 |
| person | 0,933 | 0,863 | 0,926 | 0,812 |
| gun | 0,817 | 0,697 | 0,765 | 0,506 |

`results/final_summary/` contiene las tablas de validación y TEST, contrastadas con los archivos originales del grupo, y sus figuras. `configs/final_selection.json` identifica el modelo elegido. Las carpetas `imported_B1/`, `imported_M1/` e `imported_M2/` conservan los historiales y curvas de las tres corridas completas. `results/final_test/` reúne las curvas y la matriz de confusión de M2 sobre TEST. Los checkpoints se almacenan en la carpeta externa de trabajo.

Para regenerar estas tablas y la figura desde el notebook original del grupo:

```bash
python scripts/import_final_results.py --notebook ../01_B1_YOLOv5s.ipynb --figures
```

## Variabilidad entre semillas

La comparación anterior usa una sola semilla. Para estimar cuánto cambia el resultado entre entrenamientos, B1 y M2 se repitieron con las semillas 0, 1 y 2 en una RTX 4090, con el mismo protocolo (100 épocas, lote 16, SGD, hiperparámetros por defecto de YOLOv5 v7.0). Junto con la corrida original del grupo, cada configuración suma cuatro entrenamientos. Todas las cifras provienen de la reevaluación en la workstation: mAP con `val.py`; omisiones de gun y recall de armas pequeñas con el evaluador complementario a confianza 0,25.

| Métrica (Validation) | B1, media ± DE | M2, media ± DE | M2 − B1 | p (permutación) |
|---|---:|---:|---:|---:|
| mAP@0.5 | 0,856 ± 0,005 | 0,859 ± 0,004 | +0,003 | 0,46 |
| mAP@0.5:0.95 | 0,679 ± 0,007 | 0,674 ± 0,005 | −0,005 | 0,31 |
| gun mAP@0.5:0.95 | 0,525 ± 0,010 | 0,522 ± 0,006 | −0,003 | 0,71 |
| Omisiones de gun (conf 0,25) | 211,0 ± 4,2 | 210,8 ± 7,5 | −0,2 | 1,00 |
| Recall de armas pequeñas | 0,167 ± 0,022 | 0,180 ± 0,029 | +0,013 | 0,51 |

Con cuatro entrenamientos por configuración no hay una diferencia medible entre B1 y M2. Las cuatro corridas de B1 van de 0,670 a 0,688 en mAP@0.5:0.95, un rango mayor que la ventaja de 0,011 observada con una sola semilla. Con la misma semilla 0, cambiar de A100 a RTX 4090 movió B1 de 0,670 a 0,678 y M2 de 0,680 a 0,671: fijar la semilla no reproduce el mismo resultado en otro hardware. El valor p proviene de una prueba de permutación exacta y bilateral sobre las 70 formas de repartir las ocho corridas; con 4 + 4 corridas, el menor p posible es 0,029.

M2 sigue siendo el modelo evaluado en TEST, porque se seleccionó y congeló con validación antes de abrir TEST. Las réplicas se ejecutaron después, solo sobre Validation, y no modificaron esa selección.

Para regenerar la tabla, el valor p y las horas de entrenamiento de las réplicas:

```bash
python scripts/seed_summary.py
```

El programa lee `results/workstation/grupo/` y `results/workstation/replicas/` y escribe en `results/workstation/semillas/`: `semillas_corridas.csv` (una fila por entrenamiento), `semillas_resumen.csv` (medias, desviaciones, diferencia y p), `semillas.tex` (tabla del artículo) y `costo_replicas.csv`. Los historiales de entrenamiento de las réplicas están en `results/workstation/replicas/training/`.

Las réplicas se entrenan y validan con una carpeta de ejecuciones propia: en `configs/paths.local.json`, `runs` debe apuntar a una carpeta llamada `runs_replicas_windows`, nombre que dirige las tablas a `results/workstation/replicas/`. Con esa configuración:

```bash
python scripts/reproduce.py train --seeds 0 1 2 --experiments B1 M2
for exp in B1 M2; do for seed in 0 1 2; do python scripts/validate_yolov5.py $exp --seed $seed; done; done
python scripts/reproduce.py validation --seeds 0 1 2 --experiments B1 M2
```

## Costo computacional

| Entrenamientos | GPU | Horas |
|---|---|---:|
| B1, M1 y M2 del grupo (semilla 0) | NVIDIA A100-SXM4-40GB | 6,87 |
| Réplicas de B1 y M2 (semillas 0, 1 y 2) | NVIDIA GeForce RTX 4090 | 8,02 |
| Total | | 14,89 |

Las horas del grupo provienen de `results/final_summary/compute_cost.csv`; las de las réplicas, de `results/workstation/semillas/costo_replicas.csv`. No incluyen las validaciones ni las pruebas de inferencia.

## Reevaluación en Corrida Local

Los resultados de validación se guardan en `results/workstation/grupo/`:

| Archivo | Contenido |
|---|---|
| `validacion_yolov5.csv` | Métricas de YOLOv5 por modelo y clase |
| `comparison.csv`, `per_class.csv` | Métricas del evaluador complementario |
| `umbrales_gun.csv` | Comparación de confianza para gun |
| `rendimiento.csv` | Tiempo, FPS y memoria |
| `resumen_automatico.md` | Resumen generado por `scripts/export_workstation.py` |

El diagnóstico por objeto/video y las mediciones repetidas se ejecutan con:

```bash
python scripts/workstation_round2.py
```

Los resultados quedan en la misma carpeta; los tiempos por imagen se conservan en la carpeta externa de ejecuciones.

## Instalación

Use Python 3.11, 3.12 o 3.13. El entrenamiento admite GPU CUDA; la demostración incluye ejecución en CPU.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

En Google Colab, abra `notebooks/01_experimentos_colab.ipynb`. Su instalación conserva PyTorch, torchvision y CUDA del entorno y configura el resto de dependencias. `notebooks/02_demo_colab.ipynb` carga los pesos y procesa una imagen.

## Datos y configuración

La versión experimental contiene 3.997 imágenes de entrenamiento, 494 de validación y 508 de prueba. Las clases son `0=person` y `1=gun`. Los archivos de datos y los checkpoints se almacenan en una carpeta de trabajo externa. La configuración de cada equipo se guarda en `configs/paths.local.json`.

```bash
python -m src.pipeline configure --work-dir ../ExamenFinal_artefactos
python -m src.pipeline extract --zip ../processed.zip
python -m src.pipeline audit
python -m src.pipeline prepare
python -m src.pipeline doctor
```

`audit` comprueba etiquetas, correspondencia con imágenes y separación de videos entre particiones. `prepare` descarga YOLOv5 v7.0, fija su revisión y aplica las adaptaciones de compatibilidad con PyTorch, Pillow y NumPy. Consulte `data/README.md` y `weights/README.md` para descargar los recursos del grupo.

Para incorporar los experimentos desde la estructura compartida del proyecto:

```bash
python -m src.pipeline import-b1 --source ../ExamenFinal_compartido
python -m src.pipeline import-m1 --source ../ExamenFinal_compartido
python -m src.pipeline import-m2 --source ../ExamenFinal_compartido
```

## Reproducción del protocolo del grupo

Con los pesos incorporados, estos comandos ejecutan `val.py` de YOLOv5 v7.0 y conservan las métricas, figuras y registros de validación en la carpeta de trabajo:

```bash
python scripts/validate_yolov5.py B1
python scripts/validate_yolov5.py M1
python scripts/validate_yolov5.py M2
```

`--device cpu` permite ejecutar estas pruebas en CPU. El lote de validación es 32, correspondiente al doble del lote de entrenamiento utilizado por YOLOv5. La evaluación final del modelo seleccionado se reproduce con:

```bash
python scripts/validate_yolov5.py M2 --split test
```

El programa comprueba que el checkpoint coincida con el seleccionado por el grupo y utiliza lote 16 para TEST.

Cada validación guarda sus métricas en CSV y actualiza las tablas bajo `results/workstation/grupo/` (o `replicas/` para la carpeta de réplicas). Para reunir validaciones ya terminadas, ejecute:

```bash
python scripts/export_workstation.py
```

El comando lee los logs existentes y genera `validacion_yolov5.csv` y `validacion_yolov5_resumen.csv`. Después del análisis complementario, también exporta automáticamente umbrales, tiempo, memoria, ejemplos para revisión y `resumen_automatico.md`.

## Entrenamiento

```bash
python -m src.pipeline train B1 --smoke
python -m src.pipeline train M1 --smoke
python -m src.pipeline train M2 --smoke
python scripts/reproduce.py train --seeds 0
```

`--smoke` ejecuta una prueba corta. El entrenamiento formal reutiliza una ejecución completa y reanuda una interrumpida mediante `last.pt`. Cada sesión conserva su configuración, duración y registro en la carpeta de trabajo. Los tres experimentos utilizan el mismo lote. Las repeticiones propias se ejecutan con una semilla o carpeta de trabajo nueva.

## Evaluación complementaria y análisis

El evaluador complementario del repositorio permite calcular AP101, recall por tamaño y ejemplos de errores a un punto operativo fijo:

```bash
python scripts/reproduce.py validation --seeds 0
python -m src.pipeline freeze --seeds 0 --experiments M2
python scripts/reproduce.py test --seeds 0 --experiments M2
```

La validación compara B1, M1 y M2; la fase final utiliza M2. Para varias semillas, añada por ejemplo `--seeds 0 1 2` a los comandos correspondientes.

El evaluador del repositorio calcula AP interpolada en 101 puntos y promedia los umbrales IoU de 0,50 a 0,95. Precision, Recall y F1 utilizan confianza operativa 0,25; AP utiliza predicciones desde 0,001. Las tablas comparativas aplican este evaluador a todos los experimentos. Las cifras de la comparación inicial corresponden a `val.py` de YOLOv5, que utiliza su propio cálculo de AP y selección del punto operativo. Por eso los valores absolutos de ambos evaluadores no se mezclan en una misma tabla: en las réplicas de B1, mAP@0.5:0.95 es 0,682 con `val.py` y 0,649 con el evaluador complementario. Las comparaciones entre experimentos se hacen siempre con el mismo evaluador.

El análisis por tamaño define un arma pequeña como una caja cuya área ocupa hasta el 1 % de la imagen original. Los informes incluyen su recall, AP por clase, curvas PR, ejemplos de errores, latencia por imagen y memoria asignada por PyTorch. La selección visual reúne las seis armas omitidas de menor área y los seis falsos positivos de mayor confianza.

```bash
python scripts/reproduce.py report --split val --seeds 0
python scripts/reproduce.py report --split test --seeds 0 --experiments M2
```

Los informes de esta evaluación complementaria se escriben en `results/<split>/`: `comparison.csv`, `per_class.csv`, `comparison_summary.json`, `comparison.tex` y figuras PNG/PDF. La fase de validación conserva además una copia publicable en `results/workstation/`, junto con las tablas automáticas de umbrales y rendimiento. Las observaciones manuales existentes en `revision_errores.csv` se conservan al regenerar los resultados.

## Demostración

```bash
python scripts/demo.py --image imagen.jpg --weights best.pt --experiment M2 --gun-conf 0.40 --output deteccion.jpg --device cpu
```

El programa dibuja cajas, clases y confianza sobre la imagen y guarda las detecciones y los umbrales en un archivo JSON junto a la imagen de salida. `--gun-conf` ajusta gun para la demostración; person conserva el umbral del experimento. La propuesta para M2 utiliza gun 0,40 y person 0,25, según el análisis de Validation.

## Presentación y publicación

La presentación se prepara en `presentation/` utilizando las tablas de `results/final_summary/` y las figuras originales recibidas. Cuando se incorpore `presentation/slides.tex`, se compila con:

```bash
bash scripts/build_documents.sh
```

Para preparar la publicación:

```bash
python scripts/check_repository.py
bash scripts/publish.sh https://github.com/therockbeats/final-mia203.git
git diff --cached
git commit -m "Integra el flujo experimental de detección"
bash scripts/publish.sh https://github.com/therockbeats/final-mia203.git --push
```

El repositorio reúne código, configuraciones, resultados y presentación. La entrega del código utiliza la etiqueta `v1.0-final` y los enlaces a los pesos. `DELIVERY.json` concentra esos enlaces. `python scripts/check_repository.py --final` comprueba los archivos de entrega.

## Contribuciones

| Integrante | Tareas en las que participó |
|---|---|
| Jose Gabriel Laymito Quispe | Planteamiento e implementación del notebook experimental; preparación del dataset; entrenamiento de B1, M1 y M2; análisis de errores y evaluación final en TEST. |
| Dina Yesica Huamán Huallpa | Planteamiento del problema; preparación del dataset, notebook y entrenamiento de B1 junto con Jose Gabriel Laymito Quispe; análisis de errores y preparación de M1; documentación del código; redacción del artículo y diapositivas. |
| Jose Miguel Alvarado Moreno | Integración del repositorio y automatización del flujo (auditoría de datos, entrenamiento, validación y exportación); reevaluación de B1, M1 y M2 en RTX 4090; réplicas de B1 y M2 con tres semillas y análisis de variabilidad; análisis de errores, umbrales y rendimiento; pruebas y documentación. |
| Frank Huaman Cancho | Interpretación de resultados; revisión y mejora del artículo; revisión del código Python y de la estructura del pipeline; diapositivas. |
| Christopher Nuñez |  |

## Alcance y fuentes

El estudio mide detección de personas y armas en YouTube-GDD. La evaluación de falsas alarmas en vigilancia requiere ampliar las escenas negativas: la versión actual contiene 41 imágenes sin armas en entrenamiento y tres en cada partición de evaluación. El artículo desarrolla las limitaciones y el uso responsable de los resultados. Las fuentes de datos, software y métodos figuran en `THIRD_PARTY.md`.
