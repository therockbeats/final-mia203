# Dataset experimental

Fuente: [YouTube-GDD](https://github.com/UCAS-GYX/YouTube-GDD), Gu, Liao y Qin (2022).
Clases: `0=person`, `1=gun`.

| Partición | Imágenes | Personas | Armas |
|---|---:|---:|---:|
| Entrenamiento | 3.997 | 3.713 | 6.432 |
| Validación | 494 | 410 | 712 |
| Prueba | 508 | 400 | 888 |

La preparación del grupo excluyó un fotograma y trasladó fotogramas de dos videos a prueba para separar los videos entre particiones. `preparation_log.json`, incluido en el ZIP, describe esas operaciones. `results/dataset_summary.json` resume la distribución.

**Descarga del ZIP del grupo:** disponible a través de los releases y almacenamiento vinculado del repositorio ([final-mia203](https://github.com/therockbeats/final-mia203/releases)).

Guarde `processed.zip` en la carpeta de trabajo y ejecute `extract` y `audit`, según el README principal. Los experimentos utilizan esta misma versión curada del dataset. El repositorio de origen publica sus condiciones de uso y su licencia Apache-2.0.
