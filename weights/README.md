# Pesos del modelo

| Recurso | Uso | Descarga |
|---|---|---|
| YOLOv5s COCO v7.0 | Inicialización de B1, M1 y M2 | [Pesos oficiales](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt) |
| B1 `best.pt` | Evaluación del baseline | [Releases](https://github.com/therockbeats/final-mia203/releases) |
| M1 `best.pt` | Evaluación a 1280 | [Releases](https://github.com/therockbeats/final-mia203/releases) |
| M2 `best.pt` | Modelo multiescala seleccionado; inferencia a 640 | Recibido; publicación en Releases pendiente |

Cada ejecución guarda `best.pt` y `last.pt` en `runs/<experimento>/seed<semilla>/run/weights/`, dentro de la carpeta de trabajo. `best.pt` contiene el modelo seleccionado por fitness de validación. `last.pt` permite recuperar entrenamientos interrumpidos mientras conserva el estado del optimizador. El protocolo final identifica los pesos elegidos para la evaluación.

`configs/final_selection.json` identifica el checkpoint M2 utilizado por el grupo para TEST.

Las corridas originales B1/M1/M2 incluyen `best.pt`, `last.pt`, historial y configuración. El comando `import-m2` incorpora la tercera corrida y verifica su checkpoint frente a `configs/final_selection.json`. Los enlaces de Releases se completan al publicar los archivos.
