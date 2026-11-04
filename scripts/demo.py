"""Carga los pesos entrenados y dibuja las detecciones sobre una imagen."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import paths, config, write_json
from src.runtime import verify_source, configure_cache
from src.inference import predict_image, verify_model_classes

p = argparse.ArgumentParser()
p.add_argument('--image', required=True)
p.add_argument('--weights', required=True)
p.add_argument('--output', required=True)
p.add_argument('--experiment', choices=['B1','M1','M2'], default='B1')
p.add_argument('--device', default='cpu')
p.add_argument('--gun-conf', type=float, help='Confianza de gun; person conserva el umbral del experimento')
a = p.parse_args()
if a.gun_conf is not None and not 0 <= a.gun_conf <= 1:
    p.error("--gun-conf debe estar entre 0 y 1")
settings = paths()
configure_cache(settings)
verify_source(settings['yolo'])
sys.path.insert(0, str(settings['yolo']))
import cv2
import torch
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
im = cv2.imread(a.image)
if im is None: raise ValueError('No se pudo leer la imagen')
device = select_device(a.device)
c = config(a.experiment)
gun_conf = a.gun_conf if a.gun_conf is not None else c['operating_conf']
c['prediction_conf'] = min(gun_conf, c['operating_conf'])
model = DetectMultiBackend(a.weights, device=device, fp16=False)
verify_model_classes(model)
detections, passes = predict_image(model, im, c, device, torch, letterbox, non_max_suppression, scale_boxes)
detections = [d for d in detections if d['confidence'] >= (gun_conf if d['class_id'] == 1 else c['operating_conf'])]
for d in detections:
    x1,y1,x2,y2 = map(round,d['box'])
    cv2.rectangle(im,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.putText(im, f"{['person','gun'][d['class_id']]} {d['confidence']:.2f}",
                (x1,max(15,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
out = Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
if not cv2.imwrite(str(out),im): raise RuntimeError('No se pudo guardar la imagen')
write_json(out.with_suffix('.json'), {'experiment':a.experiment,'confidence_by_class':{'person':c['operating_conf'],'gun':gun_conf},'detections':detections,'passes':passes})
print(out)
