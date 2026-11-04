"""Inferencia de los detectores B1, M1 y M2."""
from pathlib import Path
import sys
import time
import hashlib
import json
from .common import ROOT, paths, config, sha256, write_json, read_json, now
from .runtime import verify_source, environment, configure_cache
from .geometry import windows, nms
from .data import images


def implementation_digest():
    return hashlib.sha256(''.join(sha256(x) for x in sorted((ROOT/'src').glob('*.py'))).encode()).hexdigest()


def dataset_digest(p):
    from .data import verify_manifest
    manifest = verify_manifest(p['dataset'], p['audit'])
    return hashlib.sha256(json.dumps(manifest['manifest'], sort_keys=True).encode()).hexdigest()


def weight_path(p, name, seed):
    return p['runs'] / name / f'seed{seed}' / 'run/weights/best.pt'


def verify_model_classes(model):
    names = model.names
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    if names != ['person', 'gun']:
        raise ValueError('Se requiere un checkpoint entrenado con clases 0=person y 1=gun')


def signature(p, name, seed):
    owner = name
    imported = p['runs']/owner/f'seed{seed}'/'imported.provenance.json'
    if imported.exists():
        original = read_json(imported)
        current = config(owner)
        if original['seed'] != seed or any(current[k] != v for k,v in original['training_config'].items()):
            raise ValueError('Los pesos importados no corresponden a la configuración de entrenamiento actual')
    return {'config': config(name), 'seed': seed, 'weights_sha256': sha256(weight_path(p, name, seed)),
        'dataset_sha256': dataset_digest(p), 'implementation_sha256': implementation_digest(),
        'yolo': verify_source(p['yolo'])}


def predict_image(model, im, c, device, torch, letterbox, upstream_nms, scale_boxes):
    import numpy as np
    h, w = im.shape[:2]
    regions = [(0, 0, w, h)]
    if c['inference'] == 'tiled':
        regions += [r for r in windows(w, h, c['tile_fraction'], c['tile_overlap']) if r != regions[0]]
    detections = []
    for x1, y1, x2, y2 in regions:
        crop = im[y1:y2, x1:x2]
        img = letterbox(crop, c['imgsz'], stride=int(model.stride), auto=False)[0]
        img = np.ascontiguousarray(img.transpose((2, 0, 1))[::-1])
        tensor = torch.from_numpy(img).to(device).float() / 255
        with torch.inference_mode():
            raw = model(tensor[None], augment=False, visualize=False)
            boxes = upstream_nms(raw, c['prediction_conf'], c['nms_iou'], max_det=c['max_det'])[0]
            if len(boxes):
                boxes[:, :4] = scale_boxes(tensor.shape[1:], boxes[:, :4], crop.shape)
                for a, b, d, e, conf, cls in boxes.cpu().tolist():
                    detections.append({'box': [a+x1, b+y1, d+x1, e+y1], 'confidence': conf, 'class_id': int(cls)})
    return nms(detections, c['nms_iou'], c['max_det']), len(regions)


def predict(name, split, seed, limit=0):
    p, c = paths(), config(name)
    configure_cache(p)
    if split == 'test' and not (ROOT/'configs/frozen_protocol.json').is_file():
        raise ValueError('TEST bloqueado: primero complete validación y congele el protocolo con freeze')
    sig = signature(p, name, seed)
    if split == 'test':
        frozen = read_json(ROOT/'configs/frozen_protocol.json')
        if limit:
            raise ValueError('No se permiten evaluaciones TEST parciales')
        if frozen['experiments'].get(f'{name}/seed{seed}') != sig:
            raise ValueError('El protocolo o pesos cambiaron después de congelar')
    sys.path.insert(0, str(p['yolo']))
    import cv2
    import torch
    from models.common import DetectMultiBackend
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes
    from utils.torch_utils import select_device
    destination = p['runs']/name/f'seed{seed}'/f'{split}_predictions' / ('sample' if limit else 'full')
    if destination.exists():
        raise FileExistsError(f'Predicciones ya existentes: {destination}')
    device = select_device(c['device'])
    model = DetectMultiBackend(weight_path(p, name, seed), device=device, fp16=False)
    verify_model_classes(model)
    model.warmup(imgsz=(1, 3, c['imgsz'], c['imgsz']))
    rows = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    source_images = images(p['dataset'], split)
    if limit:
        source_images = source_images[:limit]
    destination.mkdir(parents=True)
    for image in source_images:
        start = time.perf_counter()
        im = cv2.imread(str(image))
        if im is None:
            raise ValueError(f'Imagen ilegible: {image}')
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        detections, passes = predict_image(model, im, c, device, torch, letterbox, non_max_suppression, scale_boxes)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        rows.append({'image': image.name, 'width': im.shape[1], 'height': im.shape[0],
            'image_sha256': sha256(image), 'detections': detections, 'passes': passes,
            'seconds': elapsed})
    result = {'experiment': name, 'split': split, 'seed': seed, 'partial': bool(limit),
        'signature': sig, 'environment': environment(), 'created_at': now(),
        'peak_allocated_mb': torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else None,
        'images': rows}
    write_json(destination/'predictions.json', result)
    return destination/'predictions.json'
