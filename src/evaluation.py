"""Evaluador común: AP interpolada en 101 puntos, matching por confianza.

P/R se calculan a un umbral
operativo fijo; AP usa todas las predicciones retenidas desde conf=0.001.
"""
from pathlib import Path
from collections import defaultdict
import math
from .common import read_json, write_json, sha256
from .data import labels, images
from .geometry import iou, xywh_to_xyxy


def average_precision(events, gt_count):
    if gt_count == 0:
        return None
    events = sorted(events, key=lambda x: -x[0])
    tp = 0
    curve = []
    for index, (_, correct) in enumerate(events, 1):
        tp += correct
        curve.append((tp/gt_count, tp/index))
    return sum(max((p for r, p in curve if r >= point/100), default=0) for point in range(101))/101


def match(detections, targets, threshold):
    used = set()
    result = []
    for d in sorted(detections, key=lambda x: -x['confidence']):
        candidates = [(iou(d['box'], t['box']), i) for i, t in enumerate(targets)
                      if i not in used and t['class_id'] == d['class_id']]
        overlap, j = max(candidates, default=(0, -1))
        correct = j >= 0 and overlap >= threshold
        if correct:
            used.add(j)
        result.append((d, correct, j if correct else None))
    return result, used


def evaluate(prediction_file, dataset, output):
    pred = read_json(prediction_file)
    split, c = pred['split'], pred['signature']['config']
    if pred['partial']:
        raise ValueError('Una muestra de demostración no genera métricas formales')
    expected = {x.name for x in images(dataset, split)}
    actual = [x['image'] for x in pred['images']]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Predicciones incompletas o repetidas')
    events = {i: defaultdict(list) for i in range(10)}
    gt_counts, fixed = defaultdict(int), defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    errors, curve, small = [], [], {'gt': 0, 'tp': 0}
    for row in pred['images']:
        if row['image_sha256'] != sha256(Path(dataset)/'images'/split/row['image']):
            raise ValueError('Imagen distinta de la inferida')
        raw_labels = labels(dataset, split, Path(row['image']).stem)
        targets = [{'class_id': b[0], 'box': xywh_to_xyxy(b, row['width'], row['height']),
                    'area_relative': b[3]*b[4]} for b in raw_labels]
        for t in targets:
            gt_counts[t['class_id']] += 1
        detections = row['detections']
        if any(d['class_id'] not in (0, 1) or not math.isfinite(d['confidence']) or not 0 <= d['confidence'] <= 1
               or len(d['box']) != 4 or not all(math.isfinite(x) for x in d['box']) for d in detections):
            raise ValueError('Predicción inválida')
        for ti in range(10):
            matches, _ = match(detections, targets, 0.5 + ti*0.05)
            for d, correct, _ in matches:
                events[ti][d['class_id']].append((d['confidence'], int(correct)))
        matches, used = match([d for d in detections if d['confidence'] >= c['operating_conf']], targets, 0.5)
        for d, correct, _ in matches:
            fixed[d['class_id']]['tp' if correct else 'fp'] += 1
            if not correct:
                errors.append({'image': row['image'], 'type': 'FP', **d})
        for j, t in enumerate(targets):
            if j not in used:
                fixed[t['class_id']]['fn'] += 1
                errors.append({'image': row['image'], 'type': 'FN', **t})
            if t['class_id'] == 1 and t['area_relative'] <= c['small_area_relative'] + 1e-12:
                small['gt'] += 1
                small['tp'] += j in used
    per_class = {}
    for cls in (0, 1):
        vals = [average_precision(events[ti][cls], gt_counts[cls]) for ti in range(10)]
        f = fixed[cls]
        precision = f['tp']/max(1, f['tp']+f['fp'])
        recall = f['tp']/gt_counts[cls] if gt_counts[cls] else None
        per_class[['person', 'gun'][cls]] = {**f, 'instances': gt_counts[cls], 'precision': precision,
            'recall': recall, 'f1': 2*precision*recall/max(precision+recall, 1e-12) if recall is not None else None,
            'map50': vals[0], 'map50_95': sum(vals)/10 if vals[0] is not None else None}
        sorted_events = sorted(events[0][cls], key=lambda e: -e[0])
        tp = 0
        for j, (confidence, correct) in enumerate(sorted_events, 1):
            tp += correct
            curve.append({'class': ['person', 'gun'][cls], 'confidence': confidence,
                          'precision': tp/j, 'recall': tp/max(1, gt_counts[cls])})
    valid = [v for v in per_class.values() if v['map50'] is not None]
    result = {'experiment': pred['experiment'], 'split': split, 'seed': pred['seed'],
        'evaluator': 'confidence-greedy AP101 v1; fixed P/R at operating_conf',
        'prediction_sha256': sha256(prediction_file), 'signature': pred['signature'],
        'operating_conf': c['operating_conf'], 'per_class': per_class,
        'macro': {k: sum(v[k] for v in valid)/len(valid) if valid else None for k in ['precision', 'recall', 'f1', 'map50', 'map50_95']},
        'small_gun': {**small, 'max_area_relative': c['small_area_relative'], 'recall': small['tp']/small['gt'] if small['gt'] else None},
        'mean_image_seconds': sum(x['seconds'] for x in pred['images'])/len(pred['images']),
        'peak_allocated_mb': pred['peak_allocated_mb'], 'environment': pred['environment'], 'images': len(actual)}
    write_json(output, result)
    write_json(Path(output).with_name('errors.json'), errors)
    write_json(Path(output).with_name('pr_curve.json'), curve)
    return result
