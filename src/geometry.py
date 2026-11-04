"""Geometría en coordenadas originales y NMS por clase."""
import math


def iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-12)


def nms(detections, threshold=0.6, max_det=300):
    kept = []
    for d in sorted(detections, key=lambda x: -x['confidence']):
        if all(d['class_id'] != k['class_id'] or iou(d['box'], k['box']) <= threshold for k in kept):
            kept.append(d)
            if len(kept) >= max_det:
                break
    return kept


def windows(width, height, fraction=0.6, overlap=0.2):
    if not 0 < fraction <= 1 or not 0 <= overlap < 1:
        raise ValueError('Geometría de recortes inválida')
    def starts(length, size):
        step = max(1, round(size * (1 - overlap)))
        return sorted(set(list(range(0, max(1, length - size + 1), step)) + [max(0, length - size)]))
    tw, th = max(1, math.ceil(width * fraction)), max(1, math.ceil(height * fraction))
    return [(x, y, min(width, x + tw), min(height, y + th)) for y in starts(height, th) for x in starts(width, tw)]


def xywh_to_xyxy(box, width, height):
    _, x, y, w, h = box
    return [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]
