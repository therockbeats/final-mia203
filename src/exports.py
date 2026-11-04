"""Exporta métricas y tablas publicables de las ejecuciones de la workstation."""
import csv
import json
import re
import shutil
import statistics
from pathlib import Path

from .common import ROOT, read_json

METRICS = ('precision', 'recall', 'map50', 'map50_95')
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
NUMBER = r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
ROW = re.compile(r'^\s*(all|person|gun)\s+(\d+)\s+(\d+)\s+' +
                 r'\s+'.join([f'({NUMBER})']*4) + r'(?=\s|:|$)')


def csv_file(file, rows):
    if not rows:
        return
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(file.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(file)


def output_folder(runs):
    series = 'replicas' if Path(runs).name == 'runs_replicas_windows' else 'grupo'
    return ROOT/'results/workstation'/series


def parse_yolov5_log(text):
    """Lee la última tabla completa, conservando los decimales del log original."""
    if 'Traceback (most recent call last)' in text:
        raise ValueError('La validación terminó con un error; revisar su log')
    completed, current = None, {}
    for line in ANSI.sub('', text).splitlines():
        row = ROW.match(line)
        if not row:
            continue
        cls, images, instances, *values = row.groups()
        if cls == 'all':
            current = {}
        values = list(map(float, values))
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError('Métrica YOLOv5 fuera del rango [0, 1]')
        current[cls] = {'class': cls, 'images': int(images), 'instances': int(instances),
                        **dict(zip(METRICS, values))}
        if set(current) == {'all', 'person', 'gun'}:
            completed = [current[c] for c in ['all', 'person', 'gun']]
    if completed is None or set(current) != {'all', 'person', 'gun'}:
        raise ValueError('El log todavía no contiene la tabla final all/person/gun completa')
    if len({r['images'] for r in completed}) != 1:
        raise ValueError('Cantidad de imágenes inconsistente en el log')
    if completed[0]['instances'] != sum(r['instances'] for r in completed[1:]):
        raise ValueError('Cantidad de objetos inconsistente en el log')
    return completed


def export_yolov5(runs, out=None, experiments=None, seeds=None, split='val', strict=True):
    runs, out = Path(runs), Path(out) if out is not None else output_folder(runs)
    rows = []
    for name in experiments or ['B1', 'M1', 'M2']:
        selected_seeds = seeds if seeds is not None else sorted(
            int(p.name[4:]) for p in (runs/name).glob('seed*') if p.name[4:].isdigit())
        for seed in selected_seeds:
            parent = runs/name/f'seed{seed}'/f'{split}_yolov5'
            folders = [p for p in parent.glob('run*')
                       if p.is_dir() and re.fullmatch(r'run(?:\d+)?', p.name)]
            latest = max(folders, key=lambda p: int(p.name[3:] or '1')) if folders else None
            log = latest/'validation.log' if latest else None
            try:
                if log is None or not log.is_file():
                    raise ValueError(f'Falta validación de {name}, semilla {seed}')
                parsed = parse_yolov5_log(log.read_text(encoding='utf-8', errors='replace'))
            except ValueError:
                if strict:
                    raise
                continue
            rows.extend({'experiment': name, 'seed': seed, 'split': split, **row,
                         'evaluator': 'YOLOv5 v7.0 val.py'} for row in parsed)
    if strict and not rows:
        raise ValueError('No hay validaciones completas para exportar')
    filename = 'validacion_yolov5.csv' if split == 'val' else 'evaluacion_test_yolov5.csv'
    csv_file(out/filename, rows)
    summary = []
    for name, cls in sorted({(r['experiment'], r['class']) for r in rows}):
        group = [r for r in rows if r['experiment'] == name and r['class'] == cls]
        record = {'experiment': name, 'class': cls, 'split': split, 'n_seeds': len(group)}
        for metric in METRICS:
            values = [r[metric] for r in group]
            record[metric+'_mean'] = statistics.mean(values)
            record[metric+'_sd'] = statistics.stdev(values) if len(values) > 1 else ''
        summary.append(record)
    csv_file(out/(Path(filename).stem+'_resumen.csv'), summary)
    return rows


def export_thresholds(settings, files, out):
    from .data import labels
    from .geometry import xywh_to_xyxy
    from .evaluation import match
    rows = []
    for name, seed, file in files:
        pred = read_json(file.with_name('predictions.json'))
        if pred['split'] != 'val' or pred['partial']:
            raise ValueError('Los umbrales requieren predicciones completas de Validation')
        for threshold in [0.05, 0.10, 0.25, 0.40, 0.50, 0.70]:
            tp = fp = gt = 0
            for image in pred['images']:
                targets = [{'class_id': b[0], 'box': xywh_to_xyxy(b, image['width'], image['height'])}
                           for b in labels(settings['dataset'], 'val', Path(image['image']).stem) if b[0] == 1]
                detections = [d for d in image['detections']
                              if d['class_id'] == 1 and d['confidence'] >= threshold]
                matched, _ = match(detections, targets, 0.50)
                tp += sum(correct for _, correct, _ in matched)
                fp += sum(not correct for _, correct, _ in matched)
                gt += len(targets)
            precision, recall = tp/max(1, tp+fp), tp/max(1, gt)
            rows.append({'experiment': name, 'seed': seed, 'split': 'val', 'class': 'gun',
                         'confidence': threshold, 'gt': gt, 'tp': tp, 'fp': fp, 'fn': gt-tp,
                         'precision': precision, 'recall': recall,
                         'f1': 2*precision*recall/max(precision+recall, 1e-12),
                         'fp_per_image': fp/len(pred['images'])})
    csv_file(out/'umbrales_gun.csv', rows)
    return rows


def export_workstation(settings, out=None, experiments=None, seeds=None, strict=False):
    runs = settings['runs']
    out = Path(out) if out is not None else output_folder(runs)
    official = export_yolov5(runs, out, experiments, seeds, strict=strict)
    files, timings, annotations = [], [], []
    for name in experiments or ['B1', 'M1', 'M2']:
        for file in sorted((runs/name).glob('seed*/val_predictions/full/metrics.json')):
            seed = int(file.parents[2].name[4:])
            if seeds is not None and seed not in seeds:
                continue
            files.append((name, seed, file))
            folder = file.parent.parent
            archive = sorted(p for p in folder.glob('medicion_*')
                             if p.name[9:].isdigit() and (p/'metrics.json').is_file())
            sequence = [(int(p.name[9:]), p/'metrics.json') for p in archive]
            sequence.append((max([i for i, _ in sequence], default=0)+1, file))
            for repetition, metric_file in sequence:
                metric = read_json(metric_file)
                seconds = metric['mean_image_seconds']
                timings.append({'experiment': name, 'seed': seed, 'repetition': repetition,
                                'images': metric['images'], 'imgsz': metric['signature']['config']['imgsz'],
                                'batch': 1, 'precision': 'FP32', 'gpu': metric['environment'].get('gpu'),
                                'mean_image_ms': seconds*1000, 'fps': 1/seconds if seconds > 0 else None,
                                'peak_allocated_mib': metric['peak_allocated_mb']})
            errors = read_json(file.with_name('errors.json'))
            fn = sorted([r for r in errors if r['class_id'] == 1 and r['type'] == 'FN'],
                        key=lambda r: r['area_relative'])[:6]
            fp = sorted([r for r in errors if r['class_id'] == 1 and r['type'] == 'FP'],
                        key=lambda r: -r['confidence'])[:6]
            for row in fn+fp:
                area = row.get('area_relative')
                annotations.append({'experiment': name, 'seed': seed, 'image': row['image'],
                                    'error_type': row['type'], 'area_relative': area,
                                    'size_group': ('small' if area <= .01 else 'other') if area is not None else '',
                                    'occlusion': '', 'scene': '', 'observation': ''})
    if not files:
        return official
    # Keep generated figures together with their tables. Manual observations have separate filenames.
    generated = ROOT/'results/val'
    if generated.is_dir():
        shutil.copytree(generated, out, dirs_exist_ok=True)
    csv_file(out/'rendimiento.csv', timings)
    time_summary = []
    for name, seed in sorted({(r['experiment'], r['seed']) for r in timings}):
        group = [r for r in timings if r['experiment'] == name and r['seed'] == seed]
        row = {'experiment': name, 'seed': seed, 'n_repetitions': len(group)}
        for key in ['mean_image_ms', 'fps', 'peak_allocated_mib']:
            values = [r[key] for r in group if r[key] is not None]
            row[key+'_mean'] = statistics.mean(values) if values else None
            row[key+'_sd'] = statistics.stdev(values) if len(values) > 1 else ''
        time_summary.append(row)
    csv_file(out/'rendimiento_resumen.csv', time_summary)
    annotation_file = out/'revision_errores.csv'
    if not annotation_file.exists() or annotation_file.stat().st_size == 0:
        csv_file(annotation_file, annotations)
    thresholds = export_thresholds(settings, files, out)
    best = max(thresholds, key=lambda r: (r['f1'], r['recall']))
    notes = ['# Resumen de evaluación en workstation', '',
             f"Modelos y semillas: {', '.join(f'{name}/seed{seed}' for name, seed, _ in files)}.", '',
             f"Se exportaron {len(official)} filas del validador YOLOv5. Las métricas complementarias se encuentran en comparison.csv y per_class.csv.", '',
             '## Umbral de demostración', '',
             'Criterio: mayor F1 de gun entre los modelos, semillas y umbrales evaluados en Validation; desempate por recall.', '',
             f"Mejor combinación: {best['experiment']}, semilla {best['seed']}, confianza {best['confidence']:.2f}; "
             f"Precision {best['precision']:.3f}, Recall {best['recall']:.3f}, F1 {best['f1']:.3f}.", '',
             '## Rendimiento', '',
             'Tiempo de procesamiento secuencial a batch 1 y FP32, con lectura, preparación, inferencia y posprocesamiento. '
             'rendimiento.csv conserva cada medición; rendimiento_resumen.csv indica cuántas se realizaron y su dispersión. '
             'La memoria representa el pico asignado por PyTorch en MiB.', '',
             '## Revisión visual', '',
             'revision_errores.csv contiene los ejemplos seleccionados. Las columnas occlusion, scene y observation se completan '
             'al revisar las imágenes. Las observaciones existentes se conservan al regenerar estas tablas.', '']
    notes.extend(['## Validación YOLOv5', '',
                  '| Modelo | Semilla | Clase | P | R | mAP50 | mAP50:95 |',
                  '|---|---:|---|---:|---:|---:|---:|'])
    for row in official:
        notes.append(f"| {row['experiment']} | {row['seed']} | {row['class']} | "
                     f"{row['precision']:.3f} | {row['recall']:.3f} | {row['map50']:.3f} | {row['map50_95']:.3f} |")
    notes.extend(['', '## Diagnóstico complementario de gun', '',
                  'P/R a confianza 0,25; armas pequeñas con área relativa ≤1 %.', '',
                  '| Modelo | Semilla | TP | FP | FN | Recall pequeño | Objetos pequeños |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for name, seed, file in files:
        metric = read_json(file)
        gun, small = metric['per_class']['gun'], metric['small_gun']
        recall = f"{small['recall']:.3f}" if small['recall'] is not None else '--'
        notes.append(f"| {name} | {seed} | {gun['tp']} | {gun['fp']} | {gun['fn']} | {recall} | {small['gt']} |")
    notes.append('')
    (out/'resumen_automatico.md').write_text('\n'.join(notes), encoding='utf-8', newline='\n')
    return official
