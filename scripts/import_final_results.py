"""Exporta B1/M1/M2 y la evaluación final registrados en el notebook del grupo."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.import_notebook_results import outputs


def csv_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def read_results(path):
    cells = list(outputs(json.loads(Path(path).read_text())))
    consolidated = next(t for _, t in cells if 'EXPERIMENTOS SOBRE VALIDATION' in t and 'MODELO FINAL M2 SOBRE TEST' in t)
    validation = []
    for line in consolidated.splitlines():
        fields = line.split()
        if fields and fields[0] in {'B1', 'M1', 'M2'} and 'Validation' in fields:
            pos = fields.index('Validation')
            values = fields[pos+1:]
            if len(values) != 8: raise ValueError('Tabla de validación con columnas inesperadas')
            validation.append({'experiment': fields[0], 'configuration': ' '.join(fields[1:pos]),
                'split': 'val', 'best_epoch': int(values[0]), 'precision': float(values[1]),
                'recall': float(values[2]), 'map50': float(values[3]), 'map50_95': float(values[4]),
                'fitness': float(values[5]), 'gun_tp_diagnostic': int(values[6]), 'gun_fn_diagnostic': int(values[7])})
    if {r['experiment'] for r in validation} != {'B1', 'M1', 'M2'}: raise ValueError('Faltan experimentos de validación')
    testing = []
    block = consolidated.split('MODELO FINAL M2 SOBRE TEST', 1)[1].split('MEJORA M2 RESPECTO DE B1', 1)[0]
    for line in block.splitlines():
        fields = line.split()
        if len(fields) == 7 and fields[0] in {'all', 'person', 'gun'} and fields[1].isdigit():
            testing.append({'experiment': 'M2', 'split': 'test', 'class': fields[0],
                'images': int(fields[1]), 'instances': int(fields[2]), 'precision': float(fields[3]),
                'recall': float(fields[4]), 'map50': float(fields[5]), 'map50_95': float(fields[6])})
    if {r['class'] for r in testing} != {'all', 'person', 'gun'}: raise ValueError('Tabla TEST incompleta')
    if any(r['images'] != 508 for r in testing): raise ValueError('Cantidad TEST inesperada')
    errors = next(t for _, t in cells if 'COMPARACION DIAGNOSTICA B1 VS M1 VS M2' in t)
    errors = errors.split('COMPARACION DIAGNOSTICA B1 VS M1 VS M2', 1)[1].split('EVALUACION DE LA HIPOTESIS M2', 1)[0]
    diagnostic = []
    for cls, block in re.findall(r'Clase:\s*(person|gun)(.*?)(?=Clase:|\Z)', errors, re.S):
        counts = {}
        for label in ['TP', 'FP', 'FN']:
            match = re.search(label + r' B1 / M1 / M2:\s*(\d+) / (\d+) / (\d+)', block)
            if not match: raise ValueError(f'Falta {label} en {cls}')
            counts[label] = [int(v) for v in match.groups()]
        for i, name in enumerate(['B1', 'M1', 'M2']):
            tp, fp, fn = [counts[k][i] for k in ['TP', 'FP', 'FN']]
            diagnostic.append({'experiment': name, 'class': cls, 'tp': tp, 'fp': fp, 'fn': fn,
                'precision': tp/(tp+fp), 'recall': tp/(tp+fn)})
    cost = next(t for _, t in cells if 'EVIDENCIA DE TIEMPOS EN LOGS' in t)
    compute = [{'experiment': name, 'training_hours': float(hours)} for name, hours in
        re.findall(r'\n(B1|M1|M2)\n.*?([\d.]+) hours', cost, re.S)]
    if len(compute) != 3: raise ValueError('Faltan tiempos de entrenamiento')
    selected = next(t for _, t in cells if 'RESULTADO: MODELO FINAL SELECCIONADO Y CONGELADO' in t)
    match = re.search(r'SHA256:\s*([a-f0-9]{64})', selected)
    if not match: raise ValueError('Falta la identificación del checkpoint final')
    selection = {'experiment': 'M2', 'model': 'YOLOv5s', 'training': 'multi-scale', 'imgsz': 640,
        'seed': 0, 'best_epoch': next(r['best_epoch'] for r in validation if r['experiment']=='M2'),
        'selection_split': 'val', 'expected_weights_sha256': match[1],
        'evaluation_split': 'test', 'evaluator': 'YOLOv5 v7.0', 'source': Path(path).name}
    return validation, testing, diagnostic, compute, selection


def export(path, figures=False):
    validation, testing, diagnostic, compute, selection = read_results(path)
    out = ROOT/'results/final_summary'
    for name, rows in [('validation_experiments', validation), ('final_test_metrics', testing),
                       ('diagnostic_validation', diagnostic), ('compute_cost', compute)]:
        csv_file(out/(name+'.csv'), rows)
    (ROOT/'configs/final_selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2)+'\n')
    summary = {'selected_experiment': 'M2', 'validation': validation, 'test': testing,
        'hard_gun': {'missed_by_b1_and_m1': 110, 'recovered_by_m2': 15, 'remaining': 95},
        'training_gpu_hours': round(sum(r['training_hours'] for r in compute), 3),
        'hardware': 'NVIDIA A100-SXM4-40GB', 'source': Path(path).name}
    (out/'final_experiment_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    folder = ROOT/'results/imported_M2'; folder.mkdir(parents=True, exist_ok=True)
    m2 = next(r for r in validation if r['experiment']=='M2')
    (folder/'summary.json').write_text(json.dumps({**m2, 'epochs': 100, 'multi_scale': True,
        'seed': 0, 'training_hours': next(r['training_hours'] for r in compute if r['experiment']=='M2'),
        'source': Path(path).name}, ensure_ascii=False, indent=2)+'\n')
    tex = [r'\begin{tabular}{lrrrr}', r'\toprule', r'Experimento & P & R & mAP50 & mAP50:95 \\', r'\midrule']
    for row in validation:
        tex.append(f"{row['experiment']} & {row['precision']:.3f} & {row['recall']:.3f} & {row['map50']:.3f} & {row['map50_95']:.3f} "+r'\\')
    (out/'validation.tex').write_text('\n'.join(tex+[r'\bottomrule', r'\end{tabular}'])+'\n')
    tex = [r'\begin{tabular}{lrrrr}', r'\toprule', r'Clase & P & R & mAP50 & mAP50:95 \\', r'\midrule']
    for row in testing:
        tex.append(f"{row['class']} & {row['precision']:.3f} & {row['recall']:.3f} & {row['map50']:.3f} & {row['map50_95']:.3f} "+r'\\')
    (out/'test.tex').write_text('\n'.join(tex+[r'\bottomrule', r'\end{tabular}'])+'\n')
    if figures:
        with tempfile.TemporaryDirectory() as cache:
            os.environ.setdefault('MPLCONFIGDIR', cache); os.environ.setdefault('XDG_CACHE_HOME', cache)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), layout='constrained')
            colors = ['#21618c', '#d68910', '#148f77']
            axes[0].bar([r['experiment'] for r in validation], [r['map50_95'] for r in validation], color=colors)
            axes[0].set(title='Validación · mAP@0.5:0.95', ylim=(0, 1), ylabel='mAP')
            guns = [next(r['fn'] for r in diagnostic if r['experiment']==name and r['class']=='gun') for name in ['B1','M1','M2']]
            axes[1].bar(['B1','M1','M2'], guns, color=colors)
            axes[1].set(title='Diagnóstico · confianza 0,001', ylabel='Armas omitidas', ylim=(0, 180))
            for ax in axes: ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
            fig.savefig(out/'validation_comparison.png', dpi=200); fig.savefig(out/'validation_comparison.pdf'); plt.close(fig)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--notebook', required=True); p.add_argument('--figures', action='store_true')
    a = p.parse_args(); print(export(a.notebook, a.figures))
