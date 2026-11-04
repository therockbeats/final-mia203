"""Exporta la comparación B1/M1 registrada en el notebook del grupo."""
from pathlib import Path
import argparse
import csv
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def outputs(notebook):
    for cell in notebook['cells']:
        text = '\n'.join(''.join(o.get('text', [])) for o in cell.get('outputs', []))
        yield ''.join(cell['source']), re.sub(r'\x1b\[[0-9;]*m', '', text)


def section(text, begin, end):
    return text.split(begin, 1)[1].split(end, 1)[0]


def scalar(text, label):
    match = re.search(r'^\s*' + re.escape(label) + r'\s*:\s*([\d.]+)', text, re.M)
    if not match:
        raise ValueError(f'Falta el campo {label} en el notebook')
    return float(match[1])


def export(notebook_file, b1_minutes=None, figures=False):
    cells = list(outputs(json.loads(Path(notebook_file).read_text(encoding='utf-8'))))
    analysis = next(t for _, t in cells if 'EPOCH SELECCIONADO POR FITNESS - M1' in t)
    formal = next(t for s, t in cells if 'CELDA 30' in s and '100 / 100' in t)
    errors = next(t for _, t in cells if 'COMPARACION DIAGNOSTICA B1 640 VS M1 1280' in t)
    chosen = section(analysis, 'EPOCH SELECCIONADO POR FITNESS - M1', 'ULTIMO EPOCH M1')
    with (ROOT/'results/imported_B1/results.csv').open(encoding='utf-8') as f:
        history = list(csv.DictReader(f))
    history = [{k.strip(): float(v) for k, v in row.items()} for row in history]
    best = max(history, key=lambda r: .1*r['metrics/mAP_0.5'] + .9*r['metrics/mAP_0.5:0.95'])
    b1 = {'experiment': 'B1', 'resolution': 640, 'best_epoch': int(best['epoch']),
          'precision': best['metrics/precision'], 'recall': best['metrics/recall'],
          'map50': best['metrics/mAP_0.5'], 'map50_95': best['metrics/mAP_0.5:0.95'],
          'fitness': .1*best['metrics/mAP_0.5'] + .9*best['metrics/mAP_0.5:0.95'],
          'execution_minutes': b1_minutes}
    minutes = re.search(r'Tiempo de esta ejecución:\s*([\d.]+) minutos', formal)
    m1 = {'experiment': 'M1', 'resolution': 1280, 'best_epoch': int(scalar(chosen, 'Epoch')),
          'precision': scalar(chosen, 'Precision'), 'recall': scalar(chosen, 'Recall'),
          'map50': scalar(chosen, 'mAP@0.5'), 'map50_95': scalar(chosen, 'mAP@0.5:0.95'),
          'fitness': scalar(chosen, 'Fitness'), 'execution_minutes': float(minutes[1]) if minutes else None}
    out = ROOT/'results/b1_m1_validation'
    out.mkdir(parents=True, exist_ok=True)
    with (out/'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(b1)); writer.writeheader(); writer.writerows([b1, m1])
    diagnostic = []
    blocks = section(errors, 'COMPARACION DIAGNOSTICA B1 640 VS M1 1280', 'EVALUACION DE LA HIPOTESIS M1')
    for cls, block in re.findall(r'Clase:\s*(person|gun)(.*?)(?=Clase:|\Z)', blocks, re.S):
        values = {}
        for label in ['TP', 'FP', 'FN']:
            match = re.search(label + r' B1 -> M1\s*:\s*(\d+) -> (\d+)', block)
            if not match:
                raise ValueError(f'Falta {label} de {cls}')
            values[label] = [int(match[1]), int(match[2])]
        for i, name in enumerate(['B1', 'M1']):
            tp, fp, fn = [values[k][i] for k in ['TP', 'FP', 'FN']]
            diagnostic.append({'experiment': name, 'class': cls, 'tp': tp, 'fp': fp, 'fn': fn,
                               'precision': tp/(tp+fp), 'recall': tp/(tp+fn)})
    with (out/'diagnostic.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(diagnostic[0])); writer.writeheader(); writer.writerows(diagnostic)
    model_dir = ROOT/'results/imported_M1'; model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir/'summary.json').write_text(json.dumps({**m1, 'epochs': 100, 'seed': 0,
        'split': 'val', 'evaluator': 'YOLOv5 v7.0', 'source': Path(notebook_file).name,
        'cells': [30, 31, 32]}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    tex = [r'\begin{tabular}{lrrr}', r'\toprule', r'Métrica & B1 & M1 & Cambio relativo \\', r'\midrule']
    for key, label in [('precision', 'Precision'), ('recall', 'Recall'), ('map50', 'mAP@0.5'),
                       ('map50_95', 'mAP@0.5:0.95'), ('fitness', 'Fitness')]:
        delta = 100*(m1[key]/b1[key]-1)
        tex.append(f'{label} & {b1[key]:.3f} & {m1[key]:.3f} & {delta:+.1f}'+r'\% \\')
    tex.extend([r'\bottomrule', r'\end{tabular}'])
    (out/'comparison.tex').write_text('\n'.join(tex)+'\n', encoding='utf-8', newline='\n')
    if figures:
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as cache:
            os.environ.setdefault('MPLCONFIGDIR', cache)
            os.environ.setdefault('XDG_CACHE_HOME', cache)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(7, 3), layout='constrained')
            axes[0].bar(['B1 · 640', 'M1 · 1280'], [b1['map50_95'], m1['map50_95']], color=['#21618c', '#d68910'])
            axes[0].set(ylabel='mAP@0.5:0.95', ylim=(0, 1), title='Validación · global')
            guns = {r['experiment']: r['fn'] for r in diagnostic if r['class']=='gun'}
            axes[1].bar(['B1 · 640', 'M1 · 1280'], [guns['B1'], guns['M1']], color=['#21618c', '#d68910'])
            axes[1].set(ylabel='Armas omitidas', title='Diagnóstico · confianza 0,001', ylim=(0, 180))
            for ax in axes:
                ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
            fig.savefig(out/'comparison.png', dpi=200); fig.savefig(out/'comparison.pdf'); plt.close(fig)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--notebook', required=True)
    p.add_argument('--b1-minutes', type=float)
    p.add_argument('--figures', action='store_true')
    a = p.parse_args()
    print(export(a.notebook, a.b1_minutes, a.figures))
