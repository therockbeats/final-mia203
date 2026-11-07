"""Compara B1 y M2 con todas las corridas disponibles (variabilidad entre semillas).

Une la corrida original del grupo (semilla 0, A100) con las réplicas de la
workstation (semillas 0, 1 y 2, RTX 4090). Todas las cifras provienen de la
reevaluación en la workstation: mAP con val.py de YOLOv5 v7.0; FN de gun y recall
de armas pequeñas con el evaluador complementario a confianza 0,25.
No entrena ni infiere: solo lee las tablas de results/workstation/ y genera la
tabla y la figura de semillas del artículo.
"""
from pathlib import Path
from itertools import combinations
from statistics import mean, stdev
import argparse
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
WORKSTATION = ROOT / 'results/workstation'
SOURCES = {'grupo': 'A100', 'replicas': 'RTX 4090'}
METRICS = [('map50', 'mAP@0.5', 3), ('map50_95', 'mAP@0.5:0.95', 3),
           ('gun_map50_95', 'gun mAP@0.5:0.95', 3), ('gun_fn', 'gun FN (conf 0,25)', 1),
           ('small_gun_recall', 'Recall de armas pequeñas', 3)]


def read(path):
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def collect_runs(experiments=('B1', 'M2'), root=WORKSTATION):
    """Una fila por corrida (experimento, semilla, GPU) con las métricas de METRICS."""
    runs = []
    for folder, device in SOURCES.items():
        yolo = {(r['experiment'], r['seed'], r['class']): r for r in read(root / folder / 'validacion_yolov5.csv')}
        per_class = {(r['experiment'], r['seed'], r['class']): r for r in read(root / folder / 'per_class.csv')}
        overall = {(r['experiment'], r['seed']): r for r in read(root / folder / 'comparison.csv')}
        for (name, seed, cls), row in sorted(yolo.items()):
            if cls != 'all' or name not in experiments:
                continue
            runs.append({'experiment': name, 'seed': int(seed), 'device': device,
                         'map50': float(row['map50']), 'map50_95': float(row['map50_95']),
                         'gun_map50_95': float(yolo[name, seed, 'gun']['map50_95']),
                         'gun_fn': int(per_class[name, seed, 'gun']['fn']),
                         'small_gun_recall': float(overall[name, seed]['small_gun_recall'])})
    return runs


def permutation_p(a, b):
    """Prueba de permutación exacta y bilateral para la diferencia de medias.

    Reparte de todas las formas posibles los valores de ambos grupos y cuenta qué
    fracción de esos repartos produce una diferencia al menos tan grande como la
    observada. Con 4 + 4 corridas hay 70 repartos, así que el menor p posible es 2/70.
    """
    values = list(a) + list(b)
    observed = abs(mean(b) - mean(a))
    extreme = total = 0
    for chosen in combinations(range(len(values)), len(a)):
        first = [values[i] for i in chosen]
        second = [values[i] for i in range(len(values)) if i not in chosen]
        total += 1
        extreme += abs(mean(second) - mean(first)) >= observed - 1e-12
    return extreme / total


def summarize(runs, first='B1', second='M2'):
    summary = []
    for key, label, _ in METRICS:
        a = [r[key] for r in runs if r['experiment'] == first]
        b = [r[key] for r in runs if r['experiment'] == second]
        summary.append({'metric': key, 'label': label, 'n_' + first: len(a), 'n_' + second: len(b),
                        first + '_mean': mean(a), first + '_sd': stdev(a),
                        second + '_mean': mean(b), second + '_sd': stdev(b),
                        'difference': mean(b) - mean(a), 'p_permutation': permutation_p(a, b),
                        first + '_range': max(a) - min(a)})
    return summary


def replica_hours(root=WORKSTATION):
    hours = {}
    for path in sorted((root / 'replicas/training').glob('*_sessions.json')):
        sessions = json.loads(path.read_text(encoding='utf-8'))
        hours[path.name.replace('_sessions.json', '')] = sum(s['seconds'] for s in sessions) / 3600
    return hours


def latex_table(runs, summary):
    def number(value, digits):
        return f'{value:.{digits}f}'
    lines = [r'\begin{tabular}{llrrrrr}\toprule',
             r'Exp. & Sem. / GPU & mAP50 & mAP50:95 & gun 50:95 & gun FN & gun peq. \\ \midrule']
    for name in ('B1', 'M2'):
        for r in [x for x in runs if x['experiment'] == name]:
            values = ' & '.join(number(r[k], d) if d > 1 else str(r[k]) for k, _, d in METRICS)
            lines.append(f"{name} & {r['seed']} / {r['device']} & {values} \\\\")
        stats = ' & '.join(number(s[name + '_mean'], d) + r'$\pm$' + number(s[name + '_sd'], d)
                           for s, (_, _, d) in zip(summary, METRICS))
        lines.append(rf'\multicolumn{{2}}{{l}}{{{name}: media $\pm$ DE}} & {stats} \\ \midrule')
    diffs = ' & '.join(f"{s['difference']:+.{d}f}" for s, (_, _, d) in zip(summary, METRICS))
    pvalues = ' & '.join(f"{s['p_permutation']:.2f}" for s in summary)
    lines += [rf'\multicolumn{{2}}{{l}}{{M2 $-$ B1}} & {diffs} \\',
              rf'\multicolumn{{2}}{{l}}{{$p$ (permutación)}} & {pvalues} \\ \bottomrule', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


COLORS = {'B1': '#2a78d6', 'M2': '#eb6834'}
MARKERS = {'A100': 'D', 'RTX 4090': 'o'}


def plot_seeds(runs, path):
    """Un punto por entrenamiento; la barra negra es la media ± DE de cada configuración."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    panels = [('map50_95', 'mAP@0.5:0.95 (Validation)'), ('small_gun_recall', 'Recall de armas pequeñas')]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.1), dpi=300)
    for ax, (key, title) in zip(axes, panels):
        for x, name in enumerate(('B1', 'M2')):
            group = [r for r in runs if r['experiment'] == name]
            for i, r in enumerate(group):
                ax.scatter(x - 0.12 + 0.08 * i, r[key], s=22, marker=MARKERS[r['device']], color=COLORS[name],
                           edgecolor='white', linewidth=0.6, zorder=3)
            values = [r[key] for r in group]
            ax.errorbar(x + 0.27, mean(values), yerr=stdev(values), fmt='_', color='#222222',
                        markersize=9, capsize=2.5, linewidth=1, zorder=2)
        ax.set_xticks([0, 1], ['B1', 'M2'])
        ax.set_xlim(-0.45, 1.6)
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=6, length=2)
        ax.grid(axis='y', color='#dddddd', linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    handles = [Line2D([], [], marker=m, linestyle='', color='#666666', markersize=4, label=gpu)
               for gpu, m in MARKERS.items()]
    handles.append(Line2D([], [], marker='_', linestyle='', color='#222222', markersize=7, label='media ± DE'))
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=6, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.2)
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(Path(path).with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=WORKSTATION / 'semillas')
    args = parser.parse_args()
    runs = collect_runs()
    summary = summarize(runs)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'semillas_corridas.csv', runs)
    write_csv(args.output / 'semillas_resumen.csv', summary)
    hours = replica_hours()
    write_csv(args.output / 'costo_replicas.csv',
              [{'run': k, 'gpu': 'RTX 4090', 'training_hours': round(v, 3)} for k, v in hours.items()])
    (args.output / 'semillas.tex').write_text(latex_table(runs, summary), encoding='utf-8', newline='\n')
    plot_seeds(runs, args.output / 'semillas.png')
    for s in summary:
        print(f"{s['label']:26s} B1 {s['B1_mean']:.4f} ± {s['B1_sd']:.4f}   M2 {s['M2_mean']:.4f} ± {s['M2_sd']:.4f}"
              f"   M2-B1 {s['difference']:+.4f}   p={s['p_permutation']:.2f}")
    print(f'Horas de entrenamiento de las réplicas: {sum(hours.values()):.2f} h')
    print(f'Tablas guardadas en: {args.output}')


if __name__ == '__main__':
    main()
