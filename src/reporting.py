from pathlib import Path
import csv
import statistics
import re
import shutil
from .common import ROOT, paths, read_json, write_json, sha256, now


def import_experiment(source, experiment='B1'):
    if experiment not in {'B1', 'M1', 'M2'}:
        raise ValueError('El experimento debe ser B1, M1 o M2')
    p = paths()
    source = Path(source)
    folder = {'B1': 'B1_YOLOv5s', 'M1': 'M1_YOLOv5s_1280', 'M2': 'M2_YOLOv5s_MultiScale'}[experiment]
    run = source/'results'/folder/'run'
    target = p['runs']/experiment/'seed0/run'
    if target.exists():
        raise FileExistsError(f'{experiment} ya existe en el destino')
    with (run/'results.csv').open(encoding='utf-8') as f:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]
    if [int(r['epoch']) for r in rows] != list(range(100)):
        raise ValueError(f'{experiment} debe contener 100 épocas consecutivas')
    if not all((run/'weights'/w).is_file() for w in ['best.pt', 'last.pt']):
        raise ValueError('Faltan checkpoints')
    log_file = source/'results'/folder/f'{experiment}_training.log'
    log = log_file.read_text(errors='replace', encoding='utf-8')
    options = run/'opt.yaml'
    if options.exists():
        import yaml
        actual = yaml.safe_load(options.read_text(encoding='utf-8'))
        expected = {'imgsz': 1280 if experiment == 'M1' else 640, 'multi_scale': experiment == 'M2',
                    'batch_size': 16, 'epochs': 100, 'optimizer': 'SGD', 'seed': 0}
        if any(actual.get(k) != v for k, v in expected.items()):
            raise ValueError('La configuración de origen difiere del experimento solicitado')
    selected_file = ROOT/'configs/final_selection.json'
    if experiment == 'M2' and selected_file.exists():
        selected = read_json(selected_file)
        if sha256(run/'weights/best.pt') != selected['expected_weights_sha256']:
            raise ValueError('Los pesos M2 difieren del checkpoint seleccionado por el grupo')
    shutil.copytree(run, target)
    public = ROOT/'results'/f'imported_{experiment}'
    public.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run/'results.csv', public/'results.csv')
    for image in run.glob('*.png'):
        shutil.copy2(image, public/image.name)
    hours = re.search(r'100 epochs completed in ([\d.]+) hours', log)
    training_hours = float(hours[1]) if hours else None
    last = log[log.rfind('Validating '):]
    last = re.sub(r'\x1b\[[0-9;]*m', '', last)
    summary = []
    for line in last.splitlines():
        if re.match(r'\s+(all|person|gun)\s+494\s+', line):
            parts = line.split()
            summary.append(dict(zip(['class', 'images', 'instances', 'precision', 'recall', 'map50', 'map50_95'], parts)))
    best = max(rows, key=lambda r: .1*float(r['metrics/mAP_0.5']) + .9*float(r['metrics/mAP_0.5:0.95']))
    metadata = {'status': 'imported_completed', 'epochs': len(rows), 'seed': 0,
        'training_config': {'imgsz': 1280 if experiment == 'M1' else 640, 'multi_scale': experiment == 'M2', 'batch_size': 16, 'epochs': 100, 'optimizer': 'SGD'},
        'hardware': 'NVIDIA A100-SXM4-40GB', 'training_hours_log': training_hours,
        'best_csv_epoch_zero_based': int(best['epoch']), 'official_validation_log': summary,
        'split': 'val', 'evaluator': 'YOLOv5 v7.0 val.py',
        'sha256': {str(x.relative_to(source)): sha256(x) for x in [run/'results.csv', run/'weights/best.pt', run/'weights/last.pt', log_file]}}
    summary_metrics = metadata['official_validation_log']
    summary = {k: metadata[k] for k in ['epochs', 'seed', 'training_config', 'hardware', 'best_csv_epoch_zero_based']}
    summary.update(experiment=experiment, split='val', evaluator=metadata['evaluator'],
                   training_hours=round(training_hours, 2) if training_hours is not None else None, metrics=summary_metrics)
    write_json(public/'summary.json', summary)
    write_json(target.parent/'imported.provenance.json', metadata)
    return metadata


import_baseline = import_experiment


def compare(split, experiments=('B1', 'M1', 'M2')):
    p = paths()
    rows = []
    for name in experiments:
        for metric_path in sorted((p['runs']/name).glob(f'seed*/{split}_predictions/full/metrics.json')):
            m = read_json(metric_path)
            rows.append({'experiment': name, 'seed': m['seed'], 'split': split,
                **m['macro'], 'gun_map50': m['per_class']['gun']['map50'],
                'gun_map50_95': m['per_class']['gun']['map50_95'], 'gun_recall': m['per_class']['gun']['recall'],
                'small_gun_recall': m['small_gun']['recall'], 'mean_image_seconds': m['mean_image_seconds'],
                'peak_allocated_mb': m['peak_allocated_mb'], 'evaluator': m['evaluator'],
                'dataset_sha256': m['signature']['dataset_sha256']})
    if not rows:
        raise ValueError('No hay métricas formales para comparar')
    if len({r['dataset_sha256'] for r in rows}) != 1 or len({r['evaluator'] for r in rows}) != 1:
        raise ValueError('Dataset o evaluador incompatibles')
    out = ROOT/'results'/split
    out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row.pop('dataset_sha256')
    with (out/'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    class_rows = []
    for name in experiments:
        for file in sorted((p['runs']/name).glob(f'seed*/{split}_predictions/full/metrics.json')):
            m = read_json(file)
            for cls, values in m['per_class'].items():
                class_rows.append({'experiment': name, 'seed': m['seed'], 'split': split, 'class': cls, **values})
    with (out/'per_class.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(class_rows[0])); writer.writeheader(); writer.writerows(class_rows)
    grouped = []
    for name in experiments:
        subset = [r for r in rows if r['experiment'] == name]
        if not subset:
            continue
        g = {'experiment': name, 'n_seeds': len(subset)}
        for key in ['map50', 'map50_95', 'gun_map50', 'gun_recall', 'small_gun_recall', 'mean_image_seconds']:
            values = [r[key] for r in subset if r[key] is not None]
            g[key] = {'mean': statistics.mean(values) if values else None,
                      'sd': statistics.stdev(values) if len(values) > 1 else None}
        grouped.append(g)
    write_json(out/'comparison_summary.json', grouped)
    tex = ['% Tabla comparativa por experimento y semilla.',
           r'\begin{tabular}{lrrrr}', r'\hline', r'Experimento & Semillas & mAP50 & mAP50:95 & Recall gun \\', r'\hline']
    for g in grouped:
        def val(k):
            v = g[k]
            if v['mean'] is None: return '--'
            return f"{v['mean']:.3f}" + (f" $\\pm$ {v['sd']:.3f}" if v['sd'] is not None else '')
        tex.append(f"{g['experiment']} & {g['n_seeds']} & {val('map50')} & {val('map50_95')} & {val('gun_recall')} " + r'\\')
    tex += [r'\hline', r'\end{tabular}']
    (out/'comparison.tex').write_text('\n'.join(tex)+'\n', encoding='utf-8', newline='\n')
    return rows


def figures(split, experiments=('B1', 'M1', 'M2')):
    from .runtime import configure_cache
    configure_cache(paths())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = paths()
    out = ROOT/'results'/split
    rows = compare(split, experiments)
    for metric, label in [('gun_map50_95', 'AP gun @ IoU 0.50:0.95'), ('small_gun_recall', 'Recall gun, área relativa ≤ 1 %')]:
        fig, ax = plt.subplots(figsize=(6, 3.8), layout='constrained')
        for r in rows:
            if r[metric] is not None:
                ax.scatter(r['experiment'], r[metric], label=f"{r['experiment']} semilla {r['seed']}")
        ax.set(xlabel='Experimento', ylabel=label, ylim=(0, 1)); ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=7)
        fig.savefig(out/f'{metric}.png', dpi=200); fig.savefig(out/f'{metric}.pdf'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.8), layout='constrained')
    for name in experiments:
        for file in sorted((p['runs']/name).glob(f'seed*/{split}_predictions/full/pr_curve.json')):
            points = [r for r in read_json(file) if r['class'] == 'gun']
            ax.step([r['recall'] for r in points], [r['precision'] for r in points], where='post', label=f'{name} {file.parents[2].name}')
    ax.set(xlabel='Recall gun', ylabel='Precision gun', xlim=(0,1), ylim=(0,1)); ax.legend(); ax.grid(alpha=.2)
    fig.savefig(out/'gun_pr.png', dpi=200); fig.savefig(out/'gun_pr.pdf'); plt.close(fig)


def error_examples(name, split, seed):
    from PIL import Image, ImageDraw
    p = paths()
    folder = p['runs']/name/f'seed{seed}'/f'{split}_predictions/full'
    errors = read_json(folder/'errors.json')
    out = ROOT/'results'/split/'error_examples'/name/f'seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    # Deterministic selection: six smallest omitted guns, then six most confident FP.
    fn = sorted([x for x in errors if x['type']=='FN' and x['class_id']==1], key=lambda x:x['area_relative'])[:6]
    fp = sorted([x for x in errors if x['type']=='FP' and x['class_id']==1], key=lambda x:-x['confidence'])[:6]
    selection = fn+fp
    for i, row in enumerate(selection):
        im = Image.open(p['dataset']/'images'/split/row['image']).convert('RGB')
        draw = ImageDraw.Draw(im)
        draw.rectangle(row['box'], outline='red', width=3)
        draw.text((5,5), f"{row['type']} gun | {row['image']}", fill='red')
        im.thumbnail((1000,1000)); im.save(out/f'{i+1:02d}_{row["type"]}.jpg')
    write_json(out/'selection.json', selection)
