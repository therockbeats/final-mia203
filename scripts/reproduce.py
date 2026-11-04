"""Ejecuta entrenamiento, evaluación o generación de informes."""
import argparse
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
p = argparse.ArgumentParser()
p.add_argument('phase', choices=['train','validation','test','report'])
p.add_argument('--seeds', type=int, nargs='+', default=[0])
p.add_argument('--experiments', nargs='+', choices=['B1','M1','M2'], default=None)
p.add_argument('--split', choices=['val','test'], default='val')
a = p.parse_args()
a.experiments = a.experiments or (['M2'] if a.phase == 'test' else ['B1','M1','M2'])


def command(*args):
    subprocess.run([sys.executable,'-m','src.pipeline',*map(str,args)],cwd=root,check=True)


for seed in a.seeds:
    if a.phase=='train':
        for name in a.experiments:
            command('train',name,'--seed',seed)
    if a.phase in ['validation','test']:
        split = 'val' if a.phase=='validation' else 'test'
        from_config = root/'configs/paths.local.json'
        import json
        runs = Path(json.loads(from_config.read_text(encoding='utf-8'))['runs'])
        for name in a.experiments:
            pred = runs/name/f'seed{seed}'/f'{split}_predictions/full/predictions.json'
            if not pred.exists(): command('predict',name,'--split',split,'--seed',seed)
            command('evaluate',name,'--split',split,'--seed',seed)
if a.phase in ['validation','test','report']:
    split = 'val' if a.phase=='validation' else ('test' if a.phase=='test' else a.split)
    command('figures','--split',split,'--experiments',*a.experiments)
    for seed in a.seeds:
        for name in a.experiments: command('errors',name,'--split',split,'--seed',seed)

    if split == 'val':
        from src.common import paths
        from src.exports import export_workstation, output_folder
        settings = paths()
        export_workstation(settings, experiments=a.experiments, seeds=a.seeds)
        print(f"Tablas, figuras y resumen guardados en: {output_folder(settings['runs'])}")
