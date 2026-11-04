"""Comprueba la estructura del repositorio y los archivos de entrega."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def check(final=False):
    errors = []
    required = ['README.md','requirements.txt','data/README.md','weights/README.md',
                'src/pipeline.py','configs/B1.json','configs/M1.json','configs/M2.json',
                'notebooks/01_experimentos_colab.ipynb','notebooks/02_demo_colab.ipynb']
    for name in required:
        if not (ROOT/name).is_file(): errors.append(f'Falta {name}')
    for file in (ROOT/'configs').glob('*.json'):
        json.loads(file.read_text(encoding='utf-8'))
    for file in (ROOT/'notebooks').glob('*.ipynb'):
        notebook = json.loads(file.read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                if cell.get('outputs') or cell.get('execution_count') is not None:
                    errors.append(f'Limpiar salidas antes de publicar: {file.name}')
    if (ROOT/'.git').exists():
        tracked = subprocess.check_output(['git','ls-files','--cached'],cwd=ROOT,text=True, encoding='utf-8', errors='replace').splitlines()
        for name in tracked:
            path = ROOT/name
            if not path.is_file(): continue
            if name.endswith(('.pt','.pth','.ckpt','.zip','.pem','.key')) or name=='configs/paths.local.json':
                errors.append(f'Archivo privado/pesado en el índice: {name}')
            if path.suffix in ['.py','.json','.md','.ipynb','.tex','.yml']:
                text = path.read_text(errors='replace', encoding='utf-8')
                import re
                if re.search(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})', text):
                    errors.append(f'Posible credencial: {name}')
    if final:
        required_final = ['configs/final_selection.json', 'results/final_summary/validation_experiments.csv',
            'results/final_summary/final_test_metrics.csv', 'presentation/Equipo4_Slides.pdf', 'DELIVERY.json']
        for name in required_final:
            if not (ROOT/name).exists(): errors.append(f'Entrega pendiente: {name}')
        for name in ['README.md','data/README.md','weights/README.md','presentation/slides.tex','DELIVERY.json']:
            if (ROOT/name).exists() and 'PENDIENTE' in (ROOT/name).read_text(errors='replace', encoding='utf-8'):
                errors.append(f'Completar contenido: {name}')
        if (ROOT/'DELIVERY.json').exists():
            d = json.loads((ROOT/'DELIVERY.json').read_text(encoding='utf-8'))
            for key in ['repository_url','dataset_url','weights_url','contributors']:
                if not d.get(key): errors.append(f'Completar DELIVERY.json: {key}')
            if d.get('tag') != 'v1.0-final': errors.append('Tag final incorrecto')
        selection = ROOT/'configs/final_selection.json'
        validation = ROOT/'results/final_summary/validation_experiments.csv'
        testing = ROOT/'results/final_summary/final_test_metrics.csv'
        if selection.exists() and validation.exists() and testing.exists():
            import csv
            selected = json.loads(selection.read_text(encoding='utf-8'))['experiment']
            with validation.open(encoding='utf-8') as f: rows = list(csv.DictReader(f))
            if {r['experiment'] for r in rows} != {'B1','M1','M2'}:
                errors.append('La comparación de validación requiere B1, M1 y M2')
            with testing.open(encoding='utf-8') as f: rows = list(csv.DictReader(f))
            if {r['class'] for r in rows} != {'all','person','gun'} or any(r['experiment'] != selected for r in rows):
                errors.append('Las métricas TEST deben corresponder al modelo seleccionado')
    if errors:
        print('\n'.join(errors)); return 1
    print('Comprobación correcta' + (' de entrega final' if final else ' de estructura y publicación'))
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true')
    sys.exit(check(p.parse_args().final))
