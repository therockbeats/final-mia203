"""Instala dependencias del proyecto conservando PyTorch y CUDA de Colab."""
from pathlib import Path
import subprocess
import sys
import tempfile


def supplemental_requirements(text):
    import re
    lines = []
    for line in text.splitlines():
        stripped = line.split('#', 1)[0].strip()
        if not stripped:
            continue
        name = re.split(r'[<>=!~\[;\s]', stripped, maxsplit=1)[0].lower()
        if name not in {'torch', 'torchvision'}:
            lines.append(stripped)
    return '\n'.join(lines) + '\n'


def main():
    import torch
    import torchvision
    before = (torch.__version__, torchvision.__version__, torch.version.cuda)
    requirements = Path(__file__).resolve().parents[1]/'requirements.txt'
    with tempfile.TemporaryDirectory() as folder:
        extra = Path(folder)/'colab.txt'
        extra.write_text(supplemental_requirements(requirements.read_text()))
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(extra)], check=True)
    probe = subprocess.check_output([sys.executable, '-c',
        'import json,torch,torchvision;print(json.dumps([torch.__version__,torchvision.__version__,torch.version.cuda]))'], text=True)
    import json
    if tuple(json.loads(probe)) != before:
        raise RuntimeError('La instalación cambió PyTorch, torchvision o CUDA; revise las dependencias')
    print(f'Colab: PyTorch {before[0]} · torchvision {before[1]} · CUDA {before[2]}')


if __name__ == '__main__':
    main()
