"""Empaqueta la presentación y los resultados del repositorio."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, 'scripts/check_repository.py', '--final'], cwd=root, check=True)
out = root/'results/delivery'
out.mkdir(parents=True, exist_ok=True)
with ZipFile(out/'Equipo4_Repositorio_Resultados.zip', 'w', ZIP_DEFLATED) as z:
    for name in ['README.md', 'DELIVERY.json', 'presentation/slides.tex', 'presentation/Equipo4_Slides.pdf']:
        file = root/name
        if file.is_file(): z.write(file, file.relative_to(root))
    for file in sorted((root/'results').rglob('*')):
        if file.is_file() and not file.is_relative_to(out) and file.suffix in {'.tex', '.csv', '.json', '.png', '.pdf'}:
            z.write(file, file.relative_to(root))
print(out/'Equipo4_Repositorio_Resultados.zip')
