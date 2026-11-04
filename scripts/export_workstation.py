"""Guarda las tablas de validación existentes; no ejecuta entrenamiento ni inferencia."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', nargs='+', choices=['B1', 'M1', 'M2'], default=['B1', 'M1', 'M2'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    from src.common import paths
    from src.exports import export_workstation, output_folder
    settings = paths()
    rows = export_workstation(settings, args.output, args.experiments, args.seeds, strict=True)
    out = args.output or output_folder(settings['runs'])
    print(f'Exportadas {len(rows)} filas de validación YOLOv5.')
    print(f'Tablas guardadas en: {out}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f'ERROR: {error}')
