"""Figuras de análisis de errores del artículo.

Genera en results/paper_figures/, a partir de las imágenes de Validation de la
carpeta de datos de configs/paths.local.json:
  armas_omitidas.png    armas pequeñas (área <= 1 %) que M2 no detectó y que no
                        tienen ninguna predicción solapada; la de menor área de
                        cada video (results/workstation/grupo/diagnostico_gun_objetos.csv).
  falsos_positivos.png  los cuatro falsos positivos de gun de M2 con mayor confianza,
                        de videos distintos, causados por IoU < 0,50 con la anotación
                        (results/workstation/grupo/revision_localizacion.csv).
La figura de semillas la genera scripts/seed_summary.py.
"""
from pathlib import Path
import argparse
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GREEN = (0, 170, 60)
RED = (220, 30, 30)


def missed_small_guns(diagnostic, experiment='M2', max_area=0.01, count=6):
    """Armas pequeñas sin candidato, ordenadas por área, sin repetir video."""
    with Path(diagnostic).open(encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['experiment'] == experiment and r['type'] == 'FN'
                and r['category'] == 'sin_candidato' and float(r['area_relative_gt']) <= max_area]
    chosen, videos = [], set()
    for r in sorted(rows, key=lambda r: float(r['area_relative_gt'])):
        if r['video'] not in videos:
            videos.add(r['video'])
            chosen.append(r)
        if len(chosen) == count:
            break
    return chosen


def localization_errors(review, experiment='M2', count=4):
    """Falsos positivos por localización de mayor confianza, sin repetir video."""
    with Path(review).open(encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['experiment'] == experiment]
    chosen, videos = [], set()
    for r in sorted(rows, key=lambda r: -float(r['confidence'])):
        video = r['image'].rsplit('_', 3)[0]
        if video not in videos:
            videos.add(video)
            chosen.append(r)
        if len(chosen) == count:
            break
    return chosen


def crop(width, height, box, context=4, minimum=180):
    """Recorte cuadrado centrado en la caja, de lado `context` veces su lado mayor."""
    x1, y1, x2, y2 = box
    side = max(minimum, context * max(x2 - x1, y2 - y1))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(0, cx - side / 2), max(0, width - side))
    top = min(max(0, cy - side / 2), max(0, height - side))
    return (round(left), round(top), round(min(width, left + side)), round(min(height, top + side)))


def _pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def missed_guns_figure(rows, images_dir, path, tile=300):
    from PIL import Image, ImageDraw
    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(3.5, 2.6), dpi=300)
    for ax, r in zip(axes.flat, rows):
        image = Image.open(Path(images_dir) / r['image']).convert('RGB')
        box = [float(r[k]) for k in ('x1', 'y1', 'x2', 'y2')]
        region = crop(image.width, image.height, box)
        piece = image.crop(region)
        scale = tile / max(piece.size)
        piece = piece.resize((round(piece.width * scale), round(piece.height * scale)))
        shifted = [(box[0] - region[0]) * scale, (box[1] - region[1]) * scale,
                   (box[2] - region[0]) * scale, (box[3] - region[1]) * scale]
        ImageDraw.Draw(piece).rectangle(shifted, outline=GREEN, width=3)
        ax.imshow(piece)
        ax.set_title(f"{float(r['area_relative_gt']) * 100:.2f} % del área", fontsize=6, pad=2)
        ax.axis('off')
    for ax in list(axes.flat)[len(rows):]:
        ax.axis('off')
    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.6)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def localization_figure(rows, images_dir, path):
    from PIL import Image, ImageDraw
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 2.3), dpi=300)
    for ax, r in zip(axes.flat, rows):
        image = Image.open(Path(images_dir) / r['image']).convert('RGB')
        draw = ImageDraw.Draw(image)
        width = max(3, image.width // 250)
        draw.rectangle([float(r[k]) for k in ('gt_x1', 'gt_y1', 'gt_x2', 'gt_y2')], outline=GREEN, width=width)
        draw.rectangle([float(r[k]) for k in ('pred_x1', 'pred_y1', 'pred_x2', 'pred_y2')], outline=RED, width=width)
        ax.imshow(image)
        ax.set_title(f"conf. {float(r['confidence']):.2f} · IoU {float(r['max_iou_gt']):.2f}", fontsize=6, pad=2)
        ax.axis('off')
    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.6)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/paper_figures')
    parser.add_argument('--images', type=Path, help='imágenes de Validation (por defecto, las de paths.local.json)')
    args = parser.parse_args()
    if args.images is None:
        from src.common import paths
        args.images = Path(paths()['dataset']) / 'images/val'
    args.output.mkdir(parents=True, exist_ok=True)
    rows = missed_small_guns(ROOT / 'results/workstation/grupo/diagnostico_gun_objetos.csv')
    missed_guns_figure(rows, args.images, args.output / 'armas_omitidas.png')
    with (args.output / 'armas_omitidas.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'video', 'area_relative_gt', 'x1', 'y1', 'x2', 'y2'],
                                extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    errors = localization_errors(ROOT / 'results/workstation/grupo/revision_localizacion.csv')
    localization_figure(errors, args.images, args.output / 'falsos_positivos.png')
    print(f'Figuras guardadas en: {args.output}')


if __name__ == '__main__':
    main()
