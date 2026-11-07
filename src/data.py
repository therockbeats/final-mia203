<<<<<<< HEAD
"""Extracción de datos y comprobación de imágenes, etiquetas y particiones."""

from collections import Counter
=======
# Extracción de datos y comprobación de imágenes, etiquetas y particiones.
from collections import Counter, defaultdict
>>>>>>> 929942ae2aa3bc6dda9d0928186d0cd4478c8ea5
from pathlib import Path
from zipfile import ZipFile
import math
import re
import shutil
from .common import sha256, write_json, now

EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def images(root, split):
    """Carga las imágenes de un split específico."""
    return sorted(
        p for p in (Path(root) / 'images' / split).iterdir()
        if p.suffix.lower() in EXTENSIONS
    )


def labels(root, split, stem):
    """Lee y valida las etiquetas en formato YOLO."""
    p = Path(root) / 'labels' / split / (stem + '.txt')
    out = []
    for line in p.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        v = [float(x) for x in line.split()]

        # PARÁMETRO 1: 5 valores (clase, x, y, w, h)
        if len(v) != 5 or not all(math.isfinite(x) for x in v):
            raise ValueError(f'Etiqueta inválida: {p}')
        c, x, y, w, h = v

        # PARÁMETRO 2: Clases 0 y 1. PARÁMETRO 3: Coordenadas normales
        if c not in (0, 1) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1):
            raise ValueError(f'Clase/coordenadas inválidas: {p}')

        # PARÁMETRO 4: Validación de bordes
        if x - w/2 < -0.002 or y - h/2 < -0.002:
            raise ValueError(f'Caja fuera de imagen: {p}')
        if x + w/2 > 1.002 or y + h/2 > 1.002:
            raise ValueError(f'Caja fuera de imagen: {p}')

        out.append([int(c), x, y, w, h])
    return out

<<<<<<< HEAD

=======
# Descomprime el dataset y aplica medidas de seguridad para evitar vulnerabilidades (Zip Slip).
# Extrae en una carpeta temporal y luego la renombra para evitar datos corruptos.
>>>>>>> 929942ae2aa3bc6dda9d0928186d0cd4478c8ea5
def extract(archive, target):
    """Descomprime el dataset previniendo ataques Zip Slip."""
    target = Path(target).resolve()
    if target.exists():
        raise FileExistsError(f'La extracción requiere un destino nuevo: {target}')
    with ZipFile(archive) as z:
        for item in z.infolist():
            dest = (target / item.filename).resolve()
            if not dest.is_relative_to(target):
                raise ValueError('Entrada ZIP insegura')

        tmp = target.with_name(target.name + '.extracting')
        if tmp.exists():
            raise FileExistsError(tmp)
        tmp.mkdir(parents=True)
        try:
            z.extractall(tmp)
            source = tmp / 'processed' if (tmp / 'processed').is_dir() else tmp
            source.rename(target)
            if tmp.exists():
                tmp.rmdir()
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise


def audit(root, output):
    """Audita el dataset para prevenir data leakage."""
    root = Path(root)
    result = {'created_at': now(), 'splits': {}, 'overlap': {}, 'manifest': []}
    videos, hashes, stems = {}, {}, {}

    for split in ('train', 'val', 'test'):
        ims = images(root, split)
        labs = sorted((root / 'labels' / split).glob('*.txt'))
        image_stems = [p.stem for p in ims]

        if len(image_stems) != len(set(image_stems)):
            raise ValueError(f'Correspondencia imagen/label inválida: {split}')
        if set(image_stems) != {p.stem for p in labs}:
            raise ValueError(f'Correspondencia imagen/label inválida: {split}')

        videos[split], hashes[split], stems[split] = set(), set(), set(image_stems)
        counts = Counter()

        for p in ims:
            if not re.fullmatch(r'.+_\d+_\d+_\d+', p.stem):
                raise ValueError(f'ID de video inválido: {p.name}')
            video = p.stem.rsplit('_', 3)[0]
            videos[split].add(video)

            boxes = labels(root, split, p.stem)
            counts.update(str(b[0]) for b in boxes)
            counts['empty_labels'] += not boxes
            counts['without_gun'] += not any(b[0] == 1 for b in boxes)

            digest = sha256(p)
            hashes[split].add(digest)
            
            label_path = root / 'labels' / split / (p.stem + '.txt')
            result['manifest'].append({
                'split': split, 'image': p.name, 'video': video,
                'image_sha256': digest, 'label_sha256': sha256(label_path)
            })
            
        result['splits'][split] = {
            'images': len(ims), 'labels': len(labs), 
            'videos': len(videos[split]), **counts
        }

    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        overlap = {
            'videos': sorted(videos[a] & videos[b]),
            'stems': sorted(stems[a] & stems[b]),
            'identical_images': sorted(hashes[a] & hashes[b])
        }
        result['overlap'][a + ':' + b] = overlap

    result['valid'] = not any(any(v.values()) for v in result['overlap'].values())
    write_json(output, result)

    if not result['valid']:
        raise ValueError('Fuga entre particiones: consultar auditoría')
    return result
<<<<<<< HEAD


=======
    #  Comprueba que los datos no hayan sido alterados después de la auditoría.
    #  Usa los hashes SHA256 guardados para garantizar reproducibilidad.
>>>>>>> 929942ae2aa3bc6dda9d0928186d0cd4478c8ea5
def verify_manifest(root, audit_file):
    """Verifica que los datos no hayan sido alterados."""
    from .common import read_json
    root = Path(root)
    result = read_json(audit_file)
    
    if not result['valid']:
        raise ValueError('Auditoría inválida')

    for split in ('train', 'val', 'test'):
        expected = {r['image'] for r in result['manifest'] if r['split'] == split}
        if expected != {p.name for p in images(root, split)}:
            raise ValueError(f'Inventario cambiado: {split}')
        if {Path(n).stem for n in expected} != {p.stem for p in (root/'labels'/split).glob('*.txt')}:
            raise ValueError(f'Inventario de etiquetas cambiado: {split}')

    for row in result['manifest']:
        image = root / 'images' / row['split'] / row['image']
        label = root / 'labels' / row['split'] / (image.stem + '.txt')
        if sha256(image) != row['image_sha256'] or sha256(label) != row['label_sha256']:
<<<<<<< HEAD
            raise ValueError(f'Datos cambiados: {image.name}')
            
    return result
=======
            raise ValueError(f'Datos cambiados después de auditar: {image.name}')
    return result
>>>>>>> 929942ae2aa3bc6dda9d0928186d0cd4478c8ea5
