"""Extracción de datos y comprobación de imágenes, etiquetas y particiones.
Garantiza la integridad del dataset y previene la fuga de datos (data leakage).
"""

from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile
import math
import re
import shutil
from .common import sha256, write_json, now

# ---- Formatos de imagen aceptados para el entrenamiento ----
EXTENSIONS = {'.jpg', '.jpeg', '.png'}


# ---- 1. CARGA DE IMÁGENES ----
def images(root, split):
    return sorted(p for p in (Path(root) / 'images' / split).iterdir() if p.suffix.lower() in EXTENSIONS)


# ---- 2. VALIDACIÓN ESTRICTA DE ETIQUETAS (YOLO FORMAT) ----
def labels(root, split, stem):
    """
    Lee y valida el archivo .txt de etiquetas correspondiente a una imagen.
    AQUÍ SE DEFINEN LOS PARÁMETROS GEOMÉTRICOS Y DE CLASE PARA EL ENTRENAMIENTO.
    """
    p = Path(root) / 'labels' / split / (stem + '.txt')
    out = []
    for line in p.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        v = [float(x) for x in line.split()]

        # PARÁMETRO 1: Formato de etiqueta. Debe tener 5 valores (clase, x, y, w, h)
        if len(v) != 5 or not all(math.isfinite(x) for x in v):
            raise ValueError(f'Etiqueta inválida: {p}')
        c, x, y, w, h = v
        
        # PARÁMETRO 2: Clases permitidas. Solo 0 (person) y 1 (gun).
        # PARÁMETRO 3: Coordenadas normalizadas (deben estar entre 0 y 1).
        if c not in (0, 1) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f'Clase/coordenadas inválidas: {p}')

        # PARÁMETRO 4: Validación de bordes de la caja.
        # Se permite un margen de error mínimo (-0.002 a 1.002) por redondeo.
        if x - w/2 < -0.002 or y - h/2 < -0.002 or x + w/2 > 1.002 or y + h/2 > 1.002:
            raise ValueError(f'Caja fuera de imagen: {p}')
        out.append([int(c), x, y, w, h])
    return out


# ---- 3. EXTRACCIÓN SEGURA DE DATOS ----
def extract(archive, target):
    target = Path(target).resolve()
    if target.exists():
        raise FileExistsError(f'La extracción requiere un destino nuevo: {target}')
    with ZipFile(archive) as z:
        for item in z.infolist():
            dest = (target / item.filename).resolve()
            if not dest.is_relative_to(target) or (item.external_attr >> 16) & 0o170000 == 0o120000:
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


# ---- 4. AUDITORÍA DEL DATASET (PREVENCIÓN DE DATA LEAKAGE) ----
def audit(root, output):
    root = Path(root)
    result = {'created_at': now(), 'splits': {}, 'overlap': {}, 'manifest': []}
    videos, hashes, stems = {}, {}, {}
    
    for split in ('train', 'val', 'test'):
        ims = images(root, split)
        labs = sorted((root / 'labels' / split).glob('*.txt'))
        image_stems = [p.stem for p in ims]
        
        if len(image_stems) != len(set(image_stems)) or set(image_stems) != {p.stem for p in labs}:
            raise ValueError(f'Correspondencia imagen/label inválida: {split}')
            
        videos[split], hashes[split], stems[split] = set(), set(), set(image_stems)
        counts = Counter()

        # Formato esperado: "nombreVideo_0001_0002.jpg" -> Identifica el video "nombreVideo"
        for p in ims:
            if not re.fullmatch(r'.+_\d+_\d+_\d+', p.stem):
                raise ValueError(f'No se puede obtener el identificador de video: {p.name}')
            video = p.stem.rsplit('_', 3)[0]
            videos[split].add(video)

            # Cuenta cuántas cajas hay de cada clase (person, gun)
            boxes = labels(root, split, p.stem)
            counts.update(str(b[0]) for b in boxes)

            # Cuenta imágenes sin cajas (negativos) y sin armas
            counts['empty_labels'] += not boxes
            counts['without_gun'] += not any(b[0] == 1 for b in boxes)
            
            digest = sha256(p)
            hashes[split].add(digest)
            result['manifest'].append({
                'split': split, 'image': p.name, 'video': video,
                'image_sha256': digest, 'label_sha256': sha256(root / 'labels' / split / (p.stem + '.txt'))
            })
            
        result['splits'][split] = {'images': len(ims), 'labels': len(labs), 'videos': len(videos[split]), **counts}
        
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


# ---- 5. VERIFICACIÓN DE MANIFIESTO ----
def verify_manifest(root, audit_file):
    from .common import read_json
    root = Path(root)
    result = read_json(audit_file)
    
    if not result['valid']:
        raise ValueError('Auditoría inválida')
        
    for split in ('train', 'val', 'test'):
        expected = {r['image'] for r in result['manifest'] if r['split'] == split}
        if expected != {p.name for p in images(root, split)}:
            raise ValueError(f'Inventario cambiado después de auditar: {split}')
        if {Path(n).stem for n in expected} != {p.stem for p in (root/'labels'/split).glob('*.txt')}:
            raise ValueError(f'Inventario de etiquetas cambiado: {split}')

    # Verificación criptográfica de que cada imagen y etiqueta es exactamente igual a la auditada.
    for row in result['manifest']:
        image = root/'images'/row['split']/row['image']
        label = root/'labels'/row['split']/(image.stem+'.txt')
        if sha256(image) != row['image_sha256'] or sha256(label) != row['label_sha256']:
            raise ValueError(f'Datos cambiados después de auditar: {image.name}')
            
    return result
