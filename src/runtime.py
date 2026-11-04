"""Preparación de YOLOv5 y registro verificable del entorno."""
from pathlib import Path
import ast
import importlib.metadata
import json
import platform
import subprocess
import sys
import urllib.request
import urllib.error
import os
from .common import ROOT, YOLO_COMMIT, run, sha256, write_json, read_json, now

PATCH_FILES = ['train.py', 'utils/general.py', 'models/experimental.py', 'utils/plots.py', 'utils/metrics.py']
INITIAL_WEIGHTS_SHA256 = '8b3b748c1e592ddd8868022e8732fde20025197328490623cc16c6f24d0782ee'


def configure_cache(settings):
    base = settings['runs'].parent / 'cache'
    for key, name in [('YOLOV5_CONFIG_DIR', 'yolo'), ('MPLCONFIGDIR', 'matplotlib'), ('XDG_CACHE_HOME', 'xdg')]:
        folder = base/name
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)


def patch_source(text, name):
    if name in PATCH_FILES[:3]:
        tree = ast.parse(text)
        lines = text.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        edits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch' and node.func.attr == 'load':
                if not any(k.arg == 'weights_only' for k in node.keywords):
                    pos = offsets[node.end_lineno-1] + node.end_col_offset - 1
                    edits.append(pos)
        for pos in sorted(edits, reverse=True):
            text = text[:pos] + ', weights_only=False' + text[pos:]
    if name == 'train.py':
        text = text.replace('random.randrange(imgsz * 0.5, imgsz * 1.5 + gs)',
                            'random.randrange(int(imgsz * 0.5), int(imgsz * 1.5 + gs))')
    if name == 'utils/plots.py':
        text = text.replace('self.font.getsize(label)', '(self.font.getbbox(label)[2], self.font.getbbox(label)[3])')
        text = text.replace('self.font.getsize(text)', '(self.font.getbbox(text)[2], self.font.getbbox(text)[3])')
    elif name == 'utils/metrics.py':
        text = text.replace('np.trapz(', 'np.trapezoid(')
    ast.parse(text)
    return text


def prepare(yolo, initial_weights):
    yolo = Path(yolo)
    if not yolo.exists():
        run(['git', 'clone', '--branch', 'v7.0', '--depth', '1', 'https://github.com/ultralytics/yolov5.git', yolo])
    commit = subprocess.check_output(['git', '-C', str(yolo), 'rev-parse', 'HEAD'], text=True, encoding='utf-8', errors='replace').strip()
    if commit != YOLO_COMMIT:
        raise ValueError('Commit YOLOv5 distinto del protocolo')
    manifest = {'commit': commit, 'files': {}}
    for name in PATCH_FILES:
        original = subprocess.check_output(['git', '-C', str(yolo), 'show', f'HEAD:{name}']).decode('utf-8')
        patched = patch_source(original, name)
        p = yolo / name
        legacy = patched.replace('random.randrange(int(imgsz * 0.5), int(imgsz * 1.5 + gs))',
                                 'random.randrange(imgsz * 0.5, imgsz * 1.5 + gs)') if name == 'train.py' else patched
        if p.read_text(encoding='utf-8') not in (original, patched, legacy):
            raise ValueError(f'Hay modificaciones ajenas al protocolo: {p}')
        p.write_text(patched, encoding='utf-8', newline='\n')
        manifest['files'][name] = sha256(p)
    write_json(yolo / 'compatibility_manifest.json', manifest)
    initial_weights = Path(initial_weights)
    if not initial_weights.exists():
        initial_weights.parent.mkdir(parents=True, exist_ok=True)
        tmp = initial_weights.with_suffix('.download')
        url = 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt'
        try:
            urllib.request.urlretrieve(url, tmp)
        except urllib.error.URLError:
            # curl usa el almacén de certificados del sistema en macOS/Windows.
            run(['curl', '--fail', '--location', '--retry', '3', '--output', tmp, url])
        tmp.replace(initial_weights)
    if sha256(initial_weights) != INITIAL_WEIGHTS_SHA256:
        raise ValueError('El checkpoint inicial difiere del oficial verificado v7.0')
    write_json(initial_weights.with_suffix('.provenance.json'), {'source': 'ultralytics/yolov5 release v7.0',
        'sha256': sha256(initial_weights), 'recorded_at': now(), 'note': 'Pesos Pickle: utilizar únicamente fuentes de confianza.'})
    return manifest


def verify_source(yolo):
    manifest = read_json(Path(yolo) / 'compatibility_manifest.json')
    head = subprocess.check_output(['git', '-C', str(yolo), 'rev-parse', 'HEAD'], text=True, encoding='utf-8', errors='replace').strip()
    if head != YOLO_COMMIT:
        raise ValueError('Commit YOLOv5 cambiado')
    if manifest['commit'] != YOLO_COMMIT:
        raise ValueError('Versión incorrecta')
    for name, digest in manifest['files'].items():
        if sha256(Path(yolo) / name) != digest:
            raise ValueError(f'Fuente modificada después de preparar: {name}')
    return manifest


def environment():
    packages = {}
    for name in ['torch', 'torchvision', 'numpy', 'Pillow', 'opencv-python', 'pandas', 'matplotlib', 'PyYAML']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    info = {'python': sys.version, 'platform': platform.platform(), 'packages': packages, 'created_at': now()}
    try:
        import torch
        info.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        info.update(cuda=None, gpu=None)
    return info
