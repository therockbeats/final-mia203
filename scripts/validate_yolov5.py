"""Reproduce validación y evaluación final con YOLOv5 v7.0."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('experiment', choices=['B1', 'M1', 'M2'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device')
    p.add_argument('--split', choices=['val','test'], default='val')
    a = p.parse_args()
    from src.common import paths, config, sha256
    from src.inference import weight_path, signature
    from src.runtime import configure_cache
    settings, cfg = paths(), config(a.experiment)
    if a.split == 'test':
        selected = json.loads((ROOT/'configs/final_selection.json').read_text())
        if a.experiment != selected['experiment'] or a.seed != selected['seed']:
            raise ValueError('TEST corresponde al modelo y semilla seleccionados')
        if sha256(weight_path(settings, a.experiment, a.seed)) != selected['expected_weights_sha256']:
            raise ValueError('El checkpoint difiere del modelo seleccionado para TEST')
    signature(settings, a.experiment, a.seed)
    configure_cache(settings)
    data_yaml = settings['runs']/'dataset.yaml'
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text('path: '+json.dumps(str(settings['dataset']))+
        '\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 2\nnames: [person, gun]\n')
    parent = settings['runs']/a.experiment/f'seed{a.seed}'/f'{a.split}_yolov5'
    destination = parent/'run'
    index = 1
    while destination.exists():
        index += 1
        destination = parent/f'run{index}'
    destination.mkdir(parents=True)
    command = [sys.executable, str(settings['yolo']/'val.py'),
        '--data', str(data_yaml), '--weights', str(weight_path(settings, a.experiment, a.seed)),
        '--imgsz', str(cfg['imgsz']), '--batch-size', str(cfg['batch_size'] if a.split == 'test' else cfg['batch_size']*2),
        '--conf-thres', str(cfg['prediction_conf']), '--iou-thres', str(cfg['nms_iou']),
        '--task', a.split, '--device', a.device or cfg['device'],
        '--project', str(parent), '--name', destination.name, '--exist-ok']
    with (destination/'validation.log').open('w') as log:
        process = subprocess.Popen(command, cwd=settings['yolo'], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        code = process.wait()
    if code:
        raise RuntimeError(f'Validación terminada con código {code}: {destination}')
    print(destination)


if __name__ == '__main__':
    main()
