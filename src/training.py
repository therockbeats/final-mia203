from pathlib import Path
import csv
import os
import sys
import time
import subprocess
from .common import paths, config, sha256, write_json, read_json, now
from .runtime import verify_source, environment, configure_cache


def train(name, seed, smoke=False, dry_run=False):
    p, c = paths(), config(name)
    configure_cache(p)
    from .inference import dataset_digest
    dataset_hash = dataset_digest(p)
    source = verify_source(p['yolo'])
    destination = p['runs'] / name / f'seed{seed}' / ('smoke' if smoke else 'run')
    epochs = 1 if smoke else c['epochs']
    data_yaml = p['runs'] / 'dataset.yaml'
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    # JSON strings are valid YAML scalars and preserve spaces in paths.
    import json
    data_yaml.write_text('path: ' + json.dumps(str(p['dataset'])) + '\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 2\nnames: [person, gun]\n')
    command = [sys.executable, str(p['yolo']/'train.py'), '--data', str(data_yaml),
        '--weights', str(p['initial_weights']), '--imgsz', str(c['imgsz']), '--epochs', str(epochs),
        '--batch-size', str(c['batch_size']), '--workers', str(c['workers']), '--seed', str(seed),
        '--optimizer', c['optimizer'], '--device', c['device'], '--project', str(destination.parent),
        '--name', destination.name, '--save-period', '10']
    if c.get('multi_scale', False):
        command.append('--multi-scale')
    csv_path = destination / 'results.csv'
    completed = 0
    if csv_path.exists():
        completed = len(list(csv.DictReader(csv_path.open())))
    if completed >= epochs and (destination/'weights/best.pt').exists():
        protocol = destination.parent / (destination.name + '.protocol.json')
        imported = destination.parent / 'imported.provenance.json'
        if protocol.exists():
            saved = read_json(protocol)
            if saved['config'] != c or saved['seed'] != seed or saved['source'] != source or saved['dataset_sha256'] != dataset_hash:
                raise ValueError('Run completo con otro protocolo; usar una carpeta de artefactos nueva')
        elif imported.exists() and not smoke:
            saved = read_json(imported)
            if seed != saved['seed'] or any(c[k] != v for k,v in saved['training_config'].items()):
                raise ValueError('La configuración actual difiere del entrenamiento importado')
        else:
            raise ValueError('Run completo sin procedencia verificable')
        print(f'Entrenamiento completo: {completed} épocas · {destination}')
        return
    if destination.exists():
        last = destination/'weights/last.pt'
        if not last.exists():
            raise ValueError('Run existente sin last.pt; revisar sin sobrescribir')
        import torch
        sys.path.insert(0, str(p['yolo']))
        ckpt = torch.load(last, map_location='cpu', weights_only=False)
        if ckpt.get('optimizer') is None or ckpt.get('epoch', -1) < 0:
            raise ValueError('Checkpoint final sin estado de reanudación')
        saved = read_json(destination.parent / (destination.name + '.protocol.json'))
        if saved['config'] != c or saved['seed'] != seed or saved['source'] != source or saved['dataset_sha256'] != dataset_hash or saved['initial_weights_sha256'] != sha256(p['initial_weights']):
            raise ValueError('La configuración difiere de la ejecución que se desea reanudar')
        command = [sys.executable, str(p['yolo']/'train.py'), '--resume', str(last)]
    if dry_run:
        print(command)
        return
    if smoke:
        train_list = p['runs']/'smoke_images.txt'
        from .data import images
        train_list.write_text('\n'.join(str(x) for x in images(p['dataset'], 'train')[:64])+'\n')
        smoke_yaml = p['runs']/'smoke.yaml'
        smoke_yaml.write_text(data_yaml.read_text().replace('train: images/train', 'train: '+json.dumps(str(train_list))))
        if '--data' in command:
            command[command.index('--data') + 1] = str(smoke_yaml)
    protocol_file = destination.parent / (destination.name + '.protocol.json')
    if not protocol_file.exists():
        write_json(protocol_file, {'config': c, 'seed': seed, 'source': source,
            'initial_weights_sha256': sha256(p['initial_weights']), 'dataset_sha256': dataset_hash,
            'environment': environment(), 'created_at': now()})
    log = destination.parent / (destination.name + '.console.log')
    start = time.perf_counter()
    env = {**os.environ, 'WANDB_MODE': 'disabled', 'COMET_MODE': 'DISABLED', 'PYTHONHASHSEED': str(seed)}
    with log.open('a') as f:
        proc = subprocess.Popen(command, cwd=p['yolo'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end='', flush=True)
            f.write(line); f.flush()
        code = proc.wait()
    elapsed = time.perf_counter() - start
    history_path = destination.parent / (destination.name + '.sessions.json')
    history = read_json(history_path) if history_path.exists() else []
    history.append({'finished_at': now(), 'seconds': elapsed, 'returncode': code, 'environment': environment()})
    write_json(history_path, history)
    if code:
        raise RuntimeError(f'Entrenamiento interrumpido ({code}). Log: {log}')
