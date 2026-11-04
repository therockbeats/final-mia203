"""Ejecuta la entrada CLI completa sin modelos ni GPU, con predicciones ya guardadas."""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReproduceCommandTests(unittest.TestCase):
    def test_validation_and_report_finish_exports_and_reuse_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            repo = parent/'proyecto con espacios'
            (repo/'scripts').mkdir(parents=True)
            shutil.copy2(ROOT/'scripts/reproduce.py', repo/'scripts/reproduce.py')
            shutil.copytree(ROOT/'src', repo/'src', ignore=shutil.ignore_patterns('__pycache__'))
            # Record dispatched phases; inference is deliberately unavailable in this fixture.
            (repo/'src/pipeline.py').write_text('''import sys
from pathlib import Path
if sys.argv[1] == 'predict':
    raise RuntimeError('No debe repetir inferencia con predicciones guardadas')
with Path('commands.txt').open('a', encoding='utf-8') as stream:
    stream.write(sys.argv[1]+'\\n')
''', encoding='utf-8')
            work = parent/'artifacts'
            dataset = work/'dataset'
            labels = dataset/'labels/val'
            labels.mkdir(parents=True)
            (labels/'sample.txt').write_text('1 0.5 0.5 0.2 0.2\n', encoding='utf-8')
            config = repo/'configs/paths.local.json'
            config.parent.mkdir()
            config.write_text(json.dumps({'runs': str(work/'runs'), 'dataset': str(dataset)}), encoding='utf-8')
            folder = work/'runs/B1/seed0/val_predictions/full'
            folder.mkdir(parents=True)
            signature = {'config': {'imgsz': 640}}
            prediction = {'split': 'val', 'partial': False, 'signature': signature,
                          'images': [{'image': 'sample.jpg', 'width': 100, 'height': 100,
                                      'detections': [{'class_id': 1, 'confidence': .8,
                                                      'box': [40, 40, 60, 60]}]}]}
            metric = {'signature': signature, 'images': 1, 'mean_image_seconds': .01,
                      'environment': {'gpu': 'Fixture GPU'}, 'peak_allocated_mb': 100,
                      'per_class': {'gun': {'tp': 1, 'fp': 0, 'fn': 0}},
                      'small_gun': {'gt': 0, 'recall': None}}
            for name, data in [('predictions.json', prediction), ('metrics.json', metric), ('errors.json', [])]:
                (folder/name).write_text(json.dumps(data), encoding='utf-8')
            log = work/'runs/B1/seed0/val_yolov5/run/validation.log'
            log.parent.mkdir(parents=True)
            log.write_text(' all 1 1 .9 .8 .7 .6\n person 1 0 0 0 0 0\n gun 1 1 .9 .8 .7 .6\n', encoding='utf-8')
            environment = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
            environment.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
            for phase in ['validation', 'report']:
                with self.subTest(phase=phase):
                    result = subprocess.run(
                        [sys.executable, str(repo/'scripts/reproduce.py'), phase,
                         '--experiments', 'B1', '--seeds', '0'],
                        cwd=parent, env=environment, capture_output=True, text=True,
                        encoding='utf-8', timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
                    out = repo/'results/workstation/grupo'
                    self.assertIn('Tablas, figuras y resumen guardados', result.stdout)
                    for name in ['validacion_yolov5.csv', 'rendimiento.csv',
                                 'umbrales_gun.csv', 'resumen_automatico.md']:
                        self.assertTrue((out/name).is_file(), name)
                    with (out/'validacion_yolov5.csv').open(encoding='utf-8') as stream:
                        self.assertEqual(len(list(csv.DictReader(stream))), 3)
            commands = (repo/'commands.txt').read_text(encoding='utf-8').splitlines()
            self.assertEqual(commands, ['evaluate', 'figures', 'errors', 'figures', 'errors'])
