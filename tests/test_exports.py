import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common import write_json
from src.exports import parse_yolov5_log, export_yolov5, export_workstation


TABLE = '''\x1b[1mClass Images Instances P R mAP50 mAP50-95\x1b[0m
 all 494 1122 .914 .813 .863 .681
 person 494 410 .957 .927 .963 .832
 gun 494 712 .871 7e-1 .763 .529
'''


class ExportTests(unittest.TestCase):
    def test_parser_handles_ansi_progress_and_repeated_tables(self):
        text = ' Class Images Instances P R mAP50 mAP50-95: 100%|██| 16/16\r\n'
        text += TABLE.replace('.914', '.111') + TABLE
        rows = parse_yolov5_log(text)
        self.assertEqual([r['class'] for r in rows], ['all', 'person', 'gun'])
        self.assertEqual(rows[0]['precision'], .914)
        self.assertEqual(rows[2]['recall'], .7)
        self.assertEqual(rows[0]['instances'], 1122)
        with self.assertRaises(ValueError):
            parse_yolov5_log(TABLE + '\n all 494 1122 .1 .2 .3 .4\n')
        with self.assertRaises(ValueError):
            parse_yolov5_log(TABLE.replace('1122', '1000'))

    def test_export_uses_latest_run_and_replaces_empty_csv_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs, out = root/'runs', root/'publication'
            out.mkdir()
            (out/'validacion_yolov5.csv').touch()
            for model in ['B1', 'M1', 'M2']:
                for index in ['', '2', '10']:
                    log = runs/model/'seed0/val_yolov5'/('run'+index)/'validation.log'
                    log.parent.mkdir(parents=True)
                    log.write_text(TABLE.replace('.914', '.111') if index != '10' else TABLE, encoding='utf-8')
            for _ in range(2):
                rows = export_yolov5(runs, out, seeds=[0])
                self.assertEqual(len(rows), 9)
                self.assertTrue(all(r['precision'] == .914 for r in rows if r['class'] == 'all'))
            with (out/'validacion_yolov5.csv').open(encoding='utf-8') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 9)
            log = runs/'M2/seed0/val_yolov5/run11/validation.log'
            log.parent.mkdir()
            log.write_text('Evaluation interrupted\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                export_yolov5(runs, out, seeds=[0])
            self.assertEqual(len(export_yolov5(runs, out, seeds=[0], strict=False)), 6)

    def test_analysis_exports_thresholds_timings_and_preserves_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, runs, out = root/'dataset', root/'runs', root/'public'
            labels = dataset/'labels/val'
            labels.mkdir(parents=True)
            (labels/'sample.txt').write_text('1 0.5 0.5 0.2 0.2\n', encoding='utf-8')
            folder = runs/'B1/seed0/val_predictions/full'
            metric = {'mean_image_seconds': .01, 'images': 1,
                      'signature': {'config': {'imgsz': 640}},
                      'environment': {'gpu': 'Test GPU'}, 'peak_allocated_mb': 100,
                      'per_class': {'gun': {'tp': 1, 'fp': 0, 'fn': 0}},
                      'small_gun': {'gt': 1, 'recall': 1}}
            write_json(folder/'metrics.json', metric)
            write_json(folder.parent/'medicion_1/metrics.json', {**metric, 'mean_image_seconds': .02})
            write_json(folder/'errors.json', [{'class_id': 1, 'type': 'FN', 'area_relative': .004,
                                              'image': 'sample.jpg', 'box': [40, 40, 60, 60]}])
            write_json(folder/'predictions.json', {'split': 'val', 'partial': False, 'images': [
                {'image': 'sample.jpg', 'width': 100, 'height': 100, 'detections': [
                    {'class_id': 1, 'confidence': .8, 'box': [40, 40, 60, 60]},
                    {'class_id': 1, 'confidence': .2, 'box': [0, 0, 10, 10]}]}]})
            settings = {'runs': runs, 'dataset': dataset}
            with patch('src.exports.ROOT', root):
                export_workstation(settings, out, ['B1'], [0])
                with (out/'umbrales_gun.csv').open(encoding='utf-8') as stream:
                    thresholds = list(csv.DictReader(stream))
                self.assertEqual(len(thresholds), 6)
                self.assertEqual(float(thresholds[0]['precision']), .5)
                self.assertEqual(float(thresholds[2]['precision']), 1)
                with (out/'rendimiento.csv').open(encoding='utf-8') as stream:
                    timings = list(csv.DictReader(stream))
                self.assertEqual([int(row['repetition']) for row in timings], [1, 2])
                self.assertEqual(float(timings[0]['mean_image_ms']), 20)
                self.assertEqual(float(timings[1]['fps']), 100)
                with (out/'rendimiento_resumen.csv').open(encoding='utf-8') as stream:
                    summary = list(csv.DictReader(stream))[0]
                self.assertEqual(int(summary['n_repetitions']), 2)
                self.assertEqual(float(summary['mean_image_ms_mean']), 15)
                annotation = out/'revision_errores.csv'
                annotation.write_text('manual observations: preserve me\n', encoding='utf-8')
                export_workstation(settings, out, ['B1'], [0])
                self.assertIn('preserve me', annotation.read_text(encoding='utf-8'))
                self.assertIn('confianza 0.25', (out/'resumen_automatico.md').read_text(encoding='utf-8'))
