import csv
import tempfile
import unittest
from pathlib import Path

from scripts.workstation_round2 import error_diagnostics, measure_pass, summarize_benchmark
from src.common import write_json


class RoundTwoTests(unittest.TestCase):
    def test_diagnostic_separates_duplicates_localization_and_unmatched_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            dataset=root/'data'
            (dataset/'images/val').mkdir(parents=True)
            (dataset/'images/val/video_30_60_1.jpg').touch()
            (dataset/'labels/val').mkdir(parents=True)
            (dataset/'labels/val/video_30_60_1.txt').write_text('1 0.5 0.5 0.2 0.2\n',encoding='utf-8')
            folder=root/'runs/B1/seed0/val_predictions/full'
            signature={'config':{'operating_conf':.25}}
            prediction={'split':'val','partial':False,'signature':signature,'images':[
                {'image':'video_30_60_1.jpg','width':100,'height':100,'detections':[
                    {'class_id':1,'confidence':.9,'box':[40,40,60,60]},
                    {'class_id':1,'confidence':.8,'box':[40,40,60,60]},
                    {'class_id':1,'confidence':.7,'box':[40,40,50,50]},
                    {'class_id':1,'confidence':.6,'box':[0,0,10,10]}]}]}
            metric={'signature':signature,'per_class':{'gun':{'tp':1,'fp':3,'fn':0}}}
            write_json(folder/'predictions.json',prediction)
            write_json(folder/'metrics.json',metric)
            totals=error_diagnostics({'runs':root/'runs','dataset':dataset},['B1'],0,root/'out')
            self.assertEqual(totals[0]['fp_duplicado'],1)
            self.assertEqual(totals[0]['fp_localizacion'],1)
            self.assertEqual(totals[0]['fp_sin_solapamiento'],1)
            with (root/'out/diagnostico_gun_objetos.csv').open(encoding='utf-8') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),3)
            self.assertEqual({r['video'] for r in rows},{'video'})
            metric['per_class']['gun']['tp']=2
            write_json(folder/'metrics.json',metric)
            with self.assertRaises(ValueError):
                error_diagnostics({'runs':root/'runs','dataset':dataset},['B1'],0,root/'bad')

    def test_pass_times_read_and_prediction_and_summarizes_repetitions(self):
        clock=iter([0.,.01,1.,1.03])
        calls=[]
        values,rows=measure_pass([Path('a.jpg'),Path('b.jpg')],
            lambda name: calls.append('read') or name,
            lambda image:calls.append('predict'),lambda:None,
            lambda:calls.append('reset'),lambda:123.,timer=lambda:next(clock))
        self.assertEqual(calls,['reset','read','predict','read','predict'])
        self.assertAlmostEqual(values['mean_image_ms'],20)
        self.assertAlmostEqual(values['fps'],50)
        self.assertEqual(values['peak_allocated_mib'],123)
        base={'experiment':'B1','seed':0,'images':2,'imgsz':640,'peak_allocated_mib':100}
        summary=summarize_benchmark([{**base,'mean_image_ms':10},{**base,'mean_image_ms':20}])[0]
        self.assertEqual(summary['repetitions'],2)
        self.assertEqual(summary['mean_image_ms'],15)
        self.assertAlmostEqual(summary['sd_image_ms'],7.0710678118654755)
        self.assertAlmostEqual(summary['fps_from_mean_ms'],66.66666666666667)
