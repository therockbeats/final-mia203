import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from src.geometry import iou, nms, windows
from src.evaluation import average_precision, match, evaluate
from src.data import audit, extract
from src.runtime import patch_source
from unittest.mock import patch


class DetectionTests(unittest.TestCase):
    def test_final_notebook_results_keep_validation_and_test_distinct(self):
        from scripts.import_final_results import read_results
        tables = '''EXPERIMENTOS SOBRE VALIDATION
B1 YOLOv5s / 640 Validation 84 0.928400 0.798420 0.851590 0.669360 0.687583 580 132
M1 YOLOv5s / 1280 Validation 99 0.779180 0.764080 0.790320 0.524020 0.550650 565 147
M2 YOLOv5s / 640 / Multi-Scale Validation 98 0.912800 0.812900 0.861920 0.680160 0.698336 590 122
MODELO FINAL M2 SOBRE TEST
all 508 1288 0.875 0.780 0.845 0.659
person 508 400 0.933 0.863 0.926 0.812
gun 508 888 0.817 0.697 0.765 0.506
MEJORA M2 RESPECTO DE B1'''
        diagnostic = '''COMPARACION DIAGNOSTICA B1 VS M1 VS M2
Clase: person
TP B1 / M1 / M2: 394 / 394 / 396
FN B1 / M1 / M2: 16 / 16 / 14
FP B1 / M1 / M2: 816 / 1546 / 884
Clase: gun
TP B1 / M1 / M2: 580 / 565 / 590
FN B1 / M1 / M2: 132 / 147 / 122
FP B1 / M1 / M2: 3232 / 4141 / 3978
EVALUACION DE LA HIPOTESIS M2'''
        cost = 'EVIDENCIA DE TIEMPOS EN LOGS\nB1\n1.823 hours\nM1\n3.206 hours\nM2\n1.837 hours'
        selection = 'RESULTADO: MODELO FINAL SELECCIONADO Y CONGELADO\nSHA256: '+('a'*64)
        cells = [{'source':[], 'outputs':[{'text':[text]}]} for text in [tables, diagnostic, cost, selection]]
        with tempfile.TemporaryDirectory() as d:
            file = Path(d)/'run.ipynb'; file.write_text(json.dumps({'cells':cells}))
            validation, testing, errors, times, selected = read_results(file)
            self.assertEqual({r['experiment'] for r in validation}, {'B1','M1','M2'})
            self.assertEqual({r['experiment'] for r in testing}, {'M2'})
            self.assertEqual(next(r['map50_95'] for r in validation if r['experiment']=='M2'), .68016)
            self.assertEqual(next(r['map50_95'] for r in testing if r['class']=='all'), .659)
            self.assertEqual(next(r['fn'] for r in errors if r['experiment']=='M2' and r['class']=='gun'), 122)
            self.assertAlmostEqual(sum(r['training_hours'] for r in times), 6.866)
            self.assertEqual(selected['best_epoch'], 98)

    def test_prepare_uses_utf8_with_windows_default_and_can_resume(self):
        import src.runtime as runtime
        original = "# 诏: UTF-8 source\nimport torch\nx = torch.load('weights.pt')\n"
        with self.assertRaises(UnicodeDecodeError):
            original.encode('utf-8').decode('cp1252')
        native_open = Path.open

        def windows_open(path, mode='r', buffering=-1, encoding=None,
                         errors=None, newline=None):
            if 'b' not in mode and encoding is None:
                encoding = 'cp1252'
            return native_open(path, mode, buffering, encoding, errors, newline)

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            yolo = root/'yolo'
            yolo.mkdir()
            weights = root/'weights.pt'
            weights.write_bytes(b'fixture')
            for name in runtime.PATCH_FILES:
                file = yolo/name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(original, encoding='utf-8')

            def git_output(command, **kwargs):
                if 'rev-parse' in command:
                    return runtime.YOLO_COMMIT+'\n'
                return original.encode('utf-8')

            with patch.object(Path, 'open', windows_open), \
                 patch.object(runtime.subprocess, 'check_output', side_effect=git_output), \
                 patch.object(runtime, 'INITIAL_WEIGHTS_SHA256', runtime.sha256(weights)):
                first = runtime.prepare(yolo, weights)
                second = runtime.prepare(yolo, weights)
                self.assertEqual(first, second)
                self.assertEqual(runtime.verify_source(yolo), first)
            for name in runtime.PATCH_FILES:
                self.assertIn('诏', (yolo/name).read_text(encoding='utf-8'))
                self.assertNotIn(b'\r\n', (yolo/name).read_bytes())

    def test_multiscale_uses_integer_bounds(self):
        original = 'sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs\n'
        patched = patch_source(original, 'train.py')
        self.assertIn('random.randrange(int(imgsz * 0.5), int(imgsz * 1.5 + gs))', patched)
        self.assertEqual(patched, patch_source(patched, 'train.py'))

    def test_m2_trains_multiscale_and_loads_own_weights(self):
        import io
        from contextlib import redirect_stdout
        import src.training as training
        from src.inference import weight_path
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            settings = {'runs': root/'runs', 'dataset': root/'data', 'initial_weights': root/'initial.pt', 'yolo': root/'yolo'}
            self.assertEqual(weight_path(settings, 'M2', 0), root/'runs/M2/seed0/run/weights/best.pt')
            with patch.object(training, 'paths', return_value=settings), patch.object(training, 'verify_source', return_value={}), patch('src.inference.dataset_digest', return_value='fixture'):
                text = io.StringIO()
                with redirect_stdout(text): training.train('M2', 0, dry_run=True)
                self.assertIn('--multi-scale', text.getvalue())
                text = io.StringIO()
                with redirect_stdout(text): training.train('B1', 0, dry_run=True)
                self.assertNotIn('--multi-scale', text.getvalue())

    def test_import_m1_run_and_summary(self):
        import csv
        import src.reporting as report
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root/'source/results/M1_YOLOv5s_1280'
            run = source/'run'
            (run/'weights').mkdir(parents=True)
            for name in ['best.pt', 'last.pt']:
                (run/'weights'/name).write_bytes(b'checkpoint-fixture')
            with (run/'results.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['epoch','metrics/mAP_0.5','metrics/mAP_0.5:0.95'])
                writer.writeheader()
                writer.writerows({'epoch':i,'metrics/mAP_0.5':.7,'metrics/mAP_0.5:0.95':i/200} for i in range(100))
            (source/'M1_training.log').write_text('100 epochs completed in 3.206 hours.\nValidating best.pt\n gun 494 712 0.745 0.637 0.673 0.394\n')
            with patch.object(report,'ROOT',root/'publication'),patch.object(report,'paths',return_value={'runs':root/'work/runs'}):
                metadata = report.import_experiment(root/'source','M1')
                self.assertEqual(metadata['training_config']['imgsz'],1280)
                self.assertEqual(metadata['best_csv_epoch_zero_based'],99)
                self.assertEqual(metadata['training_hours_log'],3.206)
                self.assertTrue((root/'work/runs/M1/seed0/run/weights/best.pt').exists())
                summary = json.loads((root/'publication/results/imported_M1/summary.json').read_text())
                self.assertEqual(summary['experiment'],'M1')
                self.assertNotIn('sha256',summary)
                with self.assertRaises(FileExistsError): report.import_experiment(root/'source','M1')

    def test_colab_requirements_preserve_torch(self):
        from scripts.install_colab import supplemental_requirements
        text = '# paquetes\ntorch==2.11.0\ntorchvision==0.26.0\nnumpy==2.2.6\nPyYAML==6.0.3\n'
        extra = supplemental_requirements(text)
        self.assertNotIn('torch', extra)
        self.assertIn('numpy==2.2.6', extra)
        self.assertIn('PyYAML==6.0.3', extra)

    def test_main_protocol_freezes_b1_m1(self):
        import argparse
        import src.pipeline as module
        from src.common import write_json
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ['B1', 'M1']:
                write_json(root/f'runs/{name}/seed0/val_predictions/full/metrics.json', {'signature': {'model': name}})
            args = argparse.Namespace(seeds=[0], rationale='Comparación de resolución', experiments=['B1','M1'])
            with patch.object(module, 'ROOT', root), patch.object(module, 'paths', return_value={'runs': root/'runs'}), patch('src.inference.signature', side_effect=lambda p,n,s: {'model':n}):
                module.freeze(args)
            saved = json.loads((root/'configs/frozen_protocol.json').read_text())
            self.assertEqual(set(saved['experiments']), {'B1/seed0', 'M1/seed0'})
            self.assertEqual(saved['evaluation_split'], 'test')

    def test_iou_and_nms(self):
        self.assertAlmostEqual(iou([0,0,10,10],[0,0,5,10]),.5)
        ds=[{'class_id':0,'box':[0,0,10,10],'confidence':.9},
            {'class_id':0,'box':[0,0,10,10],'confidence':.8},
            {'class_id':1,'box':[0,0,10,10],'confidence':.7}]
        self.assertEqual([x['class_id'] for x in nms(ds)], [0,1])

    def test_windows_cover_edges_and_map_offsets(self):
        for w,h in [(100,80),(1,1),(1920,1080),(5,9)]:
            tiles=windows(w,h)
            self.assertEqual(len(tiles),len(set(tiles)))
            self.assertTrue(all(0<=a<c<=w and 0<=b<d<=h for a,b,c,d in tiles))
            for x,y in [(0,0),(w-1,0),(0,h-1),(w-1,h-1),(w//2,h//2)]:
                self.assertTrue(any(a<=x<c and b<=y<d for a,b,c,d in tiles))
        self.assertEqual(len(windows(100,100)),4)

    def test_ap_known_cases(self):
        self.assertEqual(average_precision([(.9,1)],1),1)
        self.assertEqual(average_precision([],1),0)
        self.assertIsNone(average_precision([(.9,0)],0))
        self.assertAlmostEqual(average_precision([(.9,0),(.8,1)],1),.5)
        self.assertEqual(average_precision([(.9,1),(.8,0)],1),1)
        self.assertAlmostEqual(average_precision([(.9,1)],2),51/101)

    def test_matching_one_to_one_and_class(self):
        targets=[{'class_id':1,'box':[0,0,10,10]}]
        detections=[{'class_id':0,'box':[0,0,10,10],'confidence':.99},
                    {'class_id':1,'box':[0,0,10,10],'confidence':.9},
                    {'class_id':1,'box':[0,0,10,10],'confidence':.8}]
        matches,used=match(detections,targets,.5)
        self.assertEqual([m[1] for m in matches],[False,True,False])
        self.assertEqual(used,{0})

    def test_class_mapping_must_match_trained_checkpoint(self):
        from src.inference import verify_model_classes
        from types import SimpleNamespace
        verify_model_classes(SimpleNamespace(names={0:'person',1:'gun'}))
        with self.assertRaises(ValueError): verify_model_classes(SimpleNamespace(names=['gun','person']))
        with self.assertRaises(ValueError): verify_model_classes(SimpleNamespace(names=['person','car','gun']))

    def test_secure_extraction(self):
        with tempfile.TemporaryDirectory() as d:
            z=Path(d)/'bad.zip'
            with zipfile.ZipFile(z,'w') as f: f.writestr('../escape.txt','x')
            with self.assertRaises(ValueError): extract(z,Path(d)/'out')
            self.assertFalse((Path(d)/'escape.txt').exists())

    def test_dataset_leakage_and_correspondence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'processed'
            for split,vid in [('train','videoA'),('val','videoB'),('test','videoC')]:
                for kind in ['images','labels']: (root/kind/split).mkdir(parents=True)
                (root/'images'/split/f'{vid}_30_60_000001.jpg').write_bytes(split.encode())
                (root/'labels'/split/f'{vid}_30_60_000001.txt').write_text('1 0.5 0.5 0.1 0.1\n')
            out=Path(d)/'audit.json'
            self.assertTrue(audit(root,out)['valid'])
            (root/'images/test/videoC_30_60_000001.jpg').write_bytes(b'train')
            with self.assertRaises(ValueError): audit(root,out)
            self.assertFalse(json.loads(out.read_text())['valid'])

    def test_modern_checkpoint_patch_idempotent(self):
        text="x = torch.load(w, map_location='cpu')\ny = torch.load(last, map_location='cpu')['opt']\n"
        patched=patch_source(text,'train.py')
        self.assertEqual(patched.count('weights_only=False'),2)
        self.assertEqual(patched,patch_source(patched,'train.py'))

    def test_changed_label_invalidates_manifest(self):
        from src.data import verify_manifest
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'processed'
            for split,vid in [('train','A'),('val','B'),('test','C')]:
                for kind in ['images','labels']: (root/kind/split).mkdir(parents=True)
                (root/'images'/split/f'{vid}_30_60_1.jpg').write_bytes(split.encode())
                (root/'labels'/split/f'{vid}_30_60_1.txt').write_text('1 0.5 0.5 0.1 0.1')
            out=Path(d)/'audit.json';audit(root,out)
            verify_manifest(root,out)
            (root/'labels/test/C_30_60_1.txt').write_text('1 0.5 0.5 0.2 0.2')
            with self.assertRaises(ValueError): verify_manifest(root,out)

    def test_test_predictions_require_frozen_protocol(self):
        import src.inference as module
        with tempfile.TemporaryDirectory() as d:
            with patch.object(module,'ROOT',Path(d)), patch.object(module,'paths',return_value={'runs':Path(d)/'runs'}), patch.object(module,'signature',return_value={}):
                with self.assertRaises(ValueError): module.predict('B1','test',0)

    def test_freeze_rejects_changed_validation_signature(self):
        import argparse
        import src.pipeline as module
        from src.common import write_json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);metric=root/'runs/B1/seed0/val_predictions/full/metrics.json'
            write_json(metric,{'signature':{'weights':'old'}})
            args=argparse.Namespace(seeds=[0],rationale='Resultados de validación',experiments=['B1'])
            with patch.object(module,'ROOT',root), patch.object(module,'paths',return_value={'runs':root/'runs'}), patch('src.inference.signature',return_value={'weights':'new'}):
                with self.assertRaises(ValueError): module.freeze(args)
            self.assertFalse((root/'configs/frozen_protocol.json').exists())

    def test_evaluate_end_to_end_and_reject_partial(self):
        from src.common import sha256
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'images/val').mkdir(parents=True);(root/'labels/val').mkdir(parents=True)
            image=root/'images/val/v_30_60_1.jpg';image.write_bytes(b'fixture')
            (root/'labels/val/v_30_60_1.txt').write_text('0 0.5 0.5 1 1\n1 0.1 0.1 0.1 0.1\n')
            pred={'split':'val','experiment':'B1','seed':0,'partial':False,'peak_allocated_mb':None,'environment':{},
                'signature':{'config':{'operating_conf':.25,'small_area_relative':.01}},
                'images':[{'image':image.name,'image_sha256':sha256(image),'width':100,'height':100,'seconds':1,
                    'detections':[{'class_id':0,'box':[0,0,100,100],'confidence':.9},
                                  {'class_id':1,'box':[5,5,15,15],'confidence':.9}]}]}
            file=root/'pred.json';file.write_text(json.dumps(pred))
            result=evaluate(file,root,root/'metrics.json')
            self.assertEqual(result['macro']['map50_95'],1)
            self.assertEqual(result['per_class']['gun']['recall'],1)
            pred['partial']=True;file.write_text(json.dumps(pred))
            with self.assertRaises(ValueError): evaluate(file,root,root/'metrics.json')


if __name__=='__main__': unittest.main()
