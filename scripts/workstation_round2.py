"""Exporta diagnóstico de errores guardados y mide inferencia en tres pasadas."""
import argparse
from collections import Counter
import gc
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import paths, config, read_json, write_json, now
from src.data import images, labels
from src.geometry import iou, xywh_to_xyxy
from src.evaluation import match
from src.exports import csv_file, output_folder


def error_diagnostics(settings, experiments, seed, out):
    expected = {p.name for p in images(settings['dataset'], 'val')}
    details, totals = [], []
    by_video = Counter()
    for name in experiments:
        folder = settings['runs']/name/f'seed{seed}'/'val_predictions/full'
        pred, metric = read_json(folder/'predictions.json'), read_json(folder/'metrics.json')
        actual = [row['image'] for row in pred['images']]
        if pred['split'] != 'val' or pred['partial'] or len(actual) != len(expected) or set(actual) != expected:
            raise ValueError(f'{name}: se requieren las predicciones completas de Validation')
        if pred['signature'] != metric['signature']:
            raise ValueError(f'{name}: predicciones y métricas corresponden a protocolos distintos')
        counts = Counter()
        confidence = pred['signature']['config']['operating_conf']
        for image in pred['images']:
            gt = [{'class_id': b[0], 'box': xywh_to_xyxy(b, image['width'], image['height']),
                   'area_relative': b[3]*b[4]} for b in labels(settings['dataset'], 'val', Path(image['image']).stem) if b[0] == 1]
            detections = [d for d in image['detections'] if d['class_id'] == 1 and d['confidence'] >= confidence]
            matches, used = match(detections, gt, .5)
            counts['tp'] += len(used)
            counts['gt'] += len(gt)
            video = Path(image['image']).stem.rsplit('_', 3)[0]
            for detection, correct, _ in matches:
                if correct:
                    continue
                best = max((iou(detection['box'], t['box']) for t in gt), default=0)
                category = 'duplicado' if best >= .5 else ('localizacion' if best > 0 else 'sin_solapamiento')
                details.append({'experiment': name, 'seed': seed, 'image': image['image'], 'video': video,
                                'type': 'FP', 'category': category, 'confidence': detection['confidence'],
                                'best_iou': best, 'area_relative_gt': '',
                                **dict(zip(['x1','y1','x2','y2'], detection['box']))})
                counts['fp'] += 1
                counts['fp_'+category] += 1
                by_video[(name, video, 'FP', category)] += 1
            for index, target in enumerate(gt):
                if index in used:
                    continue
                best = max((iou(target['box'], d['box']) for d in detections), default=0)
                category = 'asignacion' if best >= .5 else ('localizacion' if best > 0 else 'sin_candidato')
                details.append({'experiment': name, 'seed': seed, 'image': image['image'], 'video': video,
                                'type': 'FN', 'category': category, 'confidence': '', 'best_iou': best,
                                'area_relative_gt': target['area_relative'],
                                **dict(zip(['x1','y1','x2','y2'], target['box']))})
                counts['fn'] += 1
                counts['fn_'+category] += 1
                by_video[(name, video, 'FN', category)] += 1
        saved = metric['per_class']['gun']
        if any(counts[key] != saved[key] for key in ['tp','fp','fn']):
            raise ValueError(f'{name}: los conteos no reproducen la evaluación guardada')
        totals.append({'experiment': name, 'seed': seed, 'confidence': confidence,
                       **{key: counts[key] for key in ['gt','tp','fp','fn','fp_localizacion','fp_duplicado',
                                                       'fp_sin_solapamiento','fn_localizacion','fn_asignacion','fn_sin_candidato']}})
    csv_file(out/'diagnostico_gun_objetos.csv', details)
    csv_file(out/'diagnostico_gun_resumen.csv', totals)
    csv_file(out/'diagnostico_gun_por_video.csv', [dict(zip(['experiment','video','type','category','count'], (*key,value)))
                                                for key,value in sorted(by_video.items())])
    return totals


def measure_pass(image_paths, read_image, predict, synchronize, reset_peak, peak_memory, timer=time.perf_counter):
    synchronize()
    reset_peak()
    rows = []
    for file in image_paths:
        synchronize()
        start = timer()
        image = read_image(str(file))
        if image is None:
            raise ValueError(f'Imagen ilegible: {file.name}')
        predict(image)
        synchronize()
        rows.append({'image': file.name, 'seconds': timer()-start})
    mean = statistics.mean(row['seconds'] for row in rows)
    return {'images': len(rows), 'mean_image_ms': mean*1000, 'fps': 1/mean,
            'peak_allocated_mib': peak_memory()}, rows


def summarize_benchmark(rows):
    summaries = []
    for name in dict.fromkeys(row['experiment'] for row in rows):
        group = [r for r in rows if r['experiment'] == name]
        milliseconds = [r['mean_image_ms'] for r in group]
        mean = statistics.mean(milliseconds)
        summaries.append({'experiment': name, 'seed': group[0]['seed'], 'repetitions': len(group),
                          'images_per_repetition': group[0]['images'], 'imgsz': group[0]['imgsz'],
                          'batch': 1, 'precision': 'FP32', 'mean_image_ms': mean,
                          'sd_image_ms': statistics.stdev(milliseconds) if len(group)>1 else '',
                          'fps_from_mean_ms': 1000/mean,
                          'peak_allocated_mib': max(r['peak_allocated_mib'] for r in group)})
    return summaries


def benchmark_model(settings, name, seed, device_name, repetitions, image_paths, session):
    import cv2
    import torch
    from src.runtime import configure_cache, verify_source
    from src.inference import predict_image, verify_model_classes, weight_path
    configure_cache(settings)
    verify_source(settings['yolo'])
    sys.path.insert(0, str(settings['yolo']))
    from models.common import DetectMultiBackend
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes
    from utils.torch_utils import select_device
    device = select_device(device_name)
    c = config(name)
    model = DetectMultiBackend(weight_path(settings, name, seed), device=device, fp16=False)
    verify_model_classes(model)
    model.warmup(imgsz=(1,3,c['imgsz'],c['imgsz']))
    predict = lambda image: predict_image(model,image,c,device,torch,letterbox,non_max_suppression,scale_boxes)
    cuda = device.type == 'cuda'
    synchronize = torch.cuda.synchronize if cuda else (lambda: None)
    reset = torch.cuda.reset_peak_memory_stats if cuda else (lambda: None)
    peak = (lambda: torch.cuda.max_memory_allocated()/1024**2) if cuda else (lambda: 0)
    warmup = cv2.imread(str(image_paths[0]))
    for _ in range(3): predict(warmup)
    synchronize()
    result = []
    for repetition in range(1,repetitions+1):
        values, timings = measure_pass(image_paths,cv2.imread,predict,synchronize,reset,peak)
        result.append({'experiment':name,'seed':seed,'repetition':repetition,'imgsz':c['imgsz'],**values})
        write_json(session/f'{name}_repetition{repetition}.json', {'metrics': result[-1], 'per_image':timings})
        print(f"{name}, pasada {repetition}/{repetitions}: {values['mean_image_ms']:.2f} ms/imagen")
    del predict, model, warmup
    gc.collect()
    if cuda: torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', nargs='+', choices=['B1','M1','M2'], default=['B1','M1','M2'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--device', default='0')
    parser.add_argument('--skip-benchmark', action='store_true')
    args = parser.parse_args()
    if args.repetitions < 2: parser.error('Usar al menos dos pasadas para calcular dispersión')
    settings = paths()
    out = output_folder(settings['runs'])
    error_diagnostics(settings, args.experiments, args.seed, out)
    print('Diagnóstico por objeto y video guardado.')
    if not args.skip_benchmark:
        from src.runtime import environment
        session = settings['runs']/'benchmark_sessions'/now().replace(':','').replace('+','_')
        session.mkdir(parents=True)
        rows = []
        files = images(settings['dataset'],'val')
        for name in args.experiments:
            rows.extend(benchmark_model(settings,name,args.seed,args.device,args.repetitions,files,session))
            csv_file(out/'benchmark_inferencia.csv',rows)
        csv_file(out/'benchmark_inferencia_resumen.csv',summarize_benchmark(rows))
        env = environment()
        commit = subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=ROOT,text=True,encoding='utf-8').strip()
        write_json(out/'benchmark_inferencia_config.json',{
            'split':'val','images':len(files),'seed':args.seed,'repetitions':args.repetitions,
            'batch':1,'precision':'FP32','warmup_images':3,'code_commit':commit,
            'imgsz':{name:config(name)['imgsz'] for name in args.experiments},
            'prediction_confidence':{name:config(name)['prediction_conf'] for name in args.experiments},
            'nms_iou':{name:config(name)['nms_iou'] for name in args.experiments},
            'gpu':env['gpu'],'python':sys.version.split()[0],'torch':env['packages']['torch'],'cuda':env.get('cuda'),
            'timed_stages':['image_read','preprocess','inference','postprocess'],
            'memory':'peak_allocated_by_pytorch_mib'})
    print(f'Ronda completada. Resultados: {out}')


if __name__ == '__main__':
    main()
