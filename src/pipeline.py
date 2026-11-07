"""Interfaz de línea de comandos del proyecto."""

""" Este script centraliza todas las operaciones del flujo de trabajo experimental:
configuración de rutas, extracción y auditoría de datos, entrenamiento, 
evaluación, generación de figuras y congelamiento de protocolos para 
asegurar la reproducibilidad.
"""
import argparse
import sys
from pathlib import Path

""" Importación de utilidades locales (configuración, rutas, hashing, etc.)"""

from .common import ROOT, paths, config, read_json, write_json, sha256, now


def configure(args):
     """
    Configura las rutas locales del proyecto (dataset, pesos, YOLOv5).
    Genera un archivo configs/paths.local.json para evitar hardcodear rutas.
    """
    
    base = Path(args.work_dir).expanduser().resolve()

   """ Diccionario con las rutas absolutas requeridas por el pipeline"""

    value = {'dataset': str(base/'data/processed'), 'runs': str(base/'runs'),
        'yolo': str(base/'external/yolov5'), 'initial_weights': str(base/'weights/yolov5s.pt'),
        'audit': str(base/'data/audit.json')}
    dest = ROOT/'configs/paths.local.json'

    """ Protección para no sobrescribir configuraciones existentes accidentalmente """
    if dest.exists():
        raise FileExistsError('La configuración local ya existe; editarla explícitamente')
    write_json(dest, value)
    print(dest)


def freeze(args):

    """
    Congela el protocolo experimental antes de evaluar en Test.
    
    Esta función es crucial para evitar la fuga de datos (data leakage). 
    Verifica que las predicciones de validación (usadas para elegir el modelo)
    coincidan con las firmas SHA256 actuales. Una vez congelado, el modelo 
    no puede ser reajustado con el conjunto de prueba.
    """ 
    from .inference import signature
    p = paths()
    dest = ROOT/'configs/frozen_protocol.json'
    if dest.exists():
        raise FileExistsError('Protocolo ya congelado; conservar su historial para cambiarlo')
    experiments = {}
    for seed in args.seeds:
        for name in getattr(args, 'experiments', ['M2']):
            m = read_json(p['runs']/name/f'seed{seed}'/'val_predictions/full/metrics.json')
            sig = signature(p, name, seed)
            if m['signature'] != sig:
                raise ValueError(f'Validación desactualizada: {name} semilla {seed}')
            experiments[f'{name}/seed{seed}'] = sig
    write_json(dest, {'frozen_at': now(), 'selection_notes': args.rationale,
        'experiments': experiments, 'evaluation_split': 'test'})
    print(dest)


def main():
    parser = argparse.ArgumentParser(description='Proyecto final MIA-203 — Grupo 4')
    sub = parser.add_subparsers(dest='command', required=True)

    """1. Configuración y preparación de datos """
    a = sub.add_parser('configure'); a.add_argument('--work-dir', required=True)
    a = sub.add_parser('extract'); a.add_argument('--zip', required=True)
    sub.add_parser('audit'); sub.add_parser('prepare'); sub.add_parser('doctor')

    """2. Importación de experimentos previos (B1, M1, M2) """
    for name in ['import-b1', 'import-m1', 'import-m2']:
        a = sub.add_parser(name); a.add_argument('--source', required=True)

    """3. Entrenamiento del modelo """
    a = sub.add_parser('train'); a.add_argument('experiment', choices=['B1','M1','M2']); a.add_argument('--seed', type=int, default=0)
    a.add_argument('--smoke', action='store_true'); a.add_argument('--dry-run', action='store_true')

    """4. Inferencia y Evaluación """
    a = sub.add_parser('predict'); a.add_argument('experiment', choices=['B1','M1','M2'])
    a.add_argument('--split', choices=['val','test'], default='val'); a.add_argument('--seed', type=int, default=0); a.add_argument('--limit', type=int, default=0)
    a = sub.add_parser('evaluate'); a.add_argument('experiment', choices=['B1','M1','M2'])
    a.add_argument('--split', choices=['val','test'], default='val'); a.add_argument('--seed', type=int, default=0)


    """5. Congelamiento de selección (Validation -> Test) """
    a = sub.add_parser('freeze'); a.add_argument('--seeds', nargs='+', type=int, default=[0]); a.add_argument('--rationale', default=''); a.add_argument('--experiments', nargs='+', choices=['B1','M1','M2'], default=['M2'])
    
    """6. Reportes y Figuras """
    for cmd in ['compare','figures']:
        a = sub.add_parser(cmd); a.add_argument('--split', choices=['val','test'], default='val'); a.add_argument('--experiments', nargs='+', choices=['B1','M1','M2'], default=['B1','M1','M2'])
    a = sub.add_parser('errors'); a.add_argument('experiment', choices=['B1','M1','M2'])
    a.add_argument('--split', choices=['val','test'], default='val'); a.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'configure':
        return configure(args)
    if args.command == 'freeze':
        return freeze(args)
    p = paths()
    if args.command == 'extract':
        from .data import extract
        extract(args.zip, p['dataset'])
        write_json(p['dataset'].parent/'archive.json', {'sha256': sha256(args.zip), 'filename': Path(args.zip).name})
    elif args.command == 'audit':
        from .data import audit
        a = audit(p['dataset'], p['audit'])
        for split, values in a['splits'].items():
            print(f"{split}: {values['images']:,} imágenes, {values['labels']:,} etiquetas, {values['videos']} videos")
    elif args.command == 'prepare':
        from .runtime import prepare
        prepare(p['yolo'], p['initial_weights'])
    elif args.command == 'doctor':
        from .runtime import environment
        value = environment(); write_json(p['runs']/'environment.json', value)
        print(f"Python {value['python'].split()[0]} · PyTorch {value['packages']['torch']} · GPU: {value['gpu'] or 'CPU'}")
    elif args.command in ['import-b1', 'import-m1', 'import-m2']:
        from .reporting import import_experiment
        name = args.command.removeprefix('import-').upper()
        metadata = import_experiment(args.source, name)
        print(f"{name} incorporado: {metadata['epochs']} épocas, semilla {metadata['seed']}.")
    elif args.command == 'train':
        from .training import train
        train(args.experiment, args.seed, args.smoke, args.dry_run)
    elif args.command == 'predict':
        from .inference import predict
        print(predict(args.experiment, args.split, args.seed, args.limit))
    elif args.command == 'evaluate':
        from .evaluation import evaluate
        from .inference import signature
        folder = p['runs']/args.experiment/f'seed{args.seed}'/f'{args.split}_predictions/full'
        pred = read_json(folder/'predictions.json')
        if pred['signature'] != signature(p, args.experiment, args.seed):
            raise ValueError('Predicciones desactualizadas')
        metrics = evaluate(folder/'predictions.json', p['dataset'], folder/'metrics.json')
        for cls, values in metrics['per_class'].items():
            print(f"{cls}: P={values['precision']:.3f}, R={values['recall']:.3f}, AP50={values['map50']:.3f}, AP50:95={values['map50_95']:.3f}")
    elif args.command in ['compare', 'figures']:
        from .reporting import compare, figures
        
        if args.command == 'compare':
            rows = compare(args.split, args.experiments)
            print(f'Tabla comparativa: {len(rows)} ejecuciones.')
        else:
            figures(args.split, args.experiments)
            print('Figuras guardadas en results/' + args.split)
    elif args.command == 'errors':
        from .reporting import error_examples
        error_examples(args.experiment, args.split, args.seed)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
