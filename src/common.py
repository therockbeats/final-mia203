from pathlib import Path
import hashlib
import json
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
YOLO_COMMIT = '915bbf294bb74c859f0b41f1c23bc395014ea679'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def paths():
    p = read_json(ROOT / 'configs/paths.local.json')
    return {k: Path(v).expanduser().resolve() for k, v in p.items()}


def config(name):
    return read_json(ROOT / f'configs/{name}.json')


def run(command, **kwargs):
    return subprocess.run([str(x) for x in command], check=True, **kwargs)


def fingerprint(path):
    return sha256(path)
