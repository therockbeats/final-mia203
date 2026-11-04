#!/usr/bin/env bash
# Prepara los archivos; --push envía los commits al repositorio remoto.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 1 ]]; then echo 'Uso: bash scripts/publish.sh URL_GITHUB [--push]'; exit 1; fi
remote_url="$1"
[[ "$remote_url" =~ ^https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(\.git)?$ ]] || { echo 'Indique una URL HTTPS de GitHub sin credenciales.'; exit 1; }
python3 scripts/check_repository.py
if [[ ! -d .git ]]; then git init -b main; fi
if git remote get-url origin >/dev/null 2>&1; then
  [[ "$(git remote get-url origin)" == "$remote_url" ]] || { echo 'origin ya apunta a otra dirección; revíselo.'; exit 1; }
else
  git remote add origin "$remote_url"
fi
git add --all
python3 scripts/check_repository.py
if [[ "${2:-}" != '--push' ]]; then
  echo 'Archivos preparados. Revise git diff --cached y cree su commit.'
  exit 0
fi
git diff --cached --quiet || { echo 'Hay cambios preparados sin commit. Revise y cree el commit antes de publicar.'; exit 1; }
git push --set-upstream origin main
