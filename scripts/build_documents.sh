#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v latexmk >/dev/null || { echo 'Instale una distribución LaTeX con latexmk o use Overleaf.'; exit 1; }
(cd presentation && latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=Equipo4_Slides slides.tex)
