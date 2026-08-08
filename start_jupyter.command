#!/bin/bash
set -e
cd "$(dirname "$0")"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate ensanut-hba1c || true
fi
python -m jupyter lab
