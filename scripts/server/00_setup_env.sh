#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements-model.txt

python -m pytest -q
python -m compileall training eval rollout scripts tests

echo "Environment ready. Activate it with: source ${VENV_DIR}/bin/activate"
