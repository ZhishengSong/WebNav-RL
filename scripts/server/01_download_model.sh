#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_DIR="${MODEL_DIR:-models/qwen2.5-0.5b-instruct}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

python scripts/download_hf_model.py \
  --repo-id "${MODEL_REPO}" \
  --local-dir "${MODEL_DIR}"
