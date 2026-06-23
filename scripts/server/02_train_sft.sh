#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/qwen2.5-0.5b-instruct}"
SFT_ADAPTER="${SFT_ADAPTER:-outputs/checkpoints/qwen2_5_0_5b_lora_sft_step200}"
SFT_STEPS="${SFT_STEPS:-200}"
SFT_LR="${SFT_LR:-2e-4}"

python training/sft_train.py \
  --model "${MODEL_DIR}" \
  --train-data training/sft_train.jsonl \
  --output-dir "${SFT_ADAPTER}" \
  --epochs 1 \
  --max-steps "${SFT_STEPS}" \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate "${SFT_LR}" \
  --max-seq-len 1024 \
  --log-every 10
