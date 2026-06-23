#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/qwen2.5-0.5b-instruct}"
SFT_ADAPTER="${SFT_ADAPTER:-outputs/checkpoints/qwen2_5_0_5b_lora_sft_step200}"
ROLLOUT_LIMIT="${ROLLOUT_LIMIT:-100}"
GROUP_SIZE="${GROUP_SIZE:-4}"
ROLLOUT_FILE="${ROLLOUT_FILE:-outputs/rollouts/grpo_sft_step200_group${GROUP_SIZE}_task${ROLLOUT_LIMIT}.jsonl}"

GRPO_ADAPTER="${GRPO_ADAPTER:-outputs/checkpoints/qwen2_5_0_5b_lora_grpo_kl_task${ROLLOUT_LIMIT}_group${GROUP_SIZE}_step100}"
GRPO_STEPS="${GRPO_STEPS:-100}"
GRPO_LR="${GRPO_LR:-1e-5}"
KL_BETA="${KL_BETA:-0.02}"

python training/grpo_train.py \
  --model "${MODEL_DIR}" \
  --adapter "${SFT_ADAPTER}" \
  --reference-adapter "${SFT_ADAPTER}" \
  --rollouts "${ROLLOUT_FILE}" \
  --output-dir "${GRPO_ADAPTER}" \
  --epochs 1 \
  --max-steps "${GRPO_STEPS}" \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --learning-rate "${GRPO_LR}" \
  --kl-beta "${KL_BETA}" \
  --clip-advantage 1.0 \
  --max-seq-len 1024 \
  --log-every 5
