#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/qwen2.5-0.5b-instruct}"
SFT_ADAPTER="${SFT_ADAPTER:-outputs/checkpoints/qwen2_5_0_5b_lora_sft_step200}"
ROLLOUT_TASKS="${ROLLOUT_TASKS:-tasks/train_tasks.jsonl}"
ROLLOUT_LIMIT="${ROLLOUT_LIMIT:-100}"
GROUP_SIZE="${GROUP_SIZE:-4}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"

ROLLOUT_FILE="${ROLLOUT_FILE:-outputs/rollouts/grpo_sft_step200_train_group${GROUP_SIZE}_task${ROLLOUT_LIMIT}.jsonl}"
ROLLOUT_REPORT="${ROLLOUT_REPORT:-outputs/eval_reports/grpo_sft_step200_train_group${GROUP_SIZE}_task${ROLLOUT_LIMIT}_report.json}"

python training/grpo_rollout.py \
  --model "${MODEL_DIR}" \
  --adapter "${SFT_ADAPTER}" \
  --tasks "${ROLLOUT_TASKS}" \
  --limit "${ROLLOUT_LIMIT}" \
  --group-size "${GROUP_SIZE}" \
  --temperature "${TEMPERATURE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output "${ROLLOUT_FILE}" \
  --report "${ROLLOUT_REPORT}" \
  --incremental \
  --resume \
  --report-every 5
