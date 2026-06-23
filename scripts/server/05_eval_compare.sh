#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/qwen2.5-0.5b-instruct}"
SFT_ADAPTER="${SFT_ADAPTER:-outputs/checkpoints/qwen2_5_0_5b_lora_sft_step200}"
ROLLOUT_LIMIT="${ROLLOUT_LIMIT:-100}"
GROUP_SIZE="${GROUP_SIZE:-4}"
EVAL_LIMIT="${EVAL_LIMIT:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"

GRPO_ADAPTER="${GRPO_ADAPTER:-outputs/checkpoints/qwen2_5_0_5b_lora_grpo_kl_task${ROLLOUT_LIMIT}_group${GROUP_SIZE}_step100}"

SFT_TRAJ="${SFT_TRAJ:-outputs/trajectories/server_sft_eval${EVAL_LIMIT}_trajectories.jsonl}"
SFT_REPORT="${SFT_REPORT:-outputs/eval_reports/server_sft_eval${EVAL_LIMIT}_report.json}"
SFT_FAILURES="${SFT_FAILURES:-outputs/eval_reports/server_sft_eval${EVAL_LIMIT}_failures.jsonl}"

GRPO_TRAJ="${GRPO_TRAJ:-outputs/trajectories/server_grpo_kl_eval${EVAL_LIMIT}_trajectories.jsonl}"
GRPO_REPORT="${GRPO_REPORT:-outputs/eval_reports/server_grpo_kl_eval${EVAL_LIMIT}_report.json}"
GRPO_FAILURES="${GRPO_FAILURES:-outputs/eval_reports/server_grpo_kl_eval${EVAL_LIMIT}_failures.jsonl}"

python scripts/run_eval.py \
  --model "${MODEL_DIR}" \
  --adapter "${SFT_ADAPTER}" \
  --tasks tasks/eval_tasks.jsonl \
  --limit "${EVAL_LIMIT}" \
  --output "${SFT_TRAJ}" \
  --report "${SFT_REPORT}" \
  --failures "${SFT_FAILURES}" \
  --device auto \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature 0.0 \
  --incremental \
  --resume \
  --report-every 10

python scripts/run_eval.py \
  --model "${MODEL_DIR}" \
  --adapter "${GRPO_ADAPTER}" \
  --tasks tasks/eval_tasks.jsonl \
  --limit "${EVAL_LIMIT}" \
  --output "${GRPO_TRAJ}" \
  --report "${GRPO_REPORT}" \
  --failures "${GRPO_FAILURES}" \
  --device auto \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature 0.0 \
  --incremental \
  --resume \
  --report-every 10

python scripts/compare_eval_reports.py \
  --baseline "${SFT_REPORT}" \
  --candidate "${GRPO_REPORT}" \
  --baseline-name "SFT step200" \
  --candidate-name "GRPO-KL" \
  --output-json outputs/eval_reports/server_sft_vs_grpo_kl_comparison.json \
  --output-md outputs/eval_reports/server_sft_vs_grpo_kl_comparison.md
