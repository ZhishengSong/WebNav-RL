#!/usr/bin/env bash
set -euo pipefail

bash scripts/server/01_download_model.sh
bash scripts/server/02_train_sft.sh
bash scripts/server/03_collect_grpo_rollouts.sh
bash scripts/server/04_train_grpo_kl.sh
bash scripts/server/05_eval_compare.sh
