from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pages.page_generator import generate
from rollout.rollout_runner import run_expert
from training.build_sft_data import build_sft_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tasks", type=int, default=1000)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    generate(num_tasks=args.num_tasks, train_ratio=args.train_ratio, seed=args.seed)
    train_traj = run_expert("tasks/train_tasks.jsonl", "outputs/trajectories/expert_train_trajectories.jsonl")
    eval_traj = run_expert("tasks/eval_tasks.jsonl", "outputs/trajectories/expert_eval_trajectories.jsonl")
    train_sft = build_sft_data("outputs/trajectories/expert_train_trajectories.jsonl", "training/sft_train.jsonl")
    eval_sft = build_sft_data("outputs/trajectories/expert_eval_trajectories.jsonl", "training/sft_eval.jsonl")

    all_traj = train_traj + eval_traj
    report = {
        "num_tasks": len(all_traj),
        "train_tasks": len(train_traj),
        "eval_tasks": len(eval_traj),
        "train_sft_examples": len(train_sft),
        "eval_sft_examples": len(eval_sft),
        "success_rate": sum(row["summary"]["success"] for row in all_traj) / len(all_traj),
        "invalid_actions": sum(row["summary"]["invalid_actions"] for row in all_traj),
        "avg_steps": sum(row["summary"]["steps"] for row in all_traj) / len(all_traj),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
