from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.evaluate import write_report
from pages.v2_generator import generate_v2_pages
from rollout.rollout_runner import run_expert
from tasks.v2_task_generator import generate_v2_tasks, write_v2_tasks
from training.build_sft_data import build_sft_data


def build_report(
    metadata: dict[str, Any],
    page_manifest: dict[str, Any],
    task_manifest: dict[str, Any],
    train_trajectories: list[dict[str, Any]],
    eval_trajectories: list[dict[str, Any]],
    train_sft: list[dict[str, Any]],
    eval_sft: list[dict[str, Any]],
) -> dict[str, Any]:
    trajectories = train_trajectories + eval_trajectories
    successes = sum(bool(row["summary"]["success"]) for row in trajectories)
    invalid_actions = sum(int(row["summary"].get("invalid_actions", 0)) for row in trajectories)
    action_errors = sum(
        action.get("status") == "error"
        for row in trajectories
        for action in row.get("actions", [])
    )
    train_next_actions = sum(
        message["role"] == "assistant"
        for row in train_trajectories
        for message in row["messages"]
    )
    eval_next_actions = sum(
        message["role"] == "assistant"
        for row in eval_trajectories
        for message in row["messages"]
    )
    path_target_matches = sum(
        metadata[row["actions"][-1]["current_page"]].get("answer") == row["summary"]["target_answer"]
        for row in trajectories
    )
    return {
        "version": "v2",
        "page_manifest": page_manifest,
        "task_manifest": task_manifest,
        "summary": {
            "num_tasks": len(trajectories),
            "train_tasks": len(train_trajectories),
            "eval_tasks": len(eval_trajectories),
            "expert_successes": successes,
            "expert_success_rate": successes / len(trajectories) if trajectories else 0.0,
            "invalid_actions": invalid_actions,
            "action_errors": action_errors,
            "expert_path_target_matches": path_target_matches,
            "average_steps": (
                sum(row["summary"]["steps"] for row in trajectories) / len(trajectories)
                if trajectories
                else 0.0
            ),
            "train_sft_trajectories": len(train_sft),
            "eval_sft_trajectories": len(eval_sft),
            "train_next_action_examples": train_next_actions,
            "eval_next_action_examples": eval_next_actions,
            "templates": len(
                set(task_manifest["train_template_counts"]).union(task_manifest["eval_template_counts"])
            ),
        },
        "paths": {
            "metadata": "pages/generated_pages_v2/metadata.json",
            "train_tasks": "tasks/v2/train_tasks.jsonl",
            "eval_tasks": "tasks/v2/eval_tasks.jsonl",
            "train_trajectories": "outputs/trajectories/v2_expert_train_trajectories.jsonl",
            "eval_trajectories": "outputs/trajectories/v2_expert_eval_trajectories.jsonl",
            "train_sft": "training/v2/sft_train.jsonl",
            "eval_sft": "training/v2/sft_eval.jsonl",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and verify the V2 structural-generalization dataset.")
    parser.add_argument("--train-tasks", type=int, default=3000)
    parser.add_argument("--eval-tasks", type=int, default=500)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--report", default="outputs/eval_reports/v2_data_report.json")
    args = parser.parse_args()

    metadata, contexts, page_manifest = generate_v2_pages(seed=args.seed)
    train_tasks, eval_tasks, task_manifest = generate_v2_tasks(
        contexts,
        train_count=args.train_tasks,
        eval_count=args.eval_tasks,
        seed=args.seed + 6,
    )
    write_v2_tasks(train_tasks, eval_tasks, task_manifest)

    metadata_path = "pages/generated_pages_v2/metadata.json"
    train_trajectories = run_expert(
        "tasks/v2/train_tasks.jsonl",
        "outputs/trajectories/v2_expert_train_trajectories.jsonl",
        metadata_path=metadata_path,
    )
    eval_trajectories = run_expert(
        "tasks/v2/eval_tasks.jsonl",
        "outputs/trajectories/v2_expert_eval_trajectories.jsonl",
        metadata_path=metadata_path,
    )
    train_sft = build_sft_data(
        "outputs/trajectories/v2_expert_train_trajectories.jsonl",
        "training/v2/sft_train.jsonl",
    )
    eval_sft = build_sft_data(
        "outputs/trajectories/v2_expert_eval_trajectories.jsonl",
        "training/v2/sft_eval.jsonl",
    )
    report = build_report(
        metadata,
        page_manifest,
        task_manifest,
        train_trajectories,
        eval_trajectories,
        train_sft,
        eval_sft,
    )
    if report["summary"]["expert_success_rate"] != 1.0:
        raise RuntimeError("V2 expert verification failed; inspect the generated trajectories.")
    if report["summary"]["invalid_actions"] or report["summary"]["action_errors"]:
        raise RuntimeError("V2 expert produced invalid actions.")
    if report["summary"]["expert_path_target_matches"] != report["summary"]["num_tasks"]:
        raise RuntimeError("V2 expert path ended on a detail page that does not match the target.")
    write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
