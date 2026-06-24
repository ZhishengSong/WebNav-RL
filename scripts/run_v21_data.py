from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.evaluate import write_report
from pages.v21_generator import generate_v21_pages
from rollout.rollout_runner import run_expert
from scripts.run_v2_data import build_report
from tasks.v2_task_generator import generate_v2_tasks, write_v2_tasks
from training.build_sft_data import build_sft_data


def element_position(page: dict[str, Any], element_id: str) -> int:
    for index, element in enumerate(page.get("elements", []), start=1):
        if element["element_id"] == element_id:
            return index
    raise ValueError(f"Element {element_id} is missing from page {page.get('page_id')}")


def target_position_audit(
    tasks: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    template_positions: dict[str, Counter[int]] = defaultdict(Counter)
    answer_positions: dict[tuple[str, str], set[int]] = defaultdict(set)
    filtered_tasks = 0
    for task in tasks:
        clicks = task["expert_clicks"]
        if len(clicks) < 2:
            continue
        filtered_tasks += 1
        page_id = task["start_page"]
        for click_index, element_id in enumerate(clicks):
            page = metadata[page_id]
            position = element_position(page, element_id)
            if click_index == len(clicks) - 1:
                template_positions[task["template"]][position] += 1
                answer_positions[(task["template"], task["target_answer"])].add(position)
            element = next(item for item in page["elements"] if item["element_id"] == element_id)
            page_id = element["target_page"]

    by_template = {}
    for template, positions in sorted(template_positions.items()):
        total = sum(positions.values())
        by_template[template] = {
            "total": total,
            "position_counts": {str(key): value for key, value in sorted(positions.items())},
            "unique_positions": len(positions),
            "max_position_share": max(positions.values()) / total if total else 0.0,
        }
    answer_coverages = [len(positions) for positions in answer_positions.values()]
    return {
        "filtered_tasks": filtered_tasks,
        "templates": len(template_positions),
        "max_template_position_share": max(
            (row["max_position_share"] for row in by_template.values()),
            default=0.0,
        ),
        "min_answer_unique_positions": min(answer_coverages) if answer_coverages else 0,
        "mean_answer_unique_positions": statistics.mean(answer_coverages) if answer_coverages else 0.0,
        "by_template": by_template,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate V2.1 candidate-shuffled pages, tasks, and SFT data.")
    parser.add_argument("--train-tasks", type=int, default=6000)
    parser.add_argument("--eval-tasks", type=int, default=1000)
    parser.add_argument("--train-instances", type=int, default=20)
    parser.add_argument("--eval-instances", type=int, default=5)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--report", default="outputs/eval_reports/v21_data_report.json")
    args = parser.parse_args()

    metadata, contexts, page_manifest = generate_v21_pages(
        seed=args.seed,
        train_instances=args.train_instances,
        eval_instances=args.eval_instances,
    )
    train_tasks, eval_tasks, task_manifest = generate_v2_tasks(
        contexts,
        train_count=args.train_tasks,
        eval_count=args.eval_tasks,
        seed=args.seed + 6,
        dataset_version="v2.1",
        task_prefix="v21",
        spread_templates_across_contexts=True,
    )
    write_v2_tasks(train_tasks, eval_tasks, task_manifest, output_dir="tasks/v21")

    train_position_audit = target_position_audit(train_tasks, metadata)
    eval_position_audit = target_position_audit(eval_tasks, metadata)
    if train_position_audit["max_template_position_share"] > 0.4:
        raise RuntimeError("V2.1 train target positions remain too concentrated")
    if train_position_audit["min_answer_unique_positions"] < 2:
        raise RuntimeError("V2.1 counterfactual answers do not cover multiple target positions")

    metadata_path = "pages/generated_pages_v21/metadata.json"
    train_trajectories = run_expert(
        "tasks/v21/train_tasks.jsonl",
        "outputs/trajectories/v21_expert_train_trajectories.jsonl",
        metadata_path=metadata_path,
    )
    eval_trajectories = run_expert(
        "tasks/v21/eval_tasks.jsonl",
        "outputs/trajectories/v21_expert_eval_trajectories.jsonl",
        metadata_path=metadata_path,
    )
    train_sft = build_sft_data(
        "outputs/trajectories/v21_expert_train_trajectories.jsonl",
        "training/v21/sft_train.jsonl",
    )
    eval_sft = build_sft_data(
        "outputs/trajectories/v21_expert_eval_trajectories.jsonl",
        "training/v21/sft_eval.jsonl",
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
    report["version"] = "v2.1"
    report["train_position_audit"] = train_position_audit
    report["eval_position_audit"] = eval_position_audit
    report["paths"] = {
        "metadata": metadata_path,
        "train_tasks": "tasks/v21/train_tasks.jsonl",
        "eval_tasks": "tasks/v21/eval_tasks.jsonl",
        "train_trajectories": "outputs/trajectories/v21_expert_train_trajectories.jsonl",
        "eval_trajectories": "outputs/trajectories/v21_expert_eval_trajectories.jsonl",
        "train_sft": "training/v21/sft_train.jsonl",
        "eval_sft": "training/v21/sft_eval.jsonl",
    }
    if report["summary"]["expert_success_rate"] != 1.0:
        raise RuntimeError("V2.1 expert verification failed")
    if report["summary"]["expert_path_target_matches"] != report["summary"]["num_tasks"]:
        raise RuntimeError("V2.1 expert path target verification failed")
    if report["summary"]["invalid_actions"] or report["summary"]["action_errors"]:
        raise RuntimeError("V2.1 expert produced invalid actions")
    write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
