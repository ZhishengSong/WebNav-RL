from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.error_analysis import classify_failure
from eval.evaluate import load_jsonl, write_report
from rollout.trajectory import save_jsonl
from tasks.task_loader import load_tasks


def click_path(row: dict[str, Any]) -> list[str]:
    return [
        action.get("arguments", {}).get("element_id")
        for action in row.get("actions", [])
        if action.get("tool_name") == "click"
    ]


def path_prefix_score(model_clicks: list[str], expert_clicks: list[str]) -> float:
    if not expert_clicks:
        return 1.0 if not model_clicks else 0.0
    matches = 0
    for model_click, expert_click in zip(model_clicks, expert_clicks):
        if model_click != expert_click:
            break
        matches += 1
    return matches / len(expert_clicks)


def compute_reward(row: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    summary = row["summary"]
    model_steps = int(summary.get("model_steps", 0))
    format_errors = int(summary.get("format_errors", 0))
    invalid_tool_calls = int(summary.get("invalid_tool_calls", 0))
    success = bool(summary.get("success"))

    format_reward = 0.2 if model_steps > 0 and format_errors == 0 else 0.0
    answer_reward = 0.4 if success else 0.0

    model_clicks = [item for item in click_path(row) if isinstance(item, str)]
    expert_clicks = task.get("expert_clicks", [])
    path_score = path_prefix_score(model_clicks, expert_clicks)
    path_reward = 0.3 * path_score

    step_penalty = -0.05 * max(0, model_steps - len(expert_clicks) - 2)
    invalid_penalty = -0.2 * invalid_tool_calls
    no_submit_penalty = 0.0 if summary.get("termination") == "submitted" else -0.2

    total_reward = format_reward + answer_reward + path_reward + step_penalty + invalid_penalty + no_submit_penalty
    error_type, error_reason = classify_failure(row, task)
    return {
        "task_id": row["task_id"],
        "success": success,
        "error_type": error_type,
        "error_reason": error_reason,
        "format_reward": round(format_reward, 6),
        "path_reward": round(path_reward, 6),
        "path_score": round(path_score, 6),
        "answer_reward": round(answer_reward, 6),
        "step_penalty": round(step_penalty, 6),
        "invalid_penalty": round(invalid_penalty, 6),
        "no_submit_penalty": round(no_submit_penalty, 6),
        "total_reward": round(total_reward, 6),
        "model_steps": model_steps,
        "expert_steps": len(expert_clicks) + 2,
        "invalid_tool_calls": invalid_tool_calls,
        "model_clicks": model_clicks,
        "expert_clicks": expert_clicks,
        "final_answer": summary.get("final_answer"),
        "target_answer": summary.get("target_answer"),
        "template": task.get("template"),
        "difficulty": task.get("difficulty"),
        "page_type": task.get("page_type"),
    }


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def summarize_rewards(rewards: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [row["total_reward"] for row in rewards]
    successes = [row for row in rewards if row["success"]]
    failures = [row for row in rewards if not row["success"]]

    by_error: dict[str, list[float]] = defaultdict(list)
    by_template: dict[str, list[float]] = defaultdict(list)
    by_difficulty: dict[str, list[float]] = defaultdict(list)
    for row in rewards:
        by_error[row["error_type"]].append(row["total_reward"])
        by_template[row.get("template") or "unknown"].append(row["total_reward"])
        by_difficulty[row.get("difficulty") or "unknown"].append(row["total_reward"])

    return {
        "num_rewards": len(rewards),
        "mean_total_reward": mean(totals),
        "min_total_reward": min(totals) if totals else 0.0,
        "max_total_reward": max(totals) if totals else 0.0,
        "success_count": len(successes),
        "failure_count": len(failures),
        "success_mean_reward": mean([row["total_reward"] for row in successes]),
        "failure_mean_reward": mean([row["total_reward"] for row in failures]),
        "mean_format_reward": mean([row["format_reward"] for row in rewards]),
        "mean_path_reward": mean([row["path_reward"] for row in rewards]),
        "mean_answer_reward": mean([row["answer_reward"] for row in rewards]),
        "mean_step_penalty": mean([row["step_penalty"] for row in rewards]),
        "mean_invalid_penalty": mean([row["invalid_penalty"] for row in rewards]),
        "by_error_type": {
            key: {"count": len(values), "mean_reward": mean(values)}
            for key, values in sorted(by_error.items())
        },
        "by_template": {
            key: {"count": len(values), "mean_reward": mean(values)}
            for key, values in sorted(by_template.items())
        },
        "by_difficulty": {
            key: {"count": len(values), "mean_reward": mean(values)}
            for key, values in sorted(by_difficulty.items())
        },
    }


def build_reward_report(
    trajectory_path: str | Path,
    tasks_path: str | Path,
    output_path: str | Path | None = None,
    breakdown_path: str | Path | None = None,
) -> dict[str, Any]:
    trajectories = load_jsonl(trajectory_path)
    tasks = {task["task_id"]: task for task in load_tasks(tasks_path)}
    rewards = [compute_reward(row, tasks[row["task_id"]]) for row in trajectories]
    report = {
        "trajectory_path": str(trajectory_path),
        "tasks_path": str(tasks_path),
        "reward_formula": {
            "format_reward": "+0.2 if all model outputs parse as tool calls",
            "path_reward": "+0.3 * prefix_match(model_clicks, expert_clicks)",
            "answer_reward": "+0.4 if final answer exact matches target",
            "step_penalty": "-0.05 per model step beyond expert_clicks + open + submit",
            "invalid_penalty": "-0.2 per invalid tool call",
            "no_submit_penalty": "-0.2 if episode does not submit",
        },
        "summary": summarize_rewards(rewards),
    }
    if output_path is not None:
        write_report(output_path, report)
    if breakdown_path is not None:
        save_jsonl(breakdown_path, rewards)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute reward breakdowns for WebNav trajectories.")
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--tasks", default="tasks/eval_tasks.jsonl")
    parser.add_argument("--output", default="outputs/eval_reports/reward_report.json")
    parser.add_argument("--breakdown", default="outputs/eval_reports/reward_breakdown.jsonl")
    args = parser.parse_args()
    report = build_reward_report(
        args.trajectories,
        args.tasks,
        output_path=args.output,
        breakdown_path=args.breakdown,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
