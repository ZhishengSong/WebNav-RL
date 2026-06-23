from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.evaluate import load_jsonl, resolve_path, write_report
from rollout.parser import ToolCallParseError, parse_tool_call
from rollout.trajectory import save_jsonl
from tasks.task_loader import load_tasks


def action_clicks(row: dict[str, Any]) -> list[str]:
    clicks = []
    for action in row.get("actions", []):
        if action.get("tool_name") == "click":
            element_id = action.get("arguments", {}).get("element_id")
            if isinstance(element_id, str):
                clicks.append(element_id)
    return clicks


def submitted_too_early(row: dict[str, Any]) -> bool:
    actions = row.get("actions", [])
    submit_index = next(
        (idx for idx, action in enumerate(actions) if action.get("tool_name") == "submit_answer"),
        None,
    )
    if submit_index is None:
        return False
    return not any(action.get("tool_name") == "click" for action in actions[:submit_index])


def invalid_action_messages(row: dict[str, Any]) -> list[str]:
    return [
        action.get("observation", "")
        for action in row.get("actions", [])
        if action.get("status") == "error"
    ]


def parsed_tool_names(row: dict[str, Any]) -> list[str]:
    names = []
    for output in row.get("raw_outputs", []):
        try:
            names.append(parse_tool_call(output).name)
        except ToolCallParseError:
            continue
    return names


def classify_failure(row: dict[str, Any], task: dict[str, Any] | None) -> tuple[str, str]:
    summary = row["summary"]
    if summary["success"]:
        return "success", "episode succeeded"
    if summary.get("format_errors", 0) > 0:
        first_error = row.get("parser_errors", [{}])[0]
        return "format_error", str(first_error.get("code", "unknown parser error"))
    if summary.get("invalid_tool_calls", 0) > 0 or invalid_action_messages(row):
        messages = invalid_action_messages(row)
        detail = messages[0] if messages else "tool call parsed but environment rejected it"
        return "invalid_tool_call", detail
    if summary.get("termination") != "submitted":
        return "max_steps_no_submit", f"terminated as {summary.get('termination')}"
    if summary.get("final_answer") is None:
        return "missing_final_answer", "submitted no answer or no submit action"
    if submitted_too_early(row):
        return "premature_submit", "submitted before any click"

    if task is not None:
        model_clicks = action_clicks(row)
        expert_clicks = task.get("expert_clicks", [])
        if model_clicks != expert_clicks:
            if model_clicks and expert_clicks and model_clicks[0] == expert_clicks[0]:
                return "wrong_candidate_after_filter", f"model_clicks={model_clicks}; expert_clicks={expert_clicks}"
            return "wrong_click_path", f"model_clicks={model_clicks}; expert_clicks={expert_clicks}"

    tool_names = parsed_tool_names(row)
    if tool_names and tool_names[-1] == "submit_answer":
        return "wrong_final_answer", (
            f"final={summary.get('final_answer')}; target={summary.get('target_answer')}"
        )
    return "other_failure", "unclassified failure"


def build_error_analysis(
    trajectory_path: str | Path,
    tasks_path: str | Path,
    output_path: str | Path | None = None,
    examples_path: str | Path | None = None,
    max_examples_per_type: int = 3,
) -> dict[str, Any]:
    trajectories = load_jsonl(trajectory_path)
    tasks = {task["task_id"]: task for task in load_tasks(tasks_path)}

    error_counts: Counter[str] = Counter()
    page_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    difficulty_counts: dict[str, Counter[str]] = defaultdict(Counter)
    template_counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    rows = []
    for row in trajectories:
        task = tasks.get(row["task_id"])
        error_type, reason = classify_failure(row, task)
        error_counts[error_type] += 1
        if task is not None:
            page_type_counts[task.get("page_type", "unknown")][error_type] += 1
            difficulty_counts[task.get("difficulty", "unknown")][error_type] += 1
            template_counts[task.get("template", "unknown")][error_type] += 1
        record = {
            "task_id": row["task_id"],
            "instruction": row["instruction"],
            "error_type": error_type,
            "reason": reason,
            "success": row["summary"]["success"],
            "final_answer": row["summary"].get("final_answer"),
            "target_answer": row["summary"].get("target_answer"),
            "model_clicks": action_clicks(row),
            "expert_clicks": task.get("expert_clicks", []) if task is not None else [],
            "template": task.get("template") if task is not None else None,
            "difficulty": task.get("difficulty") if task is not None else None,
            "page_type": task.get("page_type") if task is not None else None,
        }
        rows.append(record)
        if error_type != "success" and len(examples[error_type]) < max_examples_per_type:
            examples[error_type].append(record)

    total = len(trajectories)
    failures = total - error_counts["success"]
    report = {
        "trajectory_path": str(trajectory_path),
        "tasks_path": str(tasks_path),
        "num_trajectories": total,
        "successes": error_counts["success"],
        "failures": failures,
        "success_rate": error_counts["success"] / total if total else 0.0,
        "failure_rate": failures / total if total else 0.0,
        "error_counts": dict(error_counts),
        "error_rates": {key: value / total for key, value in error_counts.items()} if total else {},
        "by_page_type": {key: dict(value) for key, value in page_type_counts.items()},
        "by_difficulty": {key: dict(value) for key, value in difficulty_counts.items()},
        "by_template": {key: dict(value) for key, value in template_counts.items()},
        "examples": dict(examples),
    }

    if output_path is not None:
        write_report(output_path, report)
    if examples_path is not None:
        save_jsonl(examples_path, rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify WebNav rollout failures.")
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--tasks", default="tasks/eval_tasks.jsonl")
    parser.add_argument("--output", default="outputs/eval_reports/error_analysis.json")
    parser.add_argument("--examples", default="outputs/eval_reports/error_analysis_examples.jsonl")
    parser.add_argument("--max-examples-per-type", type=int, default=3)
    args = parser.parse_args()
    report = build_error_analysis(
        args.trajectories,
        args.tasks,
        output_path=args.output,
        examples_path=args.examples,
        max_examples_per_type=args.max_examples_per_type,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
