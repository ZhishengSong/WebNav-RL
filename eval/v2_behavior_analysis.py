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

from eval.error_analysis import classify_failure
from eval.evaluate import load_jsonl, write_report
from tasks.task_loader import load_tasks


def load_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return json.loads(resolved.read_text(encoding="utf-8"))


def click_actions(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    return [action for action in trajectory.get("actions", []) if action.get("tool_name") == "click"]


def candidate_position(page: dict[str, Any], element_id: str | None) -> int | None:
    if element_id is None:
        return None
    for index, element in enumerate(page.get("elements", []), start=1):
        if element.get("element_id") == element_id:
            return index
    return None


def build_v2_behavior_analysis(
    trajectories_path: str | Path,
    tasks_path: str | Path,
    metadata_path: str | Path,
) -> dict[str, Any]:
    trajectories = {row["task_id"]: row for row in load_jsonl(trajectories_path)}
    tasks = {task["task_id"]: task for task in load_tasks(tasks_path)}
    metadata = load_metadata(metadata_path)
    if set(trajectories) != set(tasks):
        raise ValueError("Trajectory and task ids do not match")

    error_counts: Counter[str] = Counter()
    difficulty_counts: dict[str, Counter[str]] = defaultdict(Counter)
    template_counts: dict[str, Counter[str]] = defaultdict(Counter)
    position_counts: Counter[int | str] = Counter()
    expert_position_counts: Counter[int | str] = Counter()
    wrong_position_counts: Counter[int | str] = Counter()
    per_template_positions: dict[str, Counter[int | str]] = defaultdict(Counter)

    filtered_tasks = 0
    direct_tasks = 0
    correct_filters = 0
    candidate_attempts_after_correct_filter = 0
    correct_candidates_after_correct_filter = 0
    wrong_candidate_rows = 0
    wrong_candidate_first_position = 0
    wrong_candidate_valid_position = 0

    for task_id, task in tasks.items():
        trajectory = trajectories[task_id]
        error_type, _ = classify_failure(trajectory, task)
        error_counts[error_type] += 1
        difficulty_counts[task["difficulty"]][error_type] += 1
        template_counts[task["template"]][error_type] += 1

        expert_clicks = task.get("expert_clicks", [])
        model_actions = click_actions(trajectory)
        model_clicks = [action.get("arguments", {}).get("element_id") for action in model_actions]
        if len(expert_clicks) < 2:
            direct_tasks += 1
            continue

        filtered_tasks += 1
        if not model_clicks or model_clicks[0] != expert_clicks[0]:
            continue
        correct_filters += 1
        if len(model_clicks) < 2:
            continue
        candidate_attempts_after_correct_filter += 1

        filtered_page_id = model_actions[0].get("current_page")
        filtered_page = metadata.get(filtered_page_id, {})
        model_position = candidate_position(filtered_page, model_clicks[1])
        expert_position = candidate_position(filtered_page, expert_clicks[1])
        position_counts[model_position if model_position is not None else "invalid"] += 1
        expert_position_counts[expert_position if expert_position is not None else "invalid"] += 1
        per_template_positions[task["template"]][model_position if model_position is not None else "invalid"] += 1

        if model_clicks[1] == expert_clicks[1]:
            correct_candidates_after_correct_filter += 1
        if error_type == "wrong_candidate_after_filter":
            wrong_candidate_rows += 1
            wrong_position_counts[model_position if model_position is not None else "invalid"] += 1
            if model_position is not None:
                wrong_candidate_valid_position += 1
            if model_position == 1:
                wrong_candidate_first_position += 1

    total = len(tasks)
    successes = error_counts["success"]
    template_summary = {}
    for template, counts in sorted(template_counts.items()):
        template_total = sum(counts.values())
        positions = per_template_positions.get(template, Counter())
        candidate_attempts = sum(positions.values())
        template_summary[template] = {
            "total": template_total,
            "successes": counts["success"],
            "success_rate": counts["success"] / template_total if template_total else 0.0,
            "error_counts": dict(counts),
            "candidate_position_counts": {str(key): value for key, value in sorted(positions.items(), key=lambda x: str(x[0]))},
            "first_position_rate": positions[1] / candidate_attempts if candidate_attempts else None,
        }

    return {
        "trajectories_path": str(trajectories_path),
        "tasks_path": str(tasks_path),
        "metadata_path": str(metadata_path),
        "summary": {
            "num_tasks": total,
            "successes": successes,
            "success_rate": successes / total if total else 0.0,
            "direct_tasks": direct_tasks,
            "filtered_tasks": filtered_tasks,
            "correct_filters": correct_filters,
            "correct_filter_rate": correct_filters / filtered_tasks if filtered_tasks else 0.0,
            "candidate_attempts_after_correct_filter": candidate_attempts_after_correct_filter,
            "correct_candidates_after_correct_filter": correct_candidates_after_correct_filter,
            "candidate_accuracy_after_correct_filter": (
                correct_candidates_after_correct_filter / candidate_attempts_after_correct_filter
                if candidate_attempts_after_correct_filter
                else 0.0
            ),
            "wrong_candidate_rows": wrong_candidate_rows,
            "wrong_candidate_valid_position": wrong_candidate_valid_position,
            "wrong_candidate_first_position": wrong_candidate_first_position,
            "wrong_candidate_first_position_rate": (
                wrong_candidate_first_position / wrong_candidate_rows if wrong_candidate_rows else 0.0
            ),
        },
        "error_counts": dict(error_counts),
        "by_difficulty": {key: dict(value) for key, value in sorted(difficulty_counts.items())},
        "model_candidate_position_counts": {
            str(key): value for key, value in sorted(position_counts.items(), key=lambda x: str(x[0]))
        },
        "expert_candidate_position_counts": {
            str(key): value for key, value in sorted(expert_position_counts.items(), key=lambda x: str(x[0]))
        },
        "wrong_candidate_position_counts": {
            str(key): value for key, value in sorted(wrong_position_counts.items(), key=lambda x: str(x[0]))
        },
        "by_template": template_summary,
    }


def percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# V2 SFT Behavior Analysis",
        "",
        "## Funnel",
        "",
        f"- Overall success: **{percent(summary['success_rate'])}**",
        f"- Filtered tasks: **{summary['filtered_tasks']}**",
        f"- Correct filter rate: **{percent(summary['correct_filter_rate'])}**",
        (
            "- Candidate accuracy after a correct filter: "
            f"**{percent(summary['candidate_accuracy_after_correct_filter'])}**"
        ),
        f"- Wrong-candidate failures: **{summary['wrong_candidate_rows']}**",
        (
            "- Wrong-candidate failures choosing position 1: "
            f"**{summary['wrong_candidate_first_position']} "
            f"({percent(summary['wrong_candidate_first_position_rate'])})**"
        ),
        "",
        "## Candidate Positions",
        "",
        f"- Model: `{report['model_candidate_position_counts']}`",
        f"- Expert: `{report['expert_candidate_position_counts']}`",
        f"- Wrong-only: `{report['wrong_candidate_position_counts']}`",
        "",
        "## Templates",
        "",
        "| Template | Success | First-position rate | Main errors |",
        "| --- | ---: | ---: | --- |",
    ]
    for template, row in sorted(report["by_template"].items(), key=lambda item: item[1]["success_rate"], reverse=True):
        errors = ", ".join(
            f"{key}={value}" for key, value in row["error_counts"].items() if key != "success"
        )
        lines.append(
            f"| `{template}` | {row['successes']}/{row['total']} ({percent(row['success_rate'])}) | "
            f"{percent(row['first_position_rate'])} | {errors or '-'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze V2 filtering and candidate-position behavior.")
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--tasks", default="tasks/v2/eval_tasks.jsonl")
    parser.add_argument("--metadata", default="pages/generated_pages_v2/metadata.json")
    parser.add_argument("--output-json", default="outputs/eval_reports/v2_behavior_analysis.json")
    parser.add_argument("--output-md", default="outputs/eval_reports/v2_behavior_analysis.md")
    args = parser.parse_args()

    report = build_v2_behavior_analysis(args.trajectories, args.tasks, args.metadata)
    write_report(args.output_json, report)
    output_md = Path(args.output_md)
    if not output_md.is_absolute():
        output_md = ROOT / output_md
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
