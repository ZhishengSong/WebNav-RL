from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from eval.metrics import build_eval_report
from rollout.model_runner import ExpertReplayGenerator, TextGenerator, run_model_task
from rollout.trajectory import save_jsonl


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = resolve_path(path)
    if not resolved.exists():
        return []
    rows = []
    for line in resolved.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def evaluate_tasks(
    tasks: list[dict[str, Any]],
    generator_factory: Callable[[dict[str, Any]], TextGenerator],
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
    failures_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    max_steps: int | None = None,
    resume: bool = False,
    incremental: bool = False,
    report_every: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    completed_task_ids: set[str] = set()
    if resume and output_path is not None:
        trajectories = load_jsonl(output_path)
        completed_task_ids = {row["task_id"] for row in trajectories}

    pending_tasks = [task for task in tasks if task["task_id"] not in completed_task_ids]
    for index, task in enumerate(pending_tasks, start=1):
        trajectory = run_model_task(task, generator_factory(task), max_steps=max_steps)
        trajectories.append(trajectory)
        if incremental and output_path is not None:
            append_jsonl(output_path, trajectory)
        if report_path is not None and (incremental and (index % report_every == 0 or index == len(pending_tasks))):
            partial_report = build_eval_report(trajectories)
            if metadata is not None:
                partial_report["metadata"] = metadata
            partial_report["metadata"] = {
                **partial_report.get("metadata", {}),
                "completed_tasks": len(trajectories),
                "pending_tasks": len(tasks) - len(trajectories),
                "resume": resume,
                "incremental": incremental,
            }
            write_report(report_path, partial_report)

    report = build_eval_report(trajectories)
    if metadata is not None:
        report["metadata"] = metadata
    report["metadata"] = {
        **report.get("metadata", {}),
        "completed_tasks": len(trajectories),
        "pending_tasks": len(tasks) - len(trajectories),
        "resume": resume,
        "incremental": incremental,
    }

    if output_path is not None and not incremental:
        save_jsonl(output_path, trajectories)
    if failures_path is not None:
        save_jsonl(failures_path, [row for row in trajectories if not row["summary"]["success"]])
    if report_path is not None:
        write_report(report_path, report)
    return trajectories, report


def expert_replay_factory(task: dict[str, Any]) -> TextGenerator:
    return ExpertReplayGenerator(task)
