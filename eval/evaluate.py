from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from eval.metrics import build_eval_report
from rollout.model_runner import ExpertReplayGenerator, TextGenerator, run_model_task
from rollout.trajectory import save_jsonl


ROOT = Path(__file__).resolve().parents[1]


def evaluate_tasks(
    tasks: list[dict[str, Any]],
    generator_factory: Callable[[dict[str, Any]], TextGenerator],
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
    failures_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trajectories = [run_model_task(task, generator_factory(task)) for task in tasks]
    report = build_eval_report(trajectories)

    if output_path is not None:
        save_jsonl(output_path, trajectories)
    if failures_path is not None:
        save_jsonl(failures_path, [row for row in trajectories if not row["summary"]["success"]])
    if report_path is not None:
        resolved = Path(report_path)
        if not resolved.is_absolute():
            resolved = ROOT / resolved
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return trajectories, report


def expert_replay_factory(task: dict[str, Any]) -> TextGenerator:
    return ExpertReplayGenerator(task)
