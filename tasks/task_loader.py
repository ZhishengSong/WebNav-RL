from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    task_path = Path(path)
    if not task_path.is_absolute():
        task_path = ROOT / task_path
    if not task_path.exists():
        raise FileNotFoundError(f"Task file not found: {task_path}")
    tasks = []
    for line in task_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            tasks.append(json.loads(line))
    return tasks


def load_task_by_id(path: str | Path, task_id: str) -> dict[str, Any]:
    for task in load_tasks(path):
        if task["task_id"] == task_id:
            return task
    raise KeyError(f"Task id not found: {task_id}")
