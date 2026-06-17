from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pages.page_generator import generate
from rollout.rollout_runner import run_expert


def main() -> None:
    generate(num_tasks=20)
    trajectories = run_expert("tasks/all_tasks.jsonl", "outputs/trajectories/expert_trajectories.jsonl")
    success = sum(1 for row in trajectories if row["summary"]["success"])
    report = {
        "num_tasks": len(trajectories),
        "success": success,
        "success_rate": success / len(trajectories),
        "avg_steps": sum(row["summary"]["steps"] for row in trajectories) / len(trajectories),
        "invalid_actions": sum(row["summary"]["invalid_actions"] for row in trajectories),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
