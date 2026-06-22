from __future__ import annotations

from collections import Counter
from typing import Any


def build_eval_report(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(trajectories)
    if count == 0:
        raise ValueError("Cannot build an evaluation report with no trajectories")

    summaries = [row["summary"] for row in trajectories]
    model_steps = sum(row["model_steps"] for row in summaries)
    format_errors = sum(row["format_errors"] for row in summaries)
    invalid_calls = sum(row["invalid_tool_calls"] for row in summaries)
    successes = sum(bool(row["success"]) for row in summaries)

    return {
        "num_tasks": count,
        "successes": successes,
        "task_success_rate": successes / count,
        "final_answer_accuracy": successes / count,
        "tool_call_format_accuracy": (model_steps - format_errors) / model_steps if model_steps else 0.0,
        "invalid_tool_call_rate": invalid_calls / model_steps if model_steps else 0.0,
        "average_model_steps": model_steps / count,
        "average_environment_actions": sum(row["steps"] for row in summaries) / count,
        "format_errors": format_errors,
        "invalid_tool_calls": invalid_calls,
        "termination_counts": dict(Counter(row["termination"] for row in summaries)),
    }
