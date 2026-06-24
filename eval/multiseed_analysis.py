from __future__ import annotations

import argparse
import json
import math
import statistics
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


def load_trajectories(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    trajectories = {row["task_id"]: row for row in rows}
    if len(trajectories) != len(rows):
        raise ValueError(f"Duplicate task ids in {path}")
    return trajectories


def mcnemar_exact_p(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    tail = min(improved, regressed)
    probability = 2 * sum(math.comb(discordant, index) for index in range(tail + 1)) / (2**discordant)
    return min(1.0, probability)


def trajectory_metrics(
    trajectories: dict[str, dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = list(trajectories.values())
    successes = sum(bool(row["summary"]["success"]) for row in rows)
    invalid_tool_calls = sum(int(row["summary"].get("invalid_tool_calls", 0)) for row in rows)
    model_steps = sum(int(row["summary"].get("model_steps", 0)) for row in rows)
    error_counts: Counter[str] = Counter()
    template_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for task_id, row in trajectories.items():
        task = tasks.get(task_id)
        error_type, _ = classify_failure(row, task)
        error_counts[error_type] += 1
        template = task.get("template", "unknown") if task else "unknown"
        template_counts[template]["total"] += 1
        if row["summary"]["success"]:
            template_counts[template]["success"] += 1
    total = len(rows)
    return {
        "num_tasks": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "invalid_tool_calls": invalid_tool_calls,
        "invalid_tool_call_rate": invalid_tool_calls / model_steps if model_steps else 0.0,
        "average_model_steps": model_steps / total if total else 0.0,
        "error_counts": dict(error_counts),
        "by_template": {
            template: {
                "total": counts["total"],
                "successes": counts["success"],
                "success_rate": counts["success"] / counts["total"] if counts["total"] else 0.0,
            }
            for template, counts in sorted(template_counts.items())
        },
    }


def paired_metrics(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if set(baseline) != set(candidate):
        missing = sorted(set(baseline) - set(candidate))
        extra = sorted(set(candidate) - set(baseline))
        raise ValueError(f"Task ids do not match. missing={missing[:5]}, extra={extra[:5]}")
    improved = []
    regressed = []
    both_correct = []
    both_wrong = []
    for task_id in baseline:
        baseline_success = bool(baseline[task_id]["summary"]["success"])
        candidate_success = bool(candidate[task_id]["summary"]["success"])
        if not baseline_success and candidate_success:
            improved.append(task_id)
        elif baseline_success and not candidate_success:
            regressed.append(task_id)
        elif baseline_success:
            both_correct.append(task_id)
        else:
            both_wrong.append(task_id)
    return {
        "improved": len(improved),
        "regressed": len(regressed),
        "both_correct": len(both_correct),
        "both_wrong": len(both_wrong),
        "net_improvement": len(improved) - len(regressed),
        "mcnemar_exact_p": mcnemar_exact_p(len(improved), len(regressed)),
        "improved_task_ids": improved,
        "regressed_task_ids": regressed,
    }


def parse_candidate(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Candidate must use NAME=PATH format")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Candidate must use non-empty NAME=PATH format")
    return name, path


def build_multiseed_analysis(
    baseline_name: str,
    baseline_path: str,
    candidates: list[tuple[str, str]],
    tasks_path: str,
) -> dict[str, Any]:
    tasks = {task["task_id"]: task for task in load_tasks(tasks_path)}
    baseline = load_trajectories(baseline_path)
    baseline_metrics = trajectory_metrics(baseline, tasks)

    candidate_results: dict[str, Any] = {}
    candidate_trajectories: dict[str, dict[str, dict[str, Any]]] = {}
    for name, path in candidates:
        trajectories = load_trajectories(path)
        candidate_trajectories[name] = trajectories
        metrics = trajectory_metrics(trajectories, tasks)
        candidate_results[name] = {
            "path": path,
            "metrics": metrics,
            "paired_vs_baseline": paired_metrics(baseline, trajectories),
        }

    success_rates = [row["metrics"]["success_rate"] for row in candidate_results.values()]
    invalid_rates = [row["metrics"]["invalid_tool_call_rate"] for row in candidate_results.values()]
    baseline_rate = baseline_metrics["success_rate"]

    robust_improvements = []
    robust_regressions = []
    majority = math.ceil(len(candidate_trajectories) / 2)
    for task_id, baseline_row in baseline.items():
        candidate_successes = sum(
            bool(rows[task_id]["summary"]["success"]) for rows in candidate_trajectories.values()
        )
        if not baseline_row["summary"]["success"] and candidate_successes >= majority:
            robust_improvements.append(task_id)
        if baseline_row["summary"]["success"] and candidate_successes < majority:
            robust_regressions.append(task_id)

    template_summary: dict[str, Any] = {}
    for template, baseline_template in baseline_metrics["by_template"].items():
        candidate_template_rates = [
            row["metrics"]["by_template"][template]["success_rate"] for row in candidate_results.values()
        ]
        mean_rate = statistics.mean(candidate_template_rates)
        template_summary[template] = {
            "num_tasks": baseline_template["total"],
            "baseline_success_rate": baseline_template["success_rate"],
            "candidate_mean_success_rate": mean_rate,
            "mean_delta": mean_rate - baseline_template["success_rate"],
            "candidate_success_rates": {
                name: row["metrics"]["by_template"][template]["success_rate"]
                for name, row in candidate_results.items()
            },
        }

    return {
        "baseline": {
            "name": baseline_name,
            "path": baseline_path,
            "metrics": baseline_metrics,
        },
        "candidates": candidate_results,
        "aggregate": {
            "num_seeds": len(candidates),
            "mean_success_rate": statistics.mean(success_rates),
            "sample_std_success_rate": statistics.stdev(success_rates) if len(success_rates) > 1 else 0.0,
            "mean_delta_vs_baseline": statistics.mean(success_rates) - baseline_rate,
            "mean_invalid_tool_call_rate": statistics.mean(invalid_rates),
            "best_candidate": max(candidate_results, key=lambda name: candidate_results[name]["metrics"]["success_rate"]),
            "worst_candidate": min(candidate_results, key=lambda name: candidate_results[name]["metrics"]["success_rate"]),
            "majority_threshold": majority,
            "robust_improvements": robust_improvements,
            "robust_regressions": robust_regressions,
        },
        "by_template": template_summary,
        "tasks_path": tasks_path,
    }


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    baseline = report["baseline"]
    aggregate = report["aggregate"]
    lines = [
        "# GRPO-KL Multi-Seed Analysis",
        "",
        f"Baseline: `{baseline['name']}` on {baseline['metrics']['num_tasks']} tasks.",
        "",
        "## Overall Results",
        "",
        "| Run | Successes | Success rate | Delta vs baseline | Invalid tool-call rate | McNemar p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {baseline['name']} | {baseline['metrics']['successes']} | "
            f"{percent(baseline['metrics']['success_rate'])} | - | "
            f"{percent(baseline['metrics']['invalid_tool_call_rate'])} | - |"
        ),
    ]
    for name, row in report["candidates"].items():
        metrics = row["metrics"]
        paired = row["paired_vs_baseline"]
        delta = metrics["success_rate"] - baseline["metrics"]["success_rate"]
        lines.append(
            f"| {name} | {metrics['successes']} | {percent(metrics['success_rate'])} | "
            f"{delta * 100:+.2f} pp | {percent(metrics['invalid_tool_call_rate'])} | "
            f"{paired['mcnemar_exact_p']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Mean success rate: **{percent(aggregate['mean_success_rate'])}**",
            f"- Sample standard deviation: **{aggregate['sample_std_success_rate'] * 100:.2f} pp**",
            f"- Mean delta vs baseline: **{aggregate['mean_delta_vs_baseline'] * 100:+.2f} pp**",
            f"- Best candidate: `{aggregate['best_candidate']}`",
            f"- Robust improvements (majority of seeds): **{len(aggregate['robust_improvements'])}**",
            f"- Robust regressions (majority of seeds): **{len(aggregate['robust_regressions'])}**",
            "",
            "## Paired Transitions",
            "",
            "| Run | Wrong to correct | Correct to wrong | Net | Both correct | Both wrong |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in report["candidates"].items():
        paired = row["paired_vs_baseline"]
        lines.append(
            f"| {name} | {paired['improved']} | {paired['regressed']} | "
            f"{paired['net_improvement']:+d} | {paired['both_correct']} | {paired['both_wrong']} |"
        )

    lines.extend(
        [
            "",
            "## Template-Level Mean Delta",
            "",
            "| Template | Tasks | Baseline | GRPO mean | Delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for template, row in sorted(
        report["by_template"].items(), key=lambda item: item[1]["mean_delta"], reverse=True
    ):
        lines.append(
            f"| `{template}` | {row['num_tasks']} | {percent(row['baseline_success_rate'])} | "
            f"{percent(row['candidate_mean_success_rate'])} | {row['mean_delta'] * 100:+.2f} pp |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze multiple candidate trajectories against one baseline.")
    parser.add_argument("--baseline-name", default="SFT step200")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--tasks", default="tasks/eval_tasks.jsonl")
    parser.add_argument("--output-json", default="outputs/eval_reports/grpo_multiseed_analysis.json")
    parser.add_argument("--output-md", default="outputs/eval_reports/grpo_multiseed_analysis.md")
    args = parser.parse_args()

    report = build_multiseed_analysis(args.baseline_name, args.baseline, args.candidate, args.tasks)
    write_report(args.output_json, report)
    output_md = Path(args.output_md)
    if not output_md.is_absolute():
        output_md = ROOT / output_md
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
