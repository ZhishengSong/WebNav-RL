from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric(report: dict[str, Any], name: str) -> Any:
    return report.get(name, report.get("summary", {}).get(name))


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two evaluation reports.")
    parser.add_argument("--baseline", required=True, help="SFT/baseline eval report JSON.")
    parser.add_argument("--candidate", required=True, help="GRPO/candidate eval report JSON.")
    parser.add_argument("--baseline-name", default="SFT")
    parser.add_argument("--candidate-name", default="GRPO-KL")
    parser.add_argument("--output-json", default="outputs/eval_reports/grpo_kl_comparison.json")
    parser.add_argument("--output-md", default="outputs/eval_reports/grpo_kl_comparison.md")
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)
    metric_names = [
        "num_tasks",
        "successes",
        "task_success_rate",
        "final_answer_accuracy",
        "tool_call_format_accuracy",
        "invalid_tool_call_rate",
        "average_model_steps",
        "format_errors",
        "invalid_tool_calls",
    ]
    rows = []
    for name in metric_names:
        base_value = metric(baseline, name)
        cand_value = metric(candidate, name)
        delta = None
        if isinstance(base_value, (int, float)) and isinstance(cand_value, (int, float)):
            delta = cand_value - base_value
        rows.append(
            {
                "metric": name,
                "baseline": base_value,
                "candidate": cand_value,
                "delta": delta,
            }
        )

    output = {
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "baseline_report": args.baseline,
        "candidate_report": args.candidate,
        "rows": rows,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# {args.baseline_name} vs {args.candidate_name}",
        "",
        f"- Baseline report: `{args.baseline}`",
        f"- Candidate report: `{args.candidate}`",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['metric']}` | {fmt(row['baseline'])} | {fmt(row['candidate'])} | {fmt(row['delta'])} |"
        )
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
