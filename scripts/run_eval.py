from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.evaluate import evaluate_tasks, expert_replay_factory
from rollout.transformers_generator import TransformersGenerator
from tasks.task_loader import load_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an expert replay or a local model.")
    parser.add_argument("--tasks", default="tasks/eval_tasks.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="outputs/trajectories/eval_trajectories.jsonl")
    parser.add_argument("--report", default="outputs/eval_reports/eval_report.json")
    parser.add_argument("--failures", default="outputs/eval_reports/failures.jsonl")
    parser.add_argument("--model", help="Local path or Hugging Face model id. Omit for expert replay.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    generator_factory = expert_replay_factory
    if args.model:
        generator = TransformersGenerator(
            args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
        )
        generator_factory = lambda task: generator

    _, report = evaluate_tasks(
        tasks,
        generator_factory,
        output_path=args.output,
        report_path=args.report,
        failures_path=args.failures,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
