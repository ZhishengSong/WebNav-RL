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
    parser.add_argument("--adapter", help="Optional LoRA adapter path for model eval.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from existing output trajectories.")
    parser.add_argument("--incremental", action="store_true", help="Append trajectories and refresh report while running.")
    parser.add_argument("--report-every", type=int, default=10, help="Refresh report every N newly completed tasks.")
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
            adapter_path=args.adapter,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
        )
        generator_factory = lambda task: generator

    metadata = {
        "eval_mode": "transformers_model" if args.model else "expert_replay",
        "model": args.model,
        "adapter": args.adapter,
        "tasks": args.tasks,
        "limit": args.limit,
        "device": args.device if args.model else None,
        "max_new_tokens": args.max_new_tokens if args.model else None,
        "max_steps": args.max_steps,
        "resume": args.resume,
        "incremental": args.incremental,
        "temperature": args.temperature if args.model else None,
        "trust_remote_code": args.trust_remote_code if args.model else None,
    }
    _, report = evaluate_tasks(
        tasks,
        generator_factory,
        output_path=args.output,
        report_path=args.report,
        failures_path=args.failures,
        metadata=metadata,
        max_steps=args.max_steps,
        resume=args.resume,
        incremental=args.incremental,
        report_every=args.report_every,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
