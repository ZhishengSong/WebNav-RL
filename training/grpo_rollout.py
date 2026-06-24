from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.evaluate import append_jsonl, load_jsonl, write_report
from env.browser_env import BrowserEnv
from env.page_state import PageStore
from rollout.model_runner import ExpertReplayGenerator, TextGenerator, run_model_task
from rollout.transformers_generator import TransformersGenerator
from rollout.trajectory import save_jsonl
from tasks.task_loader import load_tasks
from training.reward import compute_reward, mean


def build_generator(args: argparse.Namespace) -> TextGenerator:
    if not args.model:
        raise ValueError("GRPO rollout collection requires --model. Use --expert-replay only for smoke tests.")
    return TransformersGenerator(
        args.model,
        adapter_path=args.adapter,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )


def collect_group_for_task(
    task: dict[str, Any],
    generator_factory: Any,
    group_size: int,
    max_steps: int | None,
    page_store: PageStore | None = None,
) -> list[dict[str, Any]]:
    return [
        run_model_task(
            task,
            generator_factory(task),
            max_steps=max_steps,
            env=BrowserEnv(page_store=page_store) if page_store is not None else None,
        )
        for _ in range(group_size)
    ]


def group_records(
    task: dict[str, Any],
    trajectories: list[dict[str, Any]],
    group_index: int,
) -> list[dict[str, Any]]:
    rewards = [compute_reward(row, task) for row in trajectories]
    reward_values = [row["total_reward"] for row in rewards]
    group_mean = mean(reward_values)
    records = []
    for sample_index, (trajectory, reward) in enumerate(zip(trajectories, rewards)):
        records.append(
            {
                "group_id": f"{task['task_id']}::g{group_index:05d}",
                "task_id": task["task_id"],
                "sample_index": sample_index,
                "group_size": len(trajectories),
                "group_mean_reward": round(group_mean, 6),
                "advantage": round(reward["total_reward"] - group_mean, 6),
                "reward": reward,
                "trajectory": trajectory,
            }
        )
    return records


def summarize_group_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [row["reward"]["total_reward"] for row in records]
    advantages = [row["advantage"] for row in records]
    successes = [row for row in records if row["reward"]["success"]]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault(row["group_id"], []).append(row)
    nonzero_advantage_groups = sum(
        1 for rows in groups.values() if len({row["reward"]["total_reward"] for row in rows}) > 1
    )
    return {
        "num_samples": len(records),
        "num_groups": len(groups),
        "group_size": records[0]["group_size"] if records else 0,
        "mean_reward": mean(rewards),
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "success_rate": len(successes) / len(records) if records else 0.0,
        "mean_abs_advantage": mean([abs(value) for value in advantages]),
        "nonzero_advantage_groups": nonzero_advantage_groups,
        "zero_advantage_groups": len(groups) - nonzero_advantage_groups,
    }


def collect_grpo_rollouts(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks = load_tasks(args.tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    completed_group_task_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    if args.resume and args.output:
        records = load_jsonl(args.output)
        completed_group_task_ids = {row["task_id"] for row in records}

    if args.expert_replay:
        generator_factory = lambda task: ExpertReplayGenerator(task)
    else:
        shared_generator = build_generator(args)
        generator_factory = lambda task: shared_generator
    page_store = PageStore(args.metadata) if args.metadata is not None else None

    pending_tasks = [task for task in tasks if task["task_id"] not in completed_group_task_ids]
    for group_index, task in enumerate(pending_tasks, start=1):
        trajectories = collect_group_for_task(
            task,
            generator_factory,
            group_size=args.group_size,
            max_steps=args.max_steps,
            page_store=page_store,
        )
        records_for_task = group_records(task, trajectories, group_index)
        records.extend(records_for_task)
        if args.incremental:
            for row in records_for_task:
                append_jsonl(args.output, row)
        if args.report and (group_index % args.report_every == 0 or group_index == len(pending_tasks)):
            report = {
                "metadata": {
                    "model": args.model,
                    "adapter": args.adapter,
                    "tasks": args.tasks,
                    "page_metadata": args.metadata,
                    "limit": args.limit,
                    "group_size": args.group_size,
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                    "max_steps": args.max_steps,
                    "resume": args.resume,
                    "incremental": args.incremental,
                    "completed_groups": len({row["group_id"] for row in records}),
                    "pending_groups": len(tasks) - len({row["task_id"] for row in records}),
                },
                "summary": summarize_group_records(records),
            }
            write_report(args.report, report)

    if args.output and not args.incremental:
        save_jsonl(args.output, records)
    report = {
        "metadata": {
            "model": args.model,
            "adapter": args.adapter,
            "tasks": args.tasks,
            "page_metadata": args.metadata,
            "limit": args.limit,
            "group_size": args.group_size,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "max_steps": args.max_steps,
            "resume": args.resume,
            "incremental": args.incremental,
            "completed_groups": len({row["group_id"] for row in records}),
            "pending_groups": len(tasks) - len({row["task_id"] for row in records}),
        },
        "summary": summarize_group_records(records),
    }
    if args.report:
        write_report(args.report, report)
    return records, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect grouped rollouts and GRPO-style advantages.")
    parser.add_argument("--tasks", default="tasks/eval_tasks.jsonl")
    parser.add_argument("--metadata", default=None, help="Optional page metadata path for V2/custom datasets.")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--model", default="models/qwen2.5-0.5b-instruct")
    parser.add_argument("--adapter", default="outputs/checkpoints/qwen2_5_0_5b_lora_sft_step200")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--expert-replay", action="store_true")
    parser.add_argument("--output", default="outputs/rollouts/grpo_group_rollouts.jsonl")
    parser.add_argument("--report", default="outputs/eval_reports/grpo_group_rollout_report.json")
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-every", type=int, default=1)
    args = parser.parse_args()
    _, report = collect_grpo_rollouts(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
