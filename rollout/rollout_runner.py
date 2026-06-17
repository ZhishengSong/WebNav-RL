from __future__ import annotations

import argparse
from typing import Any

from env.browser_env import BrowserEnv
from rollout.trajectory import save_jsonl, tool_call_message
from tasks.task_loader import load_tasks


def run_expert_task(task: dict[str, Any]) -> dict[str, Any]:
    env = BrowserEnv()
    messages = [{"role": "user", "content": task["instruction"]}]

    env.reset(task)
    messages.append({"role": "assistant", "content": tool_call_message("open_page", {"page_id": task["start_page"]})})
    messages.append({"role": "tool", "content": env.action_log[-1]["observation"]})

    for element_id in task["expert_clicks"]:
        response = env.click(element_id)
        messages.append({"role": "assistant", "content": tool_call_message("click", {"element_id": element_id})})
        messages.append({"role": "tool", "content": response["observation"]})

    response = env.submit_answer(task["target_answer"])
    messages.append({"role": "assistant", "content": tool_call_message("submit_answer", {"answer": task["target_answer"]})})
    messages.append({"role": "tool", "content": response["observation"]})

    return {
        "task_id": task["task_id"],
        "instruction": task["instruction"],
        "messages": messages,
        "actions": env.action_log,
        "summary": env.summary(),
    }


def run_expert(tasks_path: str, output_path: str) -> list[dict[str, Any]]:
    tasks = load_tasks(tasks_path)
    trajectories = [run_expert_task(task) for task in tasks]
    save_jsonl(output_path, trajectories)
    return trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="tasks/all_tasks.jsonl")
    parser.add_argument("--output", default="outputs/trajectories/expert_trajectories.jsonl")
    args = parser.parse_args()
    trajectories = run_expert(args.tasks, args.output)
    success = sum(1 for row in trajectories if row["summary"]["success"])
    print(f"Expert success rate: {success}/{len(trajectories)} = {success / len(trajectories):.1%}")
    print(f"Saved trajectories to {args.output}")


if __name__ == "__main__":
    main()
