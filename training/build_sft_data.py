from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from rollout.trajectory import save_jsonl


ROOT = Path(__file__).resolve().parents[1]
TOOL_CALL_RE = re.compile(r"^<tool_call>(.*)</tool_call>$")


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = resolve_path(path)
    rows: list[dict[str, Any]] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def validate_tool_calls(messages: list[dict[str, str]]) -> None:
    for message in messages:
        if message["role"] != "assistant":
            continue
        match = TOOL_CALL_RE.match(message["content"])
        if match is None:
            raise ValueError(f"Assistant message is not a tool call: {message['content']}")
        payload = json.loads(match.group(1))
        if sorted(payload) != ["arguments", "name"]:
            raise ValueError(f"Invalid tool call keys: {payload}")
        if not isinstance(payload["arguments"], dict):
            raise ValueError(f"Tool call arguments must be an object: {payload}")


def trajectory_to_sft(row: dict[str, Any]) -> dict[str, Any]:
    messages = row["messages"]
    validate_tool_calls(messages)
    return {
        "id": row["task_id"],
        "instruction": row["instruction"],
        "messages": messages,
        "summary": row["summary"],
    }


def build_sft_data(input_path: str, output_path: str) -> list[dict[str, Any]]:
    rows = load_jsonl(input_path)
    examples = [trajectory_to_sft(row) for row in rows]
    save_jsonl(output_path, examples)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/trajectories/expert_trajectories.jsonl")
    parser.add_argument("--output", default="training/sft_data.jsonl")
    args = parser.parse_args()
    examples = build_sft_data(args.input, args.output)
    print(f"Saved {len(examples)} SFT examples to {args.output}")


if __name__ == "__main__":
    main()
