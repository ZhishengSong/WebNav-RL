from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


TOOL_CALL_RE = re.compile(r"\s*<tool_call>(.*?)</tool_call>\s*", re.DOTALL)


class ToolCallParseError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


def parse_tool_call(text: str) -> ToolCall:
    if not isinstance(text, str) or not text.strip():
        raise ToolCallParseError("empty_output", "Model output is empty")

    match = TOOL_CALL_RE.fullmatch(text)
    if match is None:
        raise ToolCallParseError(
            "invalid_wrapper",
            "Expected exactly one <tool_call>...</tool_call> block",
        )

    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ToolCallParseError("invalid_json", f"Invalid tool call JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ToolCallParseError("invalid_schema", "Tool call payload must be a JSON object")
    if set(payload) != {"name", "arguments"}:
        raise ToolCallParseError(
            "invalid_schema",
            "Tool call payload must contain exactly 'name' and 'arguments'",
        )
    if not isinstance(payload["name"], str) or not payload["name"].strip():
        raise ToolCallParseError("invalid_schema", "Tool name must be a non-empty string")
    if not isinstance(payload["arguments"], dict):
        raise ToolCallParseError("invalid_schema", "Tool arguments must be a JSON object")

    return ToolCall(name=payload["name"], arguments=payload["arguments"])
