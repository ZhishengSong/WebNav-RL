from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tools.web_tools import WebTools


class ToolRegistry:
    def __init__(self, tools: WebTools) -> None:
        self._tools: dict[str, Callable[..., dict[str, Any]]] = {
            "open_page": tools.open_page,
            "click": tools.click,
            "get_visible_text": tools.get_visible_text,
            "submit_answer": tools.submit_answer,
        }

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            return {
                "tool_name": name,
                "arguments": arguments,
                "status": "error",
                "current_page": None,
                "observation": f"Unknown tool: {name}",
            }
        return self._tools[name](**arguments)

    @property
    def names(self) -> list[str]:
        return list(self._tools)
