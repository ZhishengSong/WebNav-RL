from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from tools.web_tools import WebTools


class ToolRegistry:
    def __init__(self, tools: WebTools) -> None:
        self._env = tools.env
        self._tools: dict[str, Callable[..., dict[str, Any]]] = {
            "open_page": tools.open_page,
            "click": tools.click,
            "get_visible_text": tools.get_visible_text,
            "submit_answer": tools.submit_answer,
        }

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            return self._env.record_invalid_action(name, arguments, f"Unknown tool: {name}")
        tool = self._tools[name]
        try:
            inspect.signature(tool).bind(**arguments)
        except TypeError as exc:
            return self._env.record_invalid_action(
                name,
                arguments,
                f"Invalid arguments for {name}: {exc}",
            )
        return tool(**arguments)

    @property
    def names(self) -> list[str]:
        return list(self._tools)
