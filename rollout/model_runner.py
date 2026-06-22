from __future__ import annotations

from collections.abc import Callable
from typing import Any

from env.browser_env import BrowserEnv
from rollout.parser import ToolCallParseError, parse_tool_call
from rollout.trajectory import tool_call_message
from tools.tool_registry import ToolRegistry
from tools.web_tools import WebTools


Message = dict[str, str]
TextGenerator = Callable[[list[Message]], str]


SYSTEM_PROMPT = """You are a web navigation agent. Respond with exactly one tool call:
<tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>
Available tools:
- open_page(page_id)
- click(element_id)
- get_visible_text()
- submit_answer(answer)
Use only information returned by the tools."""


def run_model_task(
    task: dict[str, Any],
    generate: TextGenerator,
    max_steps: int | None = None,
    env: BrowserEnv | None = None,
) -> dict[str, Any]:
    browser = env or BrowserEnv()
    browser.start_episode(task)
    registry = ToolRegistry(WebTools(browser))
    step_limit = max_steps or int(task.get("max_steps", 8))
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["instruction"]},
    ]
    raw_outputs: list[str] = []
    parser_errors: list[dict[str, Any]] = []
    invalid_tool_calls = 0
    termination = "max_steps"

    for step in range(step_limit):
        output = generate(messages)
        raw_outputs.append(output)
        messages.append({"role": "assistant", "content": output})

        try:
            call = parse_tool_call(output)
        except ToolCallParseError as exc:
            invalid_tool_calls += 1
            error = {"step": step + 1, "code": exc.code, "message": str(exc), "output": output}
            parser_errors.append(error)
            messages.append({"role": "tool", "content": f"Tool call error: {exc}"})
            continue

        response = registry.call(call.name, call.arguments)
        if response["status"] == "error":
            invalid_tool_calls += 1
        messages.append({"role": "tool", "content": response["observation"]})
        if browser.done:
            termination = "submitted"
            break

    summary = browser.summary()
    summary.update(
        {
            "model_steps": len(raw_outputs),
            "format_errors": len(parser_errors),
            "invalid_tool_calls": invalid_tool_calls,
            "termination": termination,
        }
    )
    return {
        "task_id": task["task_id"],
        "instruction": task["instruction"],
        "messages": messages,
        "raw_outputs": raw_outputs,
        "parser_errors": parser_errors,
        "actions": browser.action_log,
        "summary": summary,
    }


class ExpertReplayGenerator:
    """Oracle generator used to smoke-test the model evaluation pipeline."""

    def __init__(self, task: dict[str, Any]) -> None:
        self._calls = [
            tool_call_message("open_page", {"page_id": task["start_page"]}),
            *(tool_call_message("click", {"element_id": item}) for item in task["expert_clicks"]),
            tool_call_message("submit_answer", {"answer": task["target_answer"]}),
        ]
        self._index = 0

    def __call__(self, messages: list[Message]) -> str:
        del messages
        if self._index >= len(self._calls):
            return tool_call_message("submit_answer", {"answer": ""})
        output = self._calls[self._index]
        self._index += 1
        return output
