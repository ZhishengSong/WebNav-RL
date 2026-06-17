from __future__ import annotations

from typing import Any

from env.page_state import PageStore
from env.verifier import ExactMatchVerifier


class BrowserEnv:
    def __init__(self, page_store: PageStore | None = None, verifier: ExactMatchVerifier | None = None) -> None:
        self.page_store = page_store or PageStore()
        self.verifier = verifier or ExactMatchVerifier()
        self.task: dict[str, Any] | None = None
        self.current_page: str | None = None
        self.final_answer: str | None = None
        self.done = False
        self.invalid_actions = 0
        self.action_log: list[dict[str, Any]] = []

    def reset(self, task: dict[str, Any]) -> dict[str, Any]:
        self.task = task
        self.current_page = None
        self.final_answer = None
        self.done = False
        self.invalid_actions = 0
        self.action_log = []
        return self.open_page(task["start_page"])

    def open_page(self, page_id: str) -> dict[str, Any]:
        try:
            page = self.page_store.get(page_id)
        except KeyError as exc:
            return self._record("open_page", {"page_id": page_id}, "error", str(exc))
        self.current_page = page_id
        return self._record(
            "open_page",
            {"page_id": page_id},
            "success",
            f"Opened {page_id}",
            observation=page["visible_text"],
        )

    def click(self, element_id: str) -> dict[str, Any]:
        if self.current_page is None:
            return self._invalid("click", {"element_id": element_id}, "No page is currently open")
        element = self.page_store.find_element(self.current_page, element_id)
        if element is None:
            return self._invalid("click", {"element_id": element_id}, f"Element not found: {element_id}")
        target_page = element["target_page"]
        page = self.page_store.get(target_page)
        self.current_page = target_page
        return self._record(
            "click",
            {"element_id": element_id},
            "success",
            f"Clicked {element_id}; opened {target_page}",
            observation=page["visible_text"],
        )

    def get_visible_text(self) -> dict[str, Any]:
        if self.current_page is None:
            return self._invalid("get_visible_text", {}, "No page is currently open")
        page = self.page_store.get(self.current_page)
        return self._record(
            "get_visible_text",
            {},
            "success",
            f"Visible text from {self.current_page}",
            observation=page["visible_text"],
        )

    def submit_answer(self, answer: str) -> dict[str, Any]:
        self.final_answer = answer
        self.done = True
        status = "success"
        message = f"Submitted answer: {answer}"
        if self.task is not None and not self.verifier.verify(answer, self.task["target_answer"]):
            status = "failure"
        return self._record("submit_answer", {"answer": answer}, status, message, observation=message)

    def summary(self) -> dict[str, Any]:
        if self.task is None:
            raise RuntimeError("Cannot summarize before reset.")
        return self.verifier.episode_summary(
            task=self.task,
            final_answer=self.final_answer,
            steps=len(self.action_log),
            invalid_actions=self.invalid_actions,
        )

    def _invalid(self, tool_name: str, arguments: dict[str, Any], message: str) -> dict[str, Any]:
        self.invalid_actions += 1
        return self._record(tool_name, arguments, "error", message, observation=message)

    def _record(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        status: str,
        message: str,
        observation: str | None = None,
    ) -> dict[str, Any]:
        response = {
            "tool_name": tool_name,
            "arguments": arguments,
            "status": status,
            "current_page": self.current_page,
            "observation": observation or message,
        }
        self.action_log.append(response)
        return response
