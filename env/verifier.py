from __future__ import annotations

from typing import Any


class ExactMatchVerifier:
    def verify(self, answer: str | None, target_answer: str) -> bool:
        if answer is None:
            return False
        return answer.strip().lower() == target_answer.strip().lower()

    def episode_summary(
        self,
        task: dict[str, Any],
        final_answer: str | None,
        steps: int,
        invalid_actions: int,
    ) -> dict[str, Any]:
        success = self.verify(final_answer, task["target_answer"])
        return {
            "task_id": task["task_id"],
            "success": success,
            "final_answer": final_answer,
            "target_answer": task["target_answer"],
            "steps": steps,
            "invalid_actions": invalid_actions,
        }
