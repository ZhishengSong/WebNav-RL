from __future__ import annotations

import unittest

from eval.metrics import build_eval_report
from env.browser_env import BrowserEnv
from rollout.model_runner import ExpertReplayGenerator, run_model_task
from tasks.task_loader import load_tasks
from tools.tool_registry import ToolRegistry
from tools.web_tools import WebTools


class ModelEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.task = load_tasks("tasks/eval_tasks.jsonl")[0]

    def test_expert_replay_completes_task(self) -> None:
        trajectory = run_model_task(self.task, ExpertReplayGenerator(self.task))
        self.assertTrue(trajectory["summary"]["success"])
        self.assertEqual(trajectory["summary"]["format_errors"], 0)
        self.assertEqual(trajectory["summary"]["invalid_tool_calls"], 0)

    def test_parser_errors_are_counted(self) -> None:
        trajectory = run_model_task(self.task, lambda messages: "not a tool call", max_steps=2)
        self.assertFalse(trajectory["summary"]["success"])
        self.assertEqual(trajectory["summary"]["format_errors"], 2)
        self.assertEqual(trajectory["summary"]["termination"], "max_steps")

    def test_report_aggregates_metrics(self) -> None:
        trajectory = run_model_task(self.task, ExpertReplayGenerator(self.task))
        report = build_eval_report([trajectory])
        self.assertEqual(report["task_success_rate"], 1.0)
        self.assertEqual(report["tool_call_format_accuracy"], 1.0)
        self.assertEqual(report["invalid_tool_call_rate"], 0.0)

    def test_registry_records_unknown_tool_and_bad_arguments(self) -> None:
        env = BrowserEnv()
        env.start_episode(self.task)
        registry = ToolRegistry(WebTools(env))
        self.assertEqual(registry.call("missing", {})["status"], "error")
        self.assertEqual(registry.call("click", {})["status"], "error")
        self.assertEqual(env.invalid_actions, 2)


if __name__ == "__main__":
    unittest.main()
