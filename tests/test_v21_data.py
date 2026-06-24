from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from env.page_state import PageStore
from pages.v21_generator import generate_v21_pages
from rollout.rollout_runner import run_expert_task
from scripts.run_v21_data import target_position_audit
from tasks.v2_task_generator import generate_v2_tasks


class V21DataTests(unittest.TestCase):
    def test_multi_instance_position_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata, contexts, page_manifest = generate_v21_pages(
                output_dir=Path(temp_dir) / "pages",
                seed=151,
                train_instances=8,
                eval_instances=3,
            )
            train_tasks, eval_tasks, task_manifest = generate_v2_tasks(
                contexts,
                train_count=2400,
                eval_count=450,
                seed=157,
                dataset_version="v2.1",
                task_prefix="v21test",
                spread_templates_across_contexts=True,
            )

            self.assertEqual(page_manifest["train_eval_element_id_overlap"], 0)
            self.assertEqual(task_manifest["expert_element_id_overlap"], 0)
            self.assertEqual(len(task_manifest["train_layouts"]), 8)
            self.assertEqual(len(task_manifest["eval_layouts"]), 3)
            for template in task_manifest["train_template_counts"]:
                layouts = {
                    task["layout_id"]
                    for task in train_tasks
                    if task["template"] == template
                }
                self.assertEqual(layouts, set(task_manifest["train_layouts"]))

            audit = target_position_audit(train_tasks, metadata)
            self.assertLessEqual(audit["max_template_position_share"], 0.4)
            self.assertGreaterEqual(audit["min_answer_unique_positions"], 2)

            store = PageStore(Path(temp_dir) / "pages" / "metadata.json")
            trajectories = [run_expert_task(task, page_store=store) for task in eval_tasks[:50]]
            self.assertTrue(all(row["summary"]["success"] for row in trajectories))

    def test_committed_data_report_meets_training_contract(self) -> None:
        report_path = Path(__file__).resolve().parents[1] / "outputs" / "eval_reports" / "v21_data_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["version"], "v2.1")
        self.assertEqual(report["page_manifest"]["train_eval_element_id_overlap"], 0)
        self.assertEqual(report["task_manifest"]["expert_element_id_overlap"], 0)
        self.assertEqual(report["summary"]["num_tasks"], 7000)
        self.assertEqual(report["summary"]["expert_successes"], 7000)
        self.assertEqual(report["summary"]["expert_path_target_matches"], 7000)
        self.assertEqual(report["summary"]["train_next_action_examples"], 22400)
        self.assertEqual(report["summary"]["invalid_actions"], 0)
        self.assertEqual(report["summary"]["action_errors"], 0)
        self.assertLessEqual(report["train_position_audit"]["max_template_position_share"], 0.3)
        self.assertGreaterEqual(report["train_position_audit"]["min_answer_unique_positions"], 4)


if __name__ == "__main__":
    unittest.main()
