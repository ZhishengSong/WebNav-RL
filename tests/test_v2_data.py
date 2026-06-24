from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from env.browser_env import BrowserEnv
from env.page_state import PageStore
from pages.v2_generator import COURSES_V2, PRODUCTS_V2, generate_v2_pages
from rollout.model_runner import ExpertReplayGenerator, run_model_task
from rollout.rollout_runner import run_expert_task
from tasks.v2_task_generator import generate_v2_tasks


class V2DataTests(unittest.TestCase):
    def test_catalog_queries_are_unambiguous(self) -> None:
        self.assertEqual(len({product.price for product in PRODUCTS_V2}), len(PRODUCTS_V2))
        self.assertEqual(
            len({(product.category, product.color) for product in PRODUCTS_V2}),
            len(PRODUCTS_V2),
        )
        self.assertEqual(len({course.code for course in COURSES_V2}), len(COURSES_V2))
        self.assertEqual(len({course.title for course in COURSES_V2}), len(COURSES_V2))
        self.assertEqual(
            len({(course.department, course.time) for course in COURSES_V2}),
            len(COURSES_V2),
        )

        for category in {product.category for product in PRODUCTS_V2}:
            matches = [product for product in PRODUCTS_V2 if product.category == category]
            self.assertEqual(sum(product.rating == max(item.rating for item in matches) for product in matches), 1)
            self.assertEqual(sum(product.price == min(item.price for item in matches) for product in matches), 1)
        for credits in {course.credits for course in COURSES_V2}:
            matches = [course for course in COURSES_V2 if course.credits == credits]
            self.assertEqual(sum(course.rating == max(item.rating for item in matches) for course in matches), 1)

    def test_structural_split_and_expert_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "pages"
            metadata, contexts, page_manifest = generate_v2_pages(output_dir=output_dir, seed=101)
            train_tasks, eval_tasks, task_manifest = generate_v2_tasks(
                contexts,
                train_count=60,
                eval_count=30,
                seed=107,
            )

            self.assertEqual(page_manifest["train_eval_element_id_overlap"], 0)
            self.assertEqual(task_manifest["expert_element_id_overlap"], 0)
            self.assertEqual({task["layout_id"] for task in train_tasks}, {"train_a", "train_b"})
            self.assertEqual({task["layout_id"] for task in eval_tasks}, {"eval_c"})
            self.assertTrue(all(task["start_page"] in task["instruction"] for task in eval_tasks))
            train_counts = task_manifest["train_template_counts"].values()
            eval_counts = task_manifest["eval_template_counts"].values()
            self.assertLessEqual(max(train_counts) - min(train_counts), 1)
            self.assertLessEqual(max(eval_counts) - min(eval_counts), 1)

            all_clicks = [click for task in train_tasks + eval_tasks for click in task["expert_clicks"]]
            self.assertTrue(all(re.fullmatch(r"el_[a-z0-9]{10}", click) for click in all_clicks))
            self.assertTrue(any("element_id=" in page["visible_text"] for page in metadata.values()))

            store = PageStore(output_dir / "metadata.json")
            trajectories = [run_expert_task(task, page_store=store) for task in train_tasks + eval_tasks]
            self.assertTrue(all(row["summary"]["success"] for row in trajectories))
            self.assertEqual(sum(row["summary"]["invalid_actions"] for row in trajectories), 0)
            for task, trajectory in zip(train_tasks + eval_tasks, trajectories):
                tool_observations = [
                    message["content"] for message in trajectory["messages"] if message["role"] == "tool"
                ]
                for click_index, element_id in enumerate(task["expert_clicks"]):
                    self.assertIn(element_id, tool_observations[click_index])
                final_page = trajectory["actions"][-1]["current_page"]
                self.assertEqual(store.get(final_page).get("answer"), task["target_answer"])

            eval_task = eval_tasks[0]
            model_trajectory = run_model_task(
                eval_task,
                ExpertReplayGenerator(eval_task),
                env=BrowserEnv(page_store=store),
            )
            self.assertTrue(model_trajectory["summary"]["success"])


if __name__ == "__main__":
    unittest.main()
