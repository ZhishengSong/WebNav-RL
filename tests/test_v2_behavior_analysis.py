from __future__ import annotations

import unittest

from eval.v2_behavior_analysis import candidate_position


class V2BehaviorAnalysisTests(unittest.TestCase):
    def test_candidate_position(self) -> None:
        page = {"elements": [{"element_id": "a"}, {"element_id": "b"}]}
        self.assertEqual(candidate_position(page, "a"), 1)
        self.assertEqual(candidate_position(page, "b"), 2)
        self.assertIsNone(candidate_position(page, "missing"))
        self.assertIsNone(candidate_position(page, None))


if __name__ == "__main__":
    unittest.main()
