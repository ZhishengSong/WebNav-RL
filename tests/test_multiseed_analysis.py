from __future__ import annotations

import unittest

from eval.multiseed_analysis import mcnemar_exact_p


class MultiseedAnalysisTests(unittest.TestCase):
    def test_mcnemar_matches_known_result(self) -> None:
        self.assertAlmostEqual(mcnemar_exact_p(12, 5), 0.143463134765625)

    def test_mcnemar_no_discordant_pairs(self) -> None:
        self.assertEqual(mcnemar_exact_p(0, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
