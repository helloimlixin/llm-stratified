import json
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vq_ar_polysemy_branch_stats import (  # noqa: E402
    binomial_tail_probability,
    load_pairs,
    paired_summary,
)


class TestVqArPolysemyBranchStats(unittest.TestCase):
    def test_binomial_tail_probability_matches_small_exact_case(self):
        self.assertAlmostEqual(binomial_tail_probability(3, 4), 5 / 16)
        self.assertAlmostEqual(binomial_tail_probability(4, 4), 1 / 16)

    def test_load_pairs_reads_singular_and_regular_in_order(self):
        payload = {
            "anchors": [
                {"kind": "singular", "crop_pairwise_l2": 0.5},
                {"kind": "regular", "crop_pairwise_l2": 0.1},
                {"kind": "singular", "crop_pairwise_l2": 0.4},
                {"kind": "regular", "crop_pairwise_l2": 0.2},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            singular, regular = load_pairs(path, "crop_pairwise_l2")

        np.testing.assert_allclose(singular, [0.5, 0.4])
        np.testing.assert_allclose(regular, [0.1, 0.2])

    def test_paired_summary_reports_wins_and_positive_effect(self):
        summary = paired_summary(
            np.asarray([0.5, 0.4, 0.6]),
            np.asarray([0.1, 0.2, 0.3]),
            branches=4,
            reps=100,
            seed=123,
        )

        self.assertEqual(summary["wins"], 3)
        self.assertEqual(summary["n_pairs"], 3)
        self.assertGreater(summary["mean_diff"], 0.0)
        self.assertEqual(summary["branch_variants_excluding_originals"], 24)


if __name__ == "__main__":
    unittest.main()
