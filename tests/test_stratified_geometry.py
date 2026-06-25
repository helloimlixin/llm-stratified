import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fiber.geometry import min_fiber_violation_pvalue, run_fiber_bundle_test, summarize_stratification


class TestStratifiedGeometry(unittest.TestCase):
    def test_run_fiber_bundle_test_smoke(self):
        torch.manual_seed(0)
        embeddings = torch.randn(12, 4)
        results, sorted_dists, unsorted_dists = run_fiber_bundle_test(embeddings, vol_min=2, vol_max=6, ws=1, alpha=0.01, nstrat=2)
        self.assertEqual(len(results), 12)
        self.assertIn("dimensions", results[0])
        self.assertEqual(sorted_dists.shape, (12, 12))
        self.assertEqual(unsorted_dists.shape, (12, 12))

    def test_collapsed_embeddings_do_not_feed_zero_radii_to_regression(self):
        embeddings = torch.zeros(8, 4)
        results, sorted_dists, _unsorted_dists = run_fiber_bundle_test(
            embeddings,
            vol_min=2,
            vol_max=6,
            ws=1,
            alpha=0.01,
            nstrat=2,
        )
        self.assertEqual(len(results), 8)
        self.assertTrue(all(not res["dimensions"] for res in results))
        self.assertTrue(torch.as_tensor(sorted_dists).eq(0).all())

    def test_fiber_irregularity_requires_slope_increase(self):
        decreasing = {"dimensions": [3.0, 1.0], "pvalues": [1e-6, 1.0]}
        increasing = {"dimensions": [1.0, 3.0], "pvalues": [1e-6, 1.0]}

        self.assertTrue(torch.isnan(torch.tensor(min_fiber_violation_pvalue(decreasing))))
        self.assertAlmostEqual(min_fiber_violation_pvalue(increasing), 1e-6)

        summary = summarize_stratification([decreasing, increasing], alpha=1e-3)
        self.assertEqual(summary["change_point_ratio"], 1.0)
        self.assertEqual(summary["irregular_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
