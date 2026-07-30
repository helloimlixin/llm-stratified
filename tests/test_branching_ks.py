import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fiber.branching_ks import (  # noqa: E402
    branch_metrics,
    branch_posteriors,
    fit_kmeans,
    ks_2samp,
    sliced_ks_test,
)


class TestBranchingKS(unittest.TestCase):
    def test_branch_entropy_is_higher_for_boundary_points(self):
        features = np.asarray(
            [
                [-3.0, 0.0],
                [-2.8, 0.1],
                [3.0, 0.0],
                [2.8, -0.1],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )
        prototypes = np.asarray([[-3.0, 0.0], [3.0, 0.0]], dtype=np.float64)

        post = branch_posteriors(features, prototypes, temperature=0.5)
        metrics = branch_metrics(post)

        self.assertGreater(metrics["branch_entropy_norm"][4], metrics["branch_entropy_norm"][0])
        self.assertLess(metrics["branch_margin"][4], metrics["branch_margin"][0])

    def test_ks_2samp_detects_shift(self):
        left = np.linspace(0.0, 1.0, 50)
        right = np.linspace(2.0, 3.0, 50)

        result = ks_2samp(left, right)

        self.assertAlmostEqual(result.statistic, 1.0)
        self.assertLess(result.pvalue, 1e-6)

    def test_sliced_ks_detects_projected_shift(self):
        rng = np.random.default_rng(0)
        group_a = rng.normal(loc=0.0, scale=0.15, size=(48, 4))
        group_b = rng.normal(loc=1.0, scale=0.15, size=(48, 4))
        features = np.vstack([group_a, group_b])
        mask = np.zeros(features.shape[0], dtype=np.bool_)
        mask[group_a.shape[0]:] = True

        result = sliced_ks_test(features, mask, projections=24, permutations=16, seed=1)

        self.assertGreater(result.median_statistic, 0.5)
        self.assertLessEqual(result.permutation_pvalue, 0.2)

    def test_fit_kmeans_returns_requested_clusters(self):
        rng = np.random.default_rng(3)
        features = rng.normal(size=(20, 3))

        centers, labels = fit_kmeans(features, n_clusters=4, seed=3, iters=5)

        self.assertEqual(centers.shape, (4, 3))
        self.assertEqual(labels.shape, (20,))


if __name__ == "__main__":
    unittest.main()
