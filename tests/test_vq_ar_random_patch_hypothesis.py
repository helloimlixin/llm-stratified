import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vq_ar_random_patch_hypothesis import (  # noqa: E402
    balanced_random_patch_diffs,
    enrichment_stat,
    mean_diff,
    one_sided_p,
    within_image_permutation_null,
)


class TestVqArRandomPatchHypothesis(unittest.TestCase):
    def test_balanced_random_patch_diffs_respect_group_direction(self):
        values = np.asarray([0.1, 0.2, 0.3, 1.1, 1.2, 1.3], dtype=float)
        selector = np.asarray([True, True, True, False, False, False])

        diffs = balanced_random_patch_diffs(values, selector, reps=50, sample_per_group=2, seed=123)

        self.assertTrue(np.all(diffs < 0.0))
        self.assertLess(float(np.mean(diffs)), -0.8)

    def test_within_image_permutation_null_preserves_observation_count(self):
        values = np.asarray([0.1, 0.2, 0.9, 1.0, 0.3, 0.4, 1.1, 1.2], dtype=float)
        selector = np.asarray([True, True, False, False, True, True, False, False])
        image_ids = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])

        null = within_image_permutation_null(values, selector, image_ids, reps=25, seed=456)

        self.assertEqual(null.shape, (25,))
        self.assertTrue(np.all(np.isfinite(null)))

    def test_one_sided_p_uses_requested_tail(self):
        null = np.asarray([-0.2, -0.1, 0.0, 0.1, 0.2])

        self.assertLess(one_sided_p(null, -0.15, "lower"), one_sided_p(null, -0.15, "higher"))
        self.assertLess(one_sided_p(null, 0.15, "higher"), one_sided_p(null, 0.15, "lower"))

    def test_enrichment_stat_is_flat_rate_minus_rest_rate(self):
        flat = np.asarray([True, True, False, False])
        selector = np.asarray([True, False, False, False])

        self.assertAlmostEqual(enrichment_stat(flat, selector), 0.5)
        self.assertAlmostEqual(mean_diff(np.asarray([1.0, 2.0, 3.0]), np.asarray([True, False, False])), -1.5)


if __name__ == "__main__":
    unittest.main()
