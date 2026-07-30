import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pretrained_var_one_sample_ks import (  # noqa: E402
    one_sample_ks_uniform_draws,
    permuted_categorical_uniform_ks,
    ranked_probability_uniform_ks,
    topk_branch_uniform_ks,
)


class TestPretrainedVarOneSampleKS(unittest.TestCase):
    def test_uniform_distribution_has_smaller_ks_than_peaked_distribution(self):
        uniform = np.full((1, 16), 1.0 / 16.0)
        peaked = np.full((1, 16), 0.02 / 15.0)
        peaked[0, 3] = 0.98
        probs = np.concatenate([uniform, peaked], axis=0)

        stat, pvalue = one_sample_ks_uniform_draws(probs, draws=1024, seed=123)

        self.assertLess(stat[0], stat[1])
        self.assertGreater(pvalue[0], pvalue[1])

    def test_permuted_categorical_ks_is_zero_for_exact_uniform(self):
        uniform = np.full((2, 8), 1.0 / 8.0)

        stats = permuted_categorical_uniform_ks(uniform, permutations=4, seed=123)

        self.assertTrue(np.allclose(stats["median"], 0.0))
        self.assertTrue(np.allclose(stats["trimmed_mean"], 0.0))

    def test_ranked_probability_ks_is_order_free(self):
        uniform = np.full((1, 8), 1.0 / 8.0)
        peaked = np.full((1, 8), 0.2 / 7.0)
        peaked[0, 5] = 0.8
        permuted = peaked[:, [3, 2, 1, 0, 7, 6, 5, 4]]

        self.assertAlmostEqual(float(ranked_probability_uniform_ks(uniform)[0]), 0.0)
        self.assertAlmostEqual(
            float(ranked_probability_uniform_ks(peaked)[0]),
            float(ranked_probability_uniform_ks(permuted)[0]),
        )

    def test_topk_branch_ks_detects_flat_local_branches(self):
        flat = np.asarray([[0.25, 0.25, 0.25, 0.25, 0.0, 0.0]])
        sharp = np.asarray([[0.91, 0.03, 0.03, 0.03, 0.0, 0.0]])

        flat_ks, flat_entropy, flat_mass = topk_branch_uniform_ks(flat, top_k=4)
        sharp_ks, sharp_entropy, sharp_mass = topk_branch_uniform_ks(sharp, top_k=4)

        self.assertLess(float(flat_ks[0]), float(sharp_ks[0]))
        self.assertGreater(float(flat_entropy[0]), float(sharp_entropy[0]))
        self.assertAlmostEqual(float(flat_mass[0]), 1.0)
        self.assertAlmostEqual(float(sharp_mass[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
