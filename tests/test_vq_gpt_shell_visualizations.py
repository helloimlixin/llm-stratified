import math
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vq_gpt_shell_visualizations import (  # noqa: E402
    equal_mass_edges,
    fit_radial_dimension,
    kl_to_uniform,
    mean_diff,
    shell_counts,
)


class TestVqGptShellVisualizations(unittest.TestCase):
    def test_equal_mass_edges_make_equal_shell_probabilities(self):
        edges = equal_mass_edges(4.0, bins=8, radius=2.0)
        probs = np.diff((edges / 2.0) ** 4.0)

        self.assertTrue(np.allclose(probs, np.full(8, 1.0 / 8.0)))

    def test_shell_counts_and_kl_are_zero_for_equal_counts(self):
        edges = np.asarray([0.0, 1.0, 2.0, 3.0])
        counts = shell_counts(np.asarray([0.5, 1.5, 2.5, 0.8, 1.8, 2.8]), edges)

        self.assertEqual(counts.tolist(), [2, 2, 2])
        self.assertAlmostEqual(kl_to_uniform(counts), 0.0)

    def test_fit_radial_dimension_matches_exact_power_grid(self):
        u = np.linspace(0.1, 0.9, 9)
        d_true = 3.0
        radii = u ** (1.0 / d_true)
        d_hat = fit_radial_dimension(radii, 1.0)
        expected = len(radii) / np.sum(-np.log(radii))

        self.assertTrue(math.isfinite(d_hat))
        self.assertAlmostEqual(d_hat, expected)

    def test_mean_diff_is_selected_minus_rest(self):
        values = np.asarray([1.0, 2.0, 10.0, 12.0])
        selector = np.asarray([False, False, True, True])

        self.assertAlmostEqual(mean_diff(values, selector), 9.5)


if __name__ == "__main__":
    unittest.main()
