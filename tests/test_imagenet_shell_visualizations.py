import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from imagenet_shell_visualizations import (  # noqa: E402
    equal_mass_edges,
    fit_radial_dimension,
    kl_to_uniform,
    shell_counts,
)


class TestImagenetShellVisualizations(unittest.TestCase):
    def test_fit_radial_dimension_recovers_power_law_grid(self):
        dim = 3.0
        log_moments = np.linspace(0.5, 1.5, 200)
        distances = np.exp(-log_moments / dim)

        estimate = fit_radial_dimension(distances, outer_radius=1.0)

        self.assertAlmostEqual(estimate, dim)

    def test_equal_mass_edges_make_balanced_counts_for_quantiles(self):
        dim = 4.0
        bins = 8
        per_bin = 10
        centers = (np.arange(bins * per_bin) + 0.5) / (bins * per_bin)
        distances = centers ** (1.0 / dim)
        edges = equal_mass_edges(dim, bins, radius=1.0)

        counts = shell_counts(distances, edges)

        self.assertTrue(np.all(counts == per_bin))
        self.assertAlmostEqual(kl_to_uniform(counts), 0.0)


if __name__ == "__main__":
    unittest.main()
