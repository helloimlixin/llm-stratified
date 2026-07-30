import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from volume_shell_hypothesis_experiment import (  # noqa: E402
    finite_sample_kl_threshold,
    kl_divergence,
    run_experiment,
    shell_edges_equal_null_mass,
    shell_probabilities,
)


class TestVolumeShellHypothesisExperiment(unittest.TestCase):
    def test_equal_null_mass_edges_make_uniform_shell_probabilities(self):
        edges = shell_edges_equal_null_mass(bins=8, dimension=4.0, radius=1.0)
        probs = shell_probabilities(edges=edges, radial_dimension=4.0, radius=1.0)

        self.assertTrue(np.allclose(probs, np.full(8, 1.0 / 8.0)))
        self.assertAlmostEqual(float(probs.sum()), 1.0)

    def test_kl_direction_zero_only_when_empirical_matches_null(self):
        p = np.asarray([0.25, 0.25, 0.25, 0.25])
        q_same = np.asarray([0.25, 0.25, 0.25, 0.25])
        q_diff = np.asarray([0.10, 0.20, 0.30, 0.40])

        self.assertAlmostEqual(float(kl_divergence(q_same, p)[0]), 0.0)
        self.assertGreater(float(kl_divergence(q_diff, p)[0]), 0.0)

    def test_finite_sample_threshold_decreases_with_sample_size(self):
        small = finite_sample_kl_threshold(samples=64, bins=8, alpha=0.05)
        large = finite_sample_kl_threshold(samples=512, bins=8, alpha=0.05)

        self.assertGreater(small, large)

    def test_smoke_run_includes_null_and_alternatives(self):
        payload = run_experiment(
            dimension=4.0,
            bins=8,
            sample_sizes=[32],
            trials=20,
            calibration_trials=50,
            alpha=0.05,
            radius=1.0,
            seed=123,
        )

        self.assertEqual(len(payload["summary"]), 3)
        names = {row["scenario"] for row in payload["summary"]}
        self.assertIn("null_d4", names)
        self.assertIn("inner_heavy_d2", names)
        self.assertIn("outer_heavy_d8", names)


if __name__ == "__main__":
    unittest.main()
