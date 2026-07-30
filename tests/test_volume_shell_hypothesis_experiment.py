import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from volume_shell_hypothesis_experiment import (  # noqa: E402
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
        self.assertEqual(names, {"null", "inner_heavy", "outer_heavy"})
        self.assertTrue(all("critical_value" in row for row in payload["summary"]))


if __name__ == "__main__":
    unittest.main()
