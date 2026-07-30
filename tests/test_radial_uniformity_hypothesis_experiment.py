import unittest

from scripts.radial_uniformity_hypothesis_experiment import run_experiment


class RadialUniformityExperimentTests(unittest.TestCase):
    def test_experiment_reports_shell_deviance_size_and_power(self):
        result = run_experiment(
            dimension=4.0,
            bins=4,
            sample_sizes=[64],
            trials=500,
            calibration_trials=1000,
            alpha=0.05,
            seed=23,
        )

        rows = {row["scenario"]: row for row in result["summary"]}
        self.assertEqual(set(rows), {"null", "inner_heavy", "outer_heavy"})
        self.assertLess(rows["null"]["rejection_rate"], 0.10)
        self.assertGreater(rows["inner_heavy"]["rejection_rate"], rows["null"]["rejection_rate"])
        self.assertGreater(rows["outer_heavy"]["rejection_rate"], rows["null"]["rejection_rate"])


if __name__ == "__main__":
    unittest.main()
