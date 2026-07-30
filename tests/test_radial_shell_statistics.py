import math
import unittest

import numpy as np

from scripts.radial_shell_statistics import (
    calibrate_fitted_shell_deviance,
    fitted_equal_mass_shell_counts,
    fitted_equal_mass_shell_deviance,
    monte_carlo_pvalue,
    shell_deviance,
)


class RadialShellStatisticsTests(unittest.TestCase):
    def test_multinomial_deviance_matches_definition(self):
        self.assertAlmostEqual(shell_deviance(np.asarray([5, 5])), 0.0)
        self.assertAlmostEqual(shell_deviance(np.asarray([10, 0])), 20.0 * math.log(2.0))

    def test_fitted_shell_statistic_is_scale_invariant(self):
        log_radii = np.asarray([0.11, 0.19, 0.27, 0.42, 0.73, 1.1, 1.9, 2.8])

        counts = fitted_equal_mass_shell_counts(log_radii, bins=4)
        rescaled_counts = fitted_equal_mass_shell_counts(7.5 * log_radii, bins=4)

        self.assertEqual(counts.tolist(), rescaled_counts.tolist())
        self.assertAlmostEqual(
            float(fitted_equal_mass_shell_deviance(log_radii, bins=4)),
            float(fitted_equal_mass_shell_deviance(7.5 * log_radii, bins=4)),
        )

    def test_monte_carlo_calibration_returns_ordered_tail_probabilities(self):
        critical, null = calibrate_fitted_shell_deviance(
            samples=31,
            bins=4,
            alpha=0.05,
            trials=1000,
            seed=19,
            batch_size=200,
        )

        self.assertEqual(null.shape, (1000,))
        self.assertTrue(np.all(null[:-1] <= null[1:]))
        self.assertGreater(critical, 0.0)
        self.assertGreater(monte_carlo_pvalue(float(null[0]), null), 0.9)
        self.assertLess(monte_carlo_pvalue(float(null[-1]) + 1.0, null), 0.01)


if __name__ == "__main__":
    unittest.main()
