import unittest

import numpy as np

from scripts.radial_uniformity_hypothesis_experiment import (
    exponential_ad_critical,
    exponential_ad_rows,
)


class RadialUniformityExperimentTests(unittest.TestCase):
    def test_exponential_quantile_grid_is_below_critical_value(self):
        probabilities = (np.arange(1, 201, dtype=np.float64) - 0.5) / 200.0
        values = -np.log1p(-probabilities)
        statistic = float(exponential_ad_rows(values[None, :])[0])
        self.assertLess(statistic, exponential_ad_critical(200))

    def test_nonexponential_shape_has_larger_statistic(self):
        probabilities = (np.arange(1, 201, dtype=np.float64) - 0.5) / 200.0
        exponential = -np.log1p(-probabilities)
        curved = exponential**2
        statistics = exponential_ad_rows(np.stack([exponential, curved]))
        self.assertLess(float(statistics[0]), float(statistics[1]))


if __name__ == "__main__":
    unittest.main()
