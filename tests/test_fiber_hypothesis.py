import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fiber.plots import HAS_MATPLOTLIB, build_embedding_scatter_figure
from fiber.hypothesis import compute_stratified_manifold_hypothesis_metrics


class TestFiberHypothesis(unittest.TestCase):
    def test_compute_hypothesis_metrics_detects_same_image_fiber_signal(self):
        embeddings = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
                [10.0, 10.0],
                [10.1, 10.0],
                [10.2, 10.0],
            ],
            dtype=torch.float32,
        )
        fiber_results = [
            {"dimensions": [2.0], "pvalues": [0.20]},
            {"dimensions": [2.1, 2.6], "pvalues": [0.001, 0.20]},
            {"dimensions": [1.9], "pvalues": [0.30]},
            {"dimensions": [3.0], "pvalues": [0.25]},
            {"dimensions": [3.1, 3.5], "pvalues": [0.005, 0.30]},
            {"dimensions": [2.9], "pvalues": [0.40]},
        ]
        image_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64)
        bboxes = torch.tensor(
            [
                [0, 0, 1, 1],
                [2, 0, 3, 1],
                [4, 0, 5, 1],
                [0, 0, 1, 1],
                [2, 0, 3, 1],
                [4, 0, 5, 1],
            ],
            dtype=torch.int32,
        )
        neighborhood_dims = [2.05, 2.10, 2.00, 3.00, 3.05, 2.95]

        metrics = compute_stratified_manifold_hypothesis_metrics(
            embeddings=embeddings,
            fiber_results=fiber_results,
            image_ids=image_ids,
            bboxes=bboxes,
            neighborhood_dims=neighborhood_dims,
            neighborhood_size=4,
            alpha=0.01,
        )

        self.assertEqual(metrics["neighbor_k"], 2)
        self.assertGreater(metrics["same_image_neighbor_ratio_mean"], metrics["same_image_neighbor_chance"])
        self.assertGreater(metrics["same_image_neighbor_lift"], 1.0)
        self.assertGreaterEqual(metrics["local_same_image_given_same_image_mean"], 0.5)
        self.assertGreater(metrics["multi_strata_ratio"], 0.0)
        self.assertGreaterEqual(metrics["hypothesis_score"], 0.0)
        self.assertLessEqual(metrics["hypothesis_score"], 1.0)
        self.assertIsInstance(metrics["hypothesis_narrative"], str)

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is required for figure generation")
    def test_build_embedding_scatter_figure_falls_back_to_2d(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.2], [2.0, 1.0, 0.4]], dtype=np.float64)
        dims = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        with patch("fiber.plots.matplotlib_supports_3d", return_value=False):
            fig = build_embedding_scatter_figure(coords, dims)
            try:
                self.assertNotEqual(fig.axes[0].name, "3d")
                self.assertIn("fallback", fig.axes[0].get_title().lower())
            finally:
                import matplotlib.pyplot as plt

                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
