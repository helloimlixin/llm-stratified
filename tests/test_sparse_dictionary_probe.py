import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fiber.sparse_probe import (
    auto_radius_from_knn,
    omp_required_sparsity,
    run_sparse_dictionary_probe,
)


class TestSparseDictionaryProbe(unittest.TestCase):
    def test_auto_radius_uses_knn_quantile(self):
        dists = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 4.0, 5.0],
                [2.0, 4.0, 0.0, 6.0],
                [3.0, 5.0, 6.0, 0.0],
            ],
            dtype=np.float64,
        )

        radius = auto_radius_from_knn(dists, neighbor_k=1, quantile=0.5)

        self.assertAlmostEqual(radius, 1.5)

    def test_omp_required_sparsity_hits_identity_dictionary_solution(self):
        dictionary = np.eye(3, dtype=np.float64)
        target = np.array([1.0, 1.0, 0.0], dtype=np.float64)

        sparsity, residual = omp_required_sparsity(
            target,
            dictionary,
            residual_threshold=1e-8,
            max_sparsity=3,
        )

        self.assertEqual(sparsity, 2)
        self.assertLessEqual(residual, 1e-8)

    def test_run_sparse_dictionary_probe_writes_summary(self):
        embeddings = torch.tensor(
            [[0.0], [0.2], [0.4], [0.6], [2.0], [2.2], [2.4], [2.6]],
            dtype=torch.float32,
        )
        dists = torch.cdist(embeddings, embeddings).numpy()
        images = torch.arange(2 * 1 * 4 * 4, dtype=torch.float32).reshape(2, 1, 4, 4)
        image_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64)
        bboxes = torch.tensor(
            [
                [0, 0, 2, 2],
                [2, 0, 4, 2],
                [0, 2, 2, 4],
                [2, 2, 4, 4],
                [0, 0, 2, 2],
                [2, 0, 4, 2],
                [0, 2, 2, 4],
                [2, 2, 4, 4],
            ],
            dtype=torch.int32,
        )
        fiber_results = [
            {"dimensions": [1.0 + 0.1 * i], "pvalues": [0.5 if i % 2 == 0 else 0.001]}
            for i in range(8)
        ]

        with TemporaryDirectory() as tmpdir:
            result = run_sparse_dictionary_probe(
                epoch=3,
                embeddings=embeddings,
                images=images,
                image_ids=image_ids,
                bboxes=bboxes,
                fiber_results=fiber_results,
                dists=dists,
                out_dir=Path(tmpdir),
                patch_size=2,
                radius=0.7,
                min_patches=2,
                max_anchors=4,
                dictionary_size=3,
                residual_threshold=0.5,
                max_sparsity=3,
            )
            summary = result["summary"]
            json_path = Path(summary["json_path"])

            self.assertTrue(json_path.exists())
            self.assertGreaterEqual(summary["evaluated_anchors"], 2)
            self.assertGreater(summary["mean_patch_count"], 0)
            self.assertGreaterEqual(summary["mean_required_sparsity"], 0)


if __name__ == "__main__":
    unittest.main()
