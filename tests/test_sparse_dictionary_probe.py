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
    iht_required_sparsity,
    omp_required_sparsity,
    run_neighborhood_dictionary_volume_curve,
    run_sparse_dictionary_probe,
    select_probe_tokens,
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

    def test_iht_required_sparsity_hits_identity_dictionary_solution(self):
        dictionary = np.eye(3, dtype=np.float64)
        target = np.array([1.0, 1.0, 0.0], dtype=np.float64)

        sparsity, residual = iht_required_sparsity(
            target,
            dictionary,
            residual_threshold=1e-6,
            max_sparsity=3,
            steps=30,
        )

        self.assertEqual(sparsity, 2)
        self.assertLessEqual(residual, 1e-6)

    def test_select_probe_tokens_defaults_to_all_tokens(self):
        embeddings = torch.randn(7, 3)

        all_tokens = select_probe_tokens(embeddings, max_tokens=None)
        capped_tokens = select_probe_tokens(embeddings, max_tokens=3)

        np.testing.assert_array_equal(all_tokens, np.arange(7))
        np.testing.assert_array_equal(capped_tokens, np.array([0, 3, 6]))

    def test_select_probe_tokens_keeps_complete_images_when_image_ids_provided(self):
        embeddings = torch.randn(12, 3)
        image_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.int64)

        under_cap = select_probe_tokens(embeddings, max_tokens=6, image_ids=image_ids)
        aligned_cap = select_probe_tokens(embeddings, max_tokens=8, image_ids=image_ids)
        tiny_cap = select_probe_tokens(embeddings, max_tokens=2, image_ids=image_ids)

        np.testing.assert_array_equal(under_cap, np.array([0, 1, 2, 3]))
        np.testing.assert_array_equal(aligned_cap, np.arange(8))
        np.testing.assert_array_equal(tiny_cap, np.array([0, 1, 2, 3]))

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
                algorithm="iht",
                iht_steps=20,
                heatmap_max_images=2,
            )
            summary = result["summary"]
            json_path = Path(summary["json_path"])

            self.assertTrue(json_path.exists())
            self.assertEqual(summary["dictionary_mode"], "local")
            self.assertEqual(summary["coding_algorithm"], "iht")
            self.assertEqual(summary["candidate_tokens"], 8)
            self.assertGreaterEqual(summary["evaluated_anchors"], 2)
            self.assertGreaterEqual(summary["evaluated_tokens"], 2)
            self.assertGreater(summary["mean_patch_count"], 0)
            self.assertGreaterEqual(summary["mean_required_sparsity"], 0)
            self.assertIn("tokens", result)
            self.assertNotIn("fiber_coord", result["tokens"][0])
            self.assertIn("neighborhood_token_indices", result["tokens"][0])
            self.assertEqual(
                len(result["tokens"][0]["neighborhood_token_indices"]),
                result["tokens"][0]["patch_count"],
            )
            for artifact_key in ("plot_path", "heatmap_path"):
                if artifact_key in summary:
                    self.assertTrue(Path(summary[artifact_key]).exists())

    def test_sparse_dictionary_probe_supports_fixed_neighbor_volume(self):
        embeddings = torch.arange(10, dtype=torch.float32).view(10, 1)
        dists = torch.cdist(embeddings, embeddings).numpy()
        images = torch.randn(2, 1, 4, 4)
        image_ids = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.int64)
        bboxes = torch.tensor([[0, 0, 2, 2]] * 10, dtype=torch.int32)
        fiber_results = [{"dimensions": [1.0], "pvalues": [0.5]} for _ in range(10)]

        with TemporaryDirectory() as tmpdir:
            result = run_sparse_dictionary_probe(
                epoch=0,
                embeddings=embeddings,
                images=images,
                image_ids=image_ids,
                bboxes=bboxes,
                fiber_results=fiber_results,
                dists=dists,
                out_dir=Path(tmpdir),
                patch_size=2,
                neighbor_k=4,
                min_patches=4,
                max_anchors=3,
                dictionary_size=3,
                residual_threshold=0.5,
                max_sparsity=3,
            )

        summary = result["summary"]
        self.assertEqual(summary["neighborhood_mode"], "knn")
        self.assertEqual(summary["neighbor_k"], 4)
        self.assertIsNone(summary["radius"])
        self.assertTrue(all(row["patch_count"] == 4 for row in result["tokens"]))
        self.assertTrue(all(len(row["neighborhood_token_indices"]) == 4 for row in result["tokens"]))

    def test_neighborhood_dictionary_volume_curve_writes_rows(self):
        embeddings = torch.arange(12, dtype=torch.float32).view(12, 1)
        dists = torch.cdist(embeddings, embeddings).numpy()
        images = torch.randn(3, 1, 4, 4)
        image_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.int64)
        bboxes = torch.tensor(
            [[0, 0, 2, 2], [2, 0, 4, 2], [0, 2, 2, 4], [2, 2, 4, 4]] * 3,
            dtype=torch.int32,
        )

        with TemporaryDirectory() as tmpdir:
            result = run_neighborhood_dictionary_volume_curve(
                epoch=1,
                embeddings=embeddings,
                images=images,
                image_ids=image_ids,
                bboxes=bboxes,
                dists=dists,
                out_dir=Path(tmpdir),
                patch_size=2,
                volumes=[2, 4, 8],
                dictionary_size=4,
                residual_threshold=0.5,
                max_sparsity=4,
            )
            summary = result["summary"]
            self.assertEqual(summary["dictionary_mode"], "per_volume_local")
            self.assertEqual(summary["anchor_mode"], "middle_patch")
            self.assertGreaterEqual(summary["evaluated_volumes"], 1)
            self.assertTrue(all(row["dictionary_patch_count"] == row["neighborhood_volume"] for row in result["rows"]))
            self.assertTrue(Path(summary["json_path"]).exists())
            if "plot_path" in summary:
                self.assertTrue(Path(summary["plot_path"]).exists())


if __name__ == "__main__":
    unittest.main()
