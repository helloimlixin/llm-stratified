import sys
from pathlib import Path
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pretrained_vq_codebook_stratification_probe import (  # noqa: E402
    chunked_sorted_neighbor_distances,
    extract_codebook_embeddings,
    finite_minimum,
    paper_style_sliding_welch_tests,
    scan_stratification_band,
)


class DummyQuantizer:
    def __init__(self, weight: torch.Tensor, *, l2_norm: bool):
        self.embedding = torch.nn.Embedding.from_pretrained(weight.clone(), freeze=False)
        self.l2_norm = l2_norm


class DummyVqModel(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, *, l2_norm: bool = True):
        super().__init__()
        self.quantize = DummyQuantizer(weight, l2_norm=l2_norm)


class TestPretrainedVqCodebookStratificationProbe(unittest.TestCase):
    def test_extract_codebook_embeddings_uses_quantizer_normalization(self):
        weight = torch.tensor([[3.0, 4.0], [0.0, 2.0], [1.0, 0.0]], dtype=torch.float32)
        model = DummyVqModel(weight, l2_norm=True)

        embeddings, source_key, normalized, quantizer_l2_norm = extract_codebook_embeddings(
            model,
            expected_size=3,
            expected_dim=2,
            geometry="quantizer",
        )

        self.assertEqual(source_key, "quantize.embedding.weight")
        self.assertTrue(normalized)
        self.assertTrue(quantizer_l2_norm)
        self.assertTrue(np.allclose(np.linalg.norm(embeddings, axis=1), 1.0))

    def test_chunked_sorted_neighbor_distances_matches_direct_distances(self):
        features = np.asarray([[0.0], [2.0], [5.0], [9.0]], dtype=np.float32)

        distances, indices = chunked_sorted_neighbor_distances(
            features,
            max_neighbors=3,
            chunk_size=2,
            device=torch.device("cpu"),
            include_self=True,
        )

        self.assertEqual(distances.shape, (4, 3))
        self.assertEqual(indices.shape, (4, 3))
        self.assertTrue(np.allclose(distances[0], [0.0, 2.0, 5.0]))
        self.assertEqual(indices[0].tolist(), [0, 1, 2])
        self.assertTrue(np.all(np.diff(distances, axis=1) >= -1e-6))

    def test_finite_minimum_ignores_nan_entries(self):
        result = finite_minimum(
            [
                np.asarray([np.nan, 0.4, 0.2]),
                np.asarray([0.1, np.nan, 0.3]),
            ]
        )

        self.assertTrue(np.allclose(result, np.asarray([0.1, 0.4, 0.2]), equal_nan=True))

    def test_paper_style_sliding_welch_detects_slope_increase(self):
        volumes = np.arange(10, 110, dtype=np.float64)
        dims = np.where(np.arange(volumes.size) < 50, 1.0, 4.0)
        radii = volumes ** (1.0 / dims)

        result = paper_style_sliding_welch_tests(radii, volumes, ws=5)

        self.assertLess(float(result["fiber_pvalue"]), 0.05)
        self.assertGreater(float(result["fiber_delta"]), 0.0)
        self.assertIsNotNone(result["fiber_index"])

    def test_scan_stratification_band_returns_code_level_arrays(self):
        base = np.linspace(0.0, 1.0, 80)
        sorted_distances = np.vstack([base, base ** 1.2 + 1e-4, base ** 0.8 + 1e-4])

        band = scan_stratification_band(
            name="test",
            sorted_distances=sorted_distances,
            vol_min=5,
            vol_max=60,
            window_size=4,
            alpha=0.05,
        )

        self.assertEqual(band.manifold_pvalue.shape, (3,))
        self.assertEqual(band.fiber_pvalue.shape, (3,))
        self.assertEqual(band.dimension.shape, (3,))
        self.assertEqual(band.dimvec.shape[0], 3)


if __name__ == "__main__":
    unittest.main()
