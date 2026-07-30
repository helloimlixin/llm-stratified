import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.cached_var_shell_lrt import analyze_cached_codebook, nearest_neighbor_distances


class CachedVarShellLrtTests(unittest.TestCase):
    def test_nearest_neighbor_distances_exclude_self_and_are_sorted(self):
        codebook = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        distances = nearest_neighbor_distances(codebook, neighbors=2, chunk_size=2)

        self.assertEqual(distances.shape, (4, 2))
        self.assertTrue(np.all(np.isfinite(distances)))
        self.assertTrue(np.all(distances[:, 0] <= distances[:, 1]))
        self.assertTrue(np.all(distances > 0.0))

    def test_cached_analysis_emits_shell_lrt_arrays(self):
        rng = np.random.default_rng(11)
        codebook = rng.normal(size=(64, 6)).astype(np.float32)
        codebook /= np.linalg.norm(codebook, axis=1, keepdims=True)
        pca = codebook[:, :2]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arrays.npz"
            np.savez_compressed(path, normalized_codebook=codebook, pca=pca)
            summary, arrays = analyze_cached_codebook(
                path,
                neighbors=16,
                bins=4,
                alpha=0.05,
                calibration_trials=500,
                seed=13,
            )

        self.assertEqual(arrays["shell_counts"].shape, (64, 4))
        self.assertTrue(np.isfinite(arrays["scores"]).all())
        self.assertIn("shell_lrt_reject_fraction", summary)


if __name__ == "__main__":
    unittest.main()
