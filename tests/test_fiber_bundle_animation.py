import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fiber.visualization import HAS_MATPLOTLIB
from fiber.animation import build_embedding_animation_frames, generate_embedding_animation


class TestFiberBundleAnimation(unittest.TestCase):
    def test_build_embedding_animation_frames_preserves_epochs_and_labels(self):
        snapshots = [
            (
                0,
                torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
                torch.tensor([2, 1], dtype=torch.int64),
            ),
            (
                3,
                torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float32),
                torch.tensor([2, 1], dtype=torch.int64),
            ),
        ]

        frames = build_embedding_animation_frames(snapshots)

        self.assertEqual([frame["epoch"] for frame in frames], [0, 3])
        self.assertEqual(frames[0]["encodings"].shape, (2, 3))
        np.testing.assert_array_equal(frames[0]["encodings"][:, 2], np.array([2, 1]))

    def test_build_embedding_animation_frames_handles_multihot_labels(self):
        snapshots = [
            (
                1,
                torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
                torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=torch.float32),
            )
        ]

        frames = build_embedding_animation_frames(snapshots)

        np.testing.assert_array_equal(frames[0]["encodings"][:, 2], np.array([1, 0]))

    def test_build_embedding_animation_frames_uses_last_nonempty_snapshot_for_projection(self):
        snapshots = [
            (
                0,
                torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32),
                torch.tensor([0, 1], dtype=torch.int64),
            ),
            (
                5,
                torch.empty((0, 2), dtype=torch.float32),
                torch.empty((0,), dtype=torch.int64),
            ),
        ]

        frames = build_embedding_animation_frames(snapshots)

        self.assertEqual([frame["epoch"] for frame in frames], [0])
        self.assertFalse(np.allclose(frames[0]["encodings"][:, :2], 0.0))

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is required for GIF generation")
    def test_generate_embedding_animation_writes_gif(self):
        encodings = [
            {"epoch": 0, "encodings": np.array([[0.0, 0.0, 0], [1.0, 0.5, 1]], dtype=np.float64)},
            {"epoch": 2, "encodings": np.array([[0.2, 0.2, 0], [1.2, 0.8, 1]], dtype=np.float64)},
        ]

        with TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "embedding_progression.gif"
            saved_path = generate_embedding_animation(encodings, output_path=out_path, fps=2)

            self.assertEqual(saved_path, out_path)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
