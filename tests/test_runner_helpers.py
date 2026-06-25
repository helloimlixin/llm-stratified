import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.runner import _offset_local_image_ids_for_gather, _trim_to_image_aligned_tokens


class FakeAccelerator:
    def __init__(self, counts):
        self.counts = torch.tensor(counts, dtype=torch.long)

    def gather_for_metrics(self, tensor):
        return self.counts.to(tensor.device)


class TestRunnerHelpers(unittest.TestCase):
    def test_offsets_image_ids_by_prior_rank_image_counts(self):
        img_ids = torch.tensor([0, 0, 1, 2], dtype=torch.int32)
        images = torch.zeros(3, 3, 4, 4)
        shifted = _offset_local_image_ids_for_gather(
            img_ids,
            images,
            device=torch.device("cpu"),
            rank=2,
            world_size=3,
            accelerator=FakeAccelerator([2, 5, 3]),
        )
        self.assertEqual(shifted.tolist(), [7, 7, 8, 9])

    def test_trim_to_image_aligned_tokens_drops_partial_boundary_image(self):
        embeddings = torch.arange(12, dtype=torch.float32).view(12, 1)
        labels = torch.arange(12)
        images = torch.randn(3, 3, 4, 4)
        bboxes = torch.zeros(12, 4, dtype=torch.int32)
        patch_indices = torch.arange(12, dtype=torch.int32)
        img_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.int32)
        pred_labels = torch.arange(12, dtype=torch.int64)

        trimmed = _trim_to_image_aligned_tokens(
            embeddings,
            labels,
            images,
            bboxes,
            patch_indices,
            img_ids,
            pred_labels,
            max_tokens=10,
        )

        trimmed_embeddings, trimmed_labels, trimmed_images, trimmed_bboxes, trimmed_patch_indices, trimmed_img_ids, _ = trimmed
        self.assertEqual(int(trimmed_embeddings.shape[0]), 8)
        self.assertEqual(int(trimmed_labels.shape[0]), 8)
        self.assertEqual(int(trimmed_bboxes.shape[0]), 8)
        self.assertEqual(int(trimmed_patch_indices.shape[0]), 8)
        self.assertEqual(int(trimmed_images.shape[0]), 2)
        self.assertEqual(trimmed_img_ids.tolist(), [0, 0, 0, 0, 1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
