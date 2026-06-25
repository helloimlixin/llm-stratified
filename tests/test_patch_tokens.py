import sys
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fiber.patch_tokens import collect_patch_tokens


class ToyPatchModel(nn.Module):
    has_dist_token = False
    embed_dim = 3

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 2)

    def forward_features(self, imgs):
        batch = int(imgs.shape[0])
        patches = 4
        return torch.arange(batch * (patches + 1) * 3, dtype=torch.float32, device=imgs.device).view(
            batch,
            patches + 1,
            3,
        )


class TestPatchTokenCollection(unittest.TestCase):
    def test_collect_patch_tokens_keeps_whole_images_under_cap(self):
        imgs = torch.randn(3, 3, 8, 8)
        labels = torch.tensor([0, 1, 0], dtype=torch.long)
        loader = DataLoader(TensorDataset(imgs, labels), batch_size=3)
        model = ToyPatchModel()

        embeddings, labels_out, images, bboxes, patch_indices, image_ids, pred_labels = collect_patch_tokens(
            model,
            loader,
            torch.device("cpu"),
            patch_size=4,
            max_tokens=6,
        )

        self.assertEqual(tuple(embeddings.shape), (4, 3))
        self.assertEqual(tuple(labels_out.shape), (4,))
        self.assertEqual(tuple(images.shape), (1, 3, 8, 8))
        self.assertEqual(tuple(bboxes.shape), (4, 4))
        self.assertEqual(patch_indices.tolist(), [0, 1, 2, 3])
        self.assertEqual(image_ids.tolist(), [0, 0, 0, 0])
        self.assertEqual(tuple(pred_labels.shape), (4,))

    def test_collect_patch_tokens_collects_multiple_full_images_when_aligned(self):
        imgs = torch.randn(3, 3, 8, 8)
        labels = torch.tensor([0, 1, 0], dtype=torch.long)
        loader = DataLoader(TensorDataset(imgs, labels), batch_size=3)
        model = ToyPatchModel()

        embeddings, _labels, images, _bboxes, _patch_indices, image_ids, _pred_labels = collect_patch_tokens(
            model,
            loader,
            torch.device("cpu"),
            patch_size=4,
            max_tokens=8,
        )

        self.assertEqual(int(embeddings.shape[0]), 8)
        self.assertEqual(int(images.shape[0]), 2)
        self.assertEqual(image_ids.tolist(), [0, 0, 0, 0, 1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
