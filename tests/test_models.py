import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import get_dataset_normalization
from models import SamImageEncoder, TinyViT, resolve_patch_size


class TestModels(unittest.TestCase):
    def test_tinyvit_forward_shape(self):
        model = TinyViT(img_size=8, patch_size=4, in_chans=3, num_classes=5, embed_dim=32, depth=2, num_heads=2)
        x = torch.randn(2, 3, 8, 8)
        y = model(x)
        self.assertEqual(list(y.shape), [2, 5])

    def test_resolve_patch_size_variants(self):
        class Dummy:
            pass

        d1 = Dummy()
        d1.patch_embed = Dummy()
        d1.patch_embed.patch_size = 4
        self.assertEqual(resolve_patch_size(d1), 4)

        d2 = Dummy()
        d2.patch_embed = Dummy()
        d2.patch_embed.patch_size = (16, 16)
        self.assertEqual(resolve_patch_size(d2), 16)

        d3 = Dummy()
        d3.patch_embed = Dummy()
        d3.patch_embed.patch_size = torch.tensor([8, 8])
        self.assertEqual(resolve_patch_size(d3), 8)

    def test_sam_numpy_image_uses_uint8_pixel_scale(self):
        raw = torch.tensor(
            [
                [[0.0, 0.5], [1.0, 0.25]],
                [[0.25, 1.0], [0.5, 0.0]],
                [[1.0, 0.0], [0.25, 0.5]],
            ],
            dtype=torch.float32,
        )
        mean, std = get_dataset_normalization("FAKEDATA", as_tensor=True)
        normalized = (raw - mean.view(3, 1, 1)) / std.view(3, 1, 1)
        encoder = SamImageEncoder.__new__(SamImageEncoder)

        np_img, img01 = encoder._image_numpy(normalized, "FAKEDATA")

        self.assertEqual(np_img.dtype.name, "uint8")
        self.assertEqual(np_img.shape, (2, 2, 3))
        self.assertTrue(torch.allclose(img01, raw, atol=1e-6))
        self.assertEqual(int(np_img.max()), 255)
        self.assertEqual(int(np_img[0, 1, 0]), 128)


if __name__ == "__main__":
    unittest.main()
