import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import TinyViT, resolve_patch_size


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


if __name__ == "__main__":
    unittest.main()
