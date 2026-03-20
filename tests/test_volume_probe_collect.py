import sys
from pathlib import Path
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volume_probe import collect_representations


class DummyPatchEmbed(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return torch.arange(batch_size * 4 * 3, dtype=torch.float32).reshape(batch_size, 4, 3)


class DummyViTProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_prefix_tokens = 1
        self.patch_embed = DummyPatchEmbed()

    def forward_features(self, x):
        patches = self.patch_embed(x)
        cls = torch.full((x.shape[0], 1, patches.shape[-1]), -1.0, dtype=patches.dtype)
        return torch.cat([cls, patches], dim=1)


class DummyDinoProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_prefix_tokens = 2
        self.patch_embeddings_include_prefix_tokens = True

    def prepare_images_for_features(self, imgs, dataset_name):
        return imgs, imgs

    def forward_feature_pack(self, imgs):
        batch_size = imgs.shape[0]
        prefix = torch.full((batch_size, 2, 3), -1.0)
        patches = torch.arange(batch_size * 4 * 3, dtype=torch.float32).reshape(batch_size, 4, 3)
        packed = torch.cat([prefix, patches], dim=1)
        return {"tokens": packed, "patch_embeddings": packed.clone()}


class DummyDinoMultiLayerProbeModel(DummyDinoProbeModel):
    def forward_feature_pack(self, imgs):
        batch_size = imgs.shape[0]
        prefix = torch.full((batch_size, 2, 3), -1.0)
        patches = torch.arange(batch_size * 4 * 3, dtype=torch.float32).reshape(batch_size, 4, 3)
        packed = torch.cat([prefix, patches], dim=1)
        return {"tokens_layer_03": packed, "tokens_layer_last": packed.clone(), "patch_embeddings": packed.clone()}


class TestCollectRepresentations(unittest.TestCase):
    def _make_loader(self):
        imgs = torch.randn(2, 3, 4, 4)
        labels = torch.zeros(2, dtype=torch.long)
        return DataLoader(TensorDataset(imgs, labels), batch_size=2, shuffle=False)

    def test_patch_embeddings_keep_all_patches_for_plain_vit_models(self):
        reps, _ = collect_representations(
            model=DummyViTProbeModel(),
            loader=self._make_loader(),
            device=torch.device("cpu"),
            dataset="FAKEDATA",
            patch_size=2,
            pixel_patch_stride=None,
            max_tokens=32,
            show_progress=False,
            viz_images=0,
        )

        self.assertEqual(reps["tokens"].shape[0], 8)
        self.assertEqual(reps["patch_embeddings"].shape[0], 8)

    def test_patch_embeddings_strip_prefix_tokens_when_model_marks_them(self):
        reps, _ = collect_representations(
            model=DummyDinoProbeModel(),
            loader=self._make_loader(),
            device=torch.device("cpu"),
            dataset="FAKEDATA",
            patch_size=2,
            pixel_patch_stride=None,
            max_tokens=32,
            show_progress=False,
            viz_images=0,
        )

        self.assertEqual(reps["tokens"].shape[0], 8)
        self.assertEqual(reps["patch_embeddings"].shape[0], 8)

    def test_multilayer_token_representations_strip_prefix_tokens(self):
        reps, _ = collect_representations(
            model=DummyDinoMultiLayerProbeModel(),
            loader=self._make_loader(),
            device=torch.device("cpu"),
            dataset="FAKEDATA",
            patch_size=2,
            pixel_patch_stride=None,
            max_tokens=32,
            show_progress=False,
            viz_images=0,
        )

        self.assertEqual(reps["tokens_layer_03"].shape[0], 8)
        self.assertEqual(reps["tokens_layer_last"].shape[0], 8)
        self.assertEqual(reps["patch_embeddings"].shape[0], 8)


if __name__ == "__main__":
    unittest.main()
