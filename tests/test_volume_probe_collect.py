import sys
import types
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

datasets_stub = types.ModuleType("datasets")
datasets_stub.create_data_loaders = lambda *args, **kwargs: None
sys.modules.setdefault("datasets", datasets_stub)

fiber_stub = types.ModuleType("fiber")
fiber_geometry_stub = types.ModuleType("fiber.geometry")
fiber_geometry_stub.normalize_volume_range = lambda npts, vol_min, vol_max: (vol_min, vol_max)
fiber_geometry_stub.run_fiber_bundle_test_from_sorted_dists = lambda *args, **kwargs: []
fiber_geometry_stub.sorted_distance_matrix = lambda coords: np.zeros((len(coords), len(coords)), dtype=np.float64)
fiber_geometry_stub.summarize_stratifications = lambda results, alpha=0.0: {}
fiber_stub.geometry = fiber_geometry_stub
sys.modules.setdefault("fiber", fiber_stub)
sys.modules.setdefault("fiber.geometry", fiber_geometry_stub)

models_stub = types.ModuleType("models")
models_stub.DinoV2Wrapper = object
models_stub.TinyViT = object
models_stub.TimmViTWrapper = object
models_stub.resolve_patch_size = lambda model: None
sys.modules.setdefault("models", models_stub)

utils_stub = types.ModuleType("utils")
utils_stub.denormalize_images = lambda imgs, dataset: imgs
utils_stub.seed_everything = lambda seed: None
utils_stub.to_serializable = lambda value: value
sys.modules.setdefault("utils", utils_stub)

from volume_probe import _result_min_pvalues, _select_visual_anchor_indices, collect_representations


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

    def test_result_min_pvalues_ignores_missing_and_nonfinite_values(self):
        values = _result_min_pvalues(
            [
                {"pvalues": [0.4, 0.2]},
                {"pvalues": [float("nan"), 0.1, 0.3]},
                {"pvalues": []},
                {},
            ]
        )

        self.assertEqual(values.shape[0], 4)
        self.assertAlmostEqual(values[0], 0.2)
        self.assertAlmostEqual(values[1], 0.1)
        self.assertTrue(torch.isnan(torch.tensor(values[2])))
        self.assertTrue(torch.isnan(torch.tensor(values[3])))

    def test_select_visual_anchor_indices_covers_high_middle_and_low_scores(self):
        picks = _select_visual_anchor_indices(np.array([0.1, 0.3, 1.2, 0.6, np.nan]), limit=3)

        self.assertEqual(int(picks[0]), 2)
        self.assertIn(3, picks.tolist())
        self.assertTrue(any(idx in {0, 1} for idx in picks.tolist()))


if __name__ == "__main__":
    unittest.main()
