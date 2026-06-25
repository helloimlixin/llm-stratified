import sys
from pathlib import Path
import unittest

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
sys.path.insert(0, str(ROOT / "src"))


def compose_config(*overrides):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=list(overrides))


class TestHydraConfigs(unittest.TestCase):
    def test_quick_test_experiment_composes(self):
        cfg = compose_config("+experiment=quick_test")
        self.assertEqual(cfg.data.name, "FAKEDATA")
        self.assertEqual(cfg.compute.mode, "single")
        self.assertFalse(cfg.wandb.enabled)

    def test_dataset_aliases_compose(self):
        aliases = [
            "cifar10",
            "cifar100",
            "stl10",
            "food101",
            "flowers102",
            "dtd",
            "eurosat",
            "svhn",
            "fakedata",
            "fake",
            "coco",
            "coco2017",
            "voc",
            "voc2007",
            "voc2012",
            "imagenet",
            "ffhq",
            "celebahq",
            "celeba",
            "clevr",
        ]
        for alias in aliases:
            with self.subTest(alias=alias):
                cfg = compose_config(f"data={alias}")
                self.assertTrue(cfg.data.name)
                self.assertTrue(cfg.data.root)
                self.assertGreater(cfg.data.batch_size, 0)
                self.assertGreater(cfg.data.batch_size_test, 0)

    def test_local_overrides_compose(self):
        cfg = compose_config(
            "+experiment=volume_probe",
            "data.root=../data",
            "data.num_workers=0",
            "wandb.enabled=false",
        )
        self.assertEqual(cfg.data.root, "../data")
        self.assertEqual(cfg.data.num_workers, 0)
        self.assertTrue(cfg.volume_probe.enabled)

    def test_coco_sam_fiber_experiment_composes(self):
        cfg = compose_config(
            "+experiment=coco_sam_fiber",
            "data.root=../data",
            "wandb.enabled=false",
        )
        self.assertEqual(cfg.data.name, "COCO")
        self.assertEqual(cfg.model.name, "sam_base")
        self.assertEqual(cfg.model.frozen_backbone, "sam")
        self.assertEqual(cfg.model.frozen_backbone_model, "facebook/sam-vit-base")
        self.assertTrue(cfg.sam_fiber.enabled)
        self.assertTrue(cfg.sam_fiber.sparse_probe)
        self.assertEqual(cfg.sam_fiber.sparse_probe_neighbor_k, 128)
        self.assertTrue(cfg.sam_fiber.sparse_probe_volume_curve)
        self.assertEqual(cfg.sam_fiber.sparse_probe_volume_curve_volumes[-1], 512)


if __name__ == "__main__":
    unittest.main()
