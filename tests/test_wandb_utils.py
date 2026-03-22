import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.wandb_utils import ensure_wandb_dir, resolve_wandb_name


class _BrokenWandbConfig:
    @property
    def name(self):
        raise RuntimeError("Interpolation key 'hydra.job.num' not found")


class TestWandbUtils(unittest.TestCase):
    def test_ensure_wandb_dir_defaults_under_output_dir(self):
        with TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("WANDB_DIR", None)
                wandb_dir = ensure_wandb_dir(enabled=True, output_dir=tmpdir)

            self.assertEqual(wandb_dir, str(Path(tmpdir) / "wandb"))
            self.assertTrue((Path(tmpdir) / "wandb").exists())

    def test_ensure_wandb_dir_preserves_existing_env(self):
        with TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"WANDB_DIR": "/tmp/custom-wandb-dir"}, clear=False):
                wandb_dir = ensure_wandb_dir(enabled=True, output_dir=tmpdir)

            self.assertEqual(wandb_dir, "/tmp/custom-wandb-dir")
            self.assertFalse((Path(tmpdir) / "wandb").exists())

    def test_resolve_wandb_name_uses_explicit_name(self):
        cfg = SimpleNamespace(
            data=SimpleNamespace(name="FLOWERS102"),
            model=SimpleNamespace(name="tinyvit"),
            wandb=SimpleNamespace(name="explicit-name"),
        )

        self.assertEqual(resolve_wandb_name(cfg), "explicit-name")

    def test_resolve_wandb_name_falls_back_when_name_resolution_breaks(self):
        cfg = SimpleNamespace(
            data=SimpleNamespace(name="FLOWERS102"),
            model=SimpleNamespace(name="tinyvit"),
            wandb=_BrokenWandbConfig(),
        )

        self.assertEqual(resolve_wandb_name(cfg, suffix="volume_probe"), "FLOWERS102_tinyvit_volume_probe")


if __name__ == "__main__":
    unittest.main()
