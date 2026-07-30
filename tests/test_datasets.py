import sys
import json
import os
import subprocess
import tempfile
from pathlib import Path
import unittest

import torch
from torch.utils.data import Dataset, Subset
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import (
    Coco2017MultilabelDataset,
    IndexedDataset,
    create_data_loaders,
    create_dataset_pair,
    get_class_names,
)


class DummyDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return torch.tensor(idx), torch.tensor(idx % 2)


class TestDatasetHelpers(unittest.TestCase):
    def test_training_import_order_loads_dataset_stack_in_fresh_process(self):
        env = os.environ.copy()
        src_path = str(ROOT / "src")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [src_path, env.get("PYTHONPATH", "")]))
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import torch; "
                    "from training.loops import evaluate; "
                    "import hydra; "
                    "from omegaconf import OmegaConf; "
                    "import datasets"
                ),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_indexed_dataset_nested_subset(self):
        base = DummyDataset()
        subset = Subset(base, [2, 5, 7])
        nested = Subset(subset, [1, 2])  # base indices 5, 7
        wrapped = IndexedDataset(nested)
        _, _, gidx0 = wrapped[0]
        _, _, gidx1 = wrapped[1]
        self.assertEqual(gidx0, 5)
        self.assertEqual(gidx1, 7)

    def test_get_class_names_from_dataset(self):
        class Dummy:
            classes = ["a", "b"]

        self.assertEqual(get_class_names(Dummy(), "CUSTOM"), ["a", "b"])

    def test_create_dataset_pair_for_fake_data(self):
        train_ds, test_ds, num_classes, in_chans, img_size, task = create_dataset_pair("FAKEDATA", root=".", img_size=8)
        self.assertEqual(num_classes, 10)
        self.assertEqual(in_chans, 3)
        self.assertEqual(img_size, 8)
        self.assertEqual(task, "multiclass")
        self.assertGreater(len(train_ds), 0)
        self.assertGreater(len(test_ds), 0)

    def test_create_data_loaders_keeps_small_train_subset(self):
        train_loader, _test_loader, _num_classes, _in_chans, _img_size, _task = create_data_loaders(
            "FAKEDATA",
            root=".",
            img_size=8,
            batch_size_train=64,
            batch_size_test=16,
            num_workers=0,
            subset_train=32,
            subset_test=16,
            device=torch.device("cpu"),
            distributed=False,
            rank=0,
            world_size=1,
        )
        self.assertEqual(len(train_loader), 1)

    def test_coco_dataset_skips_unreadable_image_and_reports_resolved_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            img_dir = root / "train2017"
            img_dir.mkdir()

            bad_path = img_dir / "bad.jpg"
            bad_path.write_bytes(b"not-a-real-image")

            good_path = img_dir / "good.jpg"
            Image.new("RGB", (8, 8), color=(255, 0, 0)).save(good_path)

            ann_path = root / "instances_train2017.json"
            ann_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {"id": 1, "file_name": "bad.jpg", "width": 8, "height": 8},
                            {"id": 2, "file_name": "good.jpg", "width": 8, "height": 8},
                        ],
                        "categories": [{"id": 7, "name": "cat"}],
                        "annotations": [{"image_id": 2, "category_id": 7, "bbox": [0, 0, 4, 4]}],
                    }
                )
            )

            dataset = Coco2017MultilabelDataset(img_dir=img_dir, ann_file=ann_path)
            img, target, resolved_idx = dataset[0]
            self.assertEqual(img.size, (8, 8))
            self.assertEqual(resolved_idx, 1)
            self.assertEqual(target.shape[0], 1)
            self.assertEqual(float(target[0].item()), 1.0)

            wrapped = IndexedDataset(dataset)
            _img_w, _target_w, wrapped_idx = wrapped[0]
            self.assertEqual(wrapped_idx, 1)


if __name__ == "__main__":
    unittest.main()
