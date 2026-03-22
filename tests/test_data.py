import sys
import json
import tempfile
from pathlib import Path
import unittest

import torch
from torch.utils.data import Dataset, Subset
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import COCO2017MultiLabel, WithIndex, build_dataset, make_loaders, resolve_class_names


class DummyDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return torch.tensor(idx), torch.tensor(idx % 2)


class TestDataHelpers(unittest.TestCase):
    def test_with_index_nested_subset(self):
        base = DummyDataset()
        subset = Subset(base, [2, 5, 7])
        nested = Subset(subset, [1, 2])  # base indices 5, 7
        wrapped = WithIndex(nested)
        _, _, gidx0 = wrapped[0]
        _, _, gidx1 = wrapped[1]
        self.assertEqual(gidx0, 5)
        self.assertEqual(gidx1, 7)

    def test_resolve_class_names_from_dataset(self):
        class Dummy:
            classes = ["a", "b"]

        self.assertEqual(resolve_class_names(Dummy(), "CUSTOM"), ["a", "b"])

    def test_fake_dataset_build(self):
        train_ds, test_ds, num_classes, in_chans, img_size, task = build_dataset("FAKEDATA", root=".", img_size=8)
        self.assertEqual(num_classes, 10)
        self.assertEqual(in_chans, 3)
        self.assertEqual(img_size, 8)
        self.assertEqual(task, "multiclass")
        self.assertGreater(len(train_ds), 0)
        self.assertGreater(len(test_ds), 0)

    def test_make_loaders_keeps_small_train_subset(self):
        train_loader, _test_loader, _num_classes, _in_chans, _img_size, _task = make_loaders(
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

            dataset = COCO2017MultiLabel(img_dir=img_dir, ann_file=ann_path)
            img, target, resolved_idx = dataset[0]
            self.assertEqual(img.size, (8, 8))
            self.assertEqual(resolved_idx, 1)
            self.assertEqual(target.shape[0], 1)
            self.assertEqual(float(target[0].item()), 1.0)

            wrapped = WithIndex(dataset)
            _img_w, _target_w, wrapped_idx = wrapped[0]
            self.assertEqual(wrapped_idx, 1)


if __name__ == "__main__":
    unittest.main()
