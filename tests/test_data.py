import sys
from pathlib import Path
import unittest

import torch
from torch.utils.data import Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import WithIndex, build_dataset, resolve_class_names


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


if __name__ == "__main__":
    unittest.main()
