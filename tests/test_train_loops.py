import sys
from pathlib import Path
import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train import evaluate, multilabel_accuracy, train_one_epoch


class SimpleModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, num_classes))

    def forward(self, x):
        return self.net(x)


class TestTrainLoops(unittest.TestCase):
    def test_train_one_epoch_runs(self):
        torch.manual_seed(0)
        x = torch.rand(8, 3, 4, 4)
        y = torch.randint(0, 3, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
        model = SimpleModel(num_classes=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss, acc = train_one_epoch(
            model,
            loader,
            optimizer,
            scaler=None,
            device=torch.device("cpu"),
            task_type="multiclass",
            grad_clip=None,
            label_smoothing=0.0,
            epoch=0,
            sampler=None,
        )
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_evaluate_runs(self):
        torch.manual_seed(0)
        x = torch.rand(6, 3, 4, 4)
        y = torch.randint(0, 2, (6,))
        loader = DataLoader(TensorDataset(x, y), batch_size=3, shuffle=False)
        model = SimpleModel(num_classes=2)
        loss, acc = evaluate(model, loader, torch.device("cpu"), task_type="multiclass", label_smoothing=0.0)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)

    def test_multilabel_accuracy(self):
        logits = torch.tensor([[1.0, -1.0], [-1.0, 2.0]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        acc = multilabel_accuracy(logits, targets)
        self.assertAlmostEqual(acc, 1.0)


if __name__ == "__main__":
    unittest.main()
