import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stratified_geometry import run_fiber_bundle_test


class TestStratifiedGeometry(unittest.TestCase):
    def test_run_fiber_bundle_test_smoke(self):
        torch.manual_seed(0)
        embeddings = torch.randn(12, 4)
        results = run_fiber_bundle_test(embeddings, vol_min=2, vol_max=6, ws=1, alpha=0.01, nstrat=2)
        self.assertEqual(len(results), 12)
        self.assertIn("dimensions", results[0])


if __name__ == "__main__":
    unittest.main()
