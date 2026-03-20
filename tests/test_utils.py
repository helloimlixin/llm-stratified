import sys
from pathlib import Path
import unittest

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import to_serializable


class TestUtils(unittest.TestCase):
    def test_to_serializable_handles_omegaconf(self):
        cfg = OmegaConf.create({"name": "probe", "layers": [3, 6, -1]})
        self.assertEqual(to_serializable(cfg), {"name": "probe", "layers": [3, 6, -1]})


if __name__ == "__main__":
    unittest.main()
