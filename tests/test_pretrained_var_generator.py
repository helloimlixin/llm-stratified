import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pretrained_var_generator import parse_class_labels, parse_patch_nums, resolve_model_defaults  # noqa: E402


class TestPretrainedVarGenerator(unittest.TestCase):
    def test_resolve_model_defaults_selects_checkpoint_family(self):
        d16 = resolve_model_defaults(16)
        self.assertEqual(d16["filename"], "var_d16.pth")
        self.assertEqual(d16["resolution"], 256)
        self.assertFalse(d16["shared_aln"])
        self.assertEqual(d16["patch_nums"][-1], 16)

        d36 = resolve_model_defaults(36)
        self.assertEqual(d36["filename"], "var_d36.pth")
        self.assertEqual(d36["resolution"], 512)
        self.assertTrue(d36["shared_aln"])
        self.assertEqual(d36["patch_nums"][-1], 32)

    def test_resolve_model_defaults_rejects_unknown_depth(self):
        with self.assertRaises(ValueError):
            resolve_model_defaults(12)

    def test_parse_patch_nums_supports_var_aliases(self):
        self.assertEqual(parse_patch_nums("auto", resolution=256), (1, 2, 3, 4, 5, 6, 8, 10, 13, 16))
        self.assertEqual(parse_patch_nums("512", resolution=256), (1, 2, 3, 4, 6, 9, 13, 18, 24, 32))
        self.assertEqual(parse_patch_nums("1_2_4", resolution=256), (1, 2, 4))

    def test_parse_class_labels_repeats_to_sample_count(self):
        self.assertEqual(parse_class_labels("980,437", samples=5), [980, 437, 980, 437, 980])
        self.assertIsNone(parse_class_labels("random", samples=3))
        with self.assertRaises(ValueError):
            parse_class_labels("1001", samples=1)


if __name__ == "__main__":
    unittest.main()
