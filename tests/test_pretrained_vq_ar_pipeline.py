import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pretrained_vq_ar_pipeline import (  # noqa: E402
    LLAMAGEN_PROFILES,
    build_compatibility_report,
    parse_class_labels,
    resolve_precision,
)

import torch  # noqa: E402


class TestPretrainedVqArPipeline(unittest.TestCase):
    def test_parse_class_labels_repeats_to_sample_count(self):
        labels = parse_class_labels("1,2", samples=5)

        self.assertEqual(labels, [1, 2, 1, 2, 1])

    def test_compatibility_report_marks_imagegpt_mismatch(self):
        report = build_compatibility_report()
        rows = {row["name"]: row for row in report["pairs"]}

        self.assertTrue(rows["llamagen-c2i"]["compatible"])
        self.assertFalse(rows["vqgan-plus-openai-imagegpt"]["compatible"])
        self.assertIn("pixel", rows["vqgan-plus-openai-imagegpt"]["reason"].lower())

    def test_llamagen_profile_has_matched_vocab_metadata(self):
        profile = LLAMAGEN_PROFILES["c2i-B-256"]

        self.assertEqual(profile["codebook_size"], 16384)
        self.assertEqual(profile["downsample_size"], 16)
        self.assertEqual(profile["image_size"], 256)

    def test_auto_precision_prefers_float32_on_cpu(self):
        self.assertEqual(resolve_precision("auto", torch.device("cpu")), torch.float32)


if __name__ == "__main__":
    unittest.main()
