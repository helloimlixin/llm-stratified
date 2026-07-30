import sys
from pathlib import Path
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vq_ar_polysemy_branch_gallery import (  # noqa: E402
    average_pairwise_l2,
    branch_codes,
    choose_anchor_pairs,
    score_records,
)


def record(
    *,
    token_index: int,
    sample_id: int,
    patch_id: int,
    selected: bool,
    local_ball_ks: float,
    local_ball_entropy: float,
    branch_ks: float,
    branch_entropy: float,
) -> dict:
    return {
        "token_index": token_index,
        "sample_id": sample_id,
        "patch_id": patch_id,
        "row": patch_id // 4,
        "col": patch_id % 4,
        "target_code": token_index + 10,
        "top_branch_codes": [1, 2, 3],
        "codebook_target_large_fiber": selected,
        "local_ball_ks": local_ball_ks,
        "local_ball_entropy": local_ball_entropy,
        "branch_ks": branch_ks,
        "branch_entropy": branch_entropy,
    }


class TestVqArPolysemyBranchGallery(unittest.TestCase):
    def test_branch_codes_deduplicates_and_limits(self):
        row = {"top_branch_codes": [7, 7, 3, 9, 3, 11]}

        self.assertEqual(branch_codes(row, branches=3), [7, 3, 9])

    def test_choose_anchor_pairs_matches_regular_control_within_sample(self):
        records = [
            record(
                token_index=0,
                sample_id=0,
                patch_id=6,
                selected=True,
                local_ball_ks=0.20,
                local_ball_entropy=0.95,
                branch_ks=0.05,
                branch_entropy=0.99,
            ),
            record(
                token_index=1,
                sample_id=0,
                patch_id=7,
                selected=False,
                local_ball_ks=0.85,
                local_ball_entropy=0.20,
                branch_ks=0.45,
                branch_entropy=0.80,
            ),
            record(
                token_index=2,
                sample_id=1,
                patch_id=8,
                selected=True,
                local_ball_ks=0.55,
                local_ball_entropy=0.55,
                branch_ks=0.25,
                branch_entropy=0.90,
            ),
            record(
                token_index=3,
                sample_id=1,
                patch_id=9,
                selected=False,
                local_ball_ks=0.65,
                local_ball_entropy=0.35,
                branch_ks=0.35,
                branch_entropy=0.85,
            ),
        ]

        pairs = choose_anchor_pairs(
            records,
            selector="codebook_target_large_fiber",
            pairs=1,
            min_patch_id=0,
            max_patch_id=15,
        )

        self.assertEqual(len(pairs), 1)
        singular, regular = pairs[0]
        self.assertTrue(singular["codebook_target_large_fiber"])
        self.assertFalse(regular["codebook_target_large_fiber"])
        self.assertEqual(singular["sample_id"], regular["sample_id"])
        self.assertEqual(singular["token_index"], 0)
        self.assertEqual(regular["token_index"], 1)

    def test_average_pairwise_l2_uses_marked_patch_crop(self):
        black = Image.new("RGB", (4, 4), (0, 0, 0))
        white = Image.new("RGB", (4, 4), (255, 255, 255))
        row = {"row": 0, "col": 0}

        self.assertAlmostEqual(average_pairwise_l2([black, white], row, grid=2, context=0), 1.0)

    def test_position_score_prefers_earlier_records_without_flatness_metrics(self):
        records = [{"token_index": idx} for idx in range(3)]

        singular_score, regular_score = score_records(records, mode="position")

        self.assertGreater(singular_score[0], singular_score[1])
        self.assertGreater(regular_score[1], regular_score[2])


if __name__ == "__main__":
    unittest.main()
