import sys
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pretrained_vq_ar_ks_probe import (  # noqa: E402
    embedding_ball_radial_uniformity_ks,
    embedding_radius_uniformity_threshold,
    estimate_stratifications_paper_original,
    finite_minimum,
    build_codebook_control_code_masks,
    codebook_position_masks_from_tokens,
    holm_bonferroni,
    knn_target_entropy,
    load_class_labels_file,
    load_codebook_singular_code_masks,
    local_ball_uniformity_ks,
    one_sample_uniform_ks,
    paper_original_hypothesis_tests,
    ranked_probability_uniform_ks,
    select_singular_mask,
    top_fraction_mask,
    topk_branch_uniform_ks,
)


class TestPretrainedVqArKsProbe(unittest.TestCase):
    def test_holm_bonferroni_step_down_stops_after_first_failure(self):
        pvalues = np.asarray([0.001, 0.030, 0.002, 0.20])

        adjusted, rejected = holm_bonferroni(pvalues, alpha=0.05)

        self.assertEqual(rejected.tolist(), [True, False, True, False])
        self.assertTrue(np.all(adjusted[np.isfinite(adjusted)] >= 0.0))
        self.assertTrue(np.all(adjusted[np.isfinite(adjusted)] <= 1.0))

    def test_finite_minimum_ignores_nan_families(self):
        left = np.asarray([np.nan, 0.4, 0.2])
        right = np.asarray([0.1, np.nan, 0.3])

        result = finite_minimum([left, right])

        self.assertTrue(np.allclose(result, np.asarray([0.1, 0.4, 0.2]), equal_nan=True))

    def test_ranked_probability_ks_is_zero_for_uniform(self):
        probs = np.full((2, 5), 0.2)

        ks = ranked_probability_uniform_ks(probs)

        self.assertTrue(np.allclose(ks, 0.0))

    def test_one_sample_uniform_ks_detects_boundary_concentration(self):
        spread = np.asarray([0.2, 0.4, 0.6, 0.8])
        concentrated = np.asarray([0.01, 0.02, 0.03, 0.04])

        self.assertLess(one_sample_uniform_ks(spread), one_sample_uniform_ks(concentrated))

    def test_embedding_ball_radial_uniformity_returns_fixed_volume_stats(self):
        rng = np.random.default_rng(123)
        features = rng.normal(size=(40, 3))

        stats = embedding_ball_radial_uniformity_ks(
            features=features,
            volume=9,
            exclude_self=True,
            dimension_trim=0.10,
            min_inner=4,
        )

        self.assertEqual(stats["ks"].shape, (40,))
        self.assertEqual(stats["dimension"].shape, (40,))
        self.assertTrue(np.all(np.isfinite(stats["ks"])))
        self.assertTrue(np.all(np.isfinite(stats["dimension"])))
        self.assertTrue(np.all(np.isfinite(stats["radius"])))
        self.assertTrue(np.all(stats["inner_count"] <= 8))
        self.assertTrue(np.all(stats["inner_count"] >= 4))

    def test_embedding_radius_uniformity_threshold_sweeps_candidate_radii(self):
        rng = np.random.default_rng(456)
        features = rng.normal(size=(36, 2))

        stats = embedding_radius_uniformity_threshold(
            features=features,
            volume_min=6,
            volume_max=12,
            volume_step=3,
            uniform_pvalue=0.0,
            max_ks=1.0,
            consecutive=1,
            min_inner=4,
        )

        self.assertEqual(stats["ks_by_volume"].shape, (36, 3))
        self.assertEqual(stats["candidate_volumes"].tolist(), [6.0, 9.0, 12.0])
        self.assertTrue(np.all(stats["threshold_found"]))
        self.assertTrue(np.all(np.isfinite(stats["threshold_radius"])))
        self.assertTrue(np.all(np.isfinite(stats["best_ks"])))

    def test_topk_branch_ks_detects_peaked_distribution(self):
        uniform = np.full((1, 8), 1.0 / 8.0)
        peaked = np.asarray([[0.9, 0.05, 0.03, 0.02, 0.0, 0.0, 0.0, 0.0]])

        ks_uniform, entropy_uniform, _mass_uniform, _codes_uniform = topk_branch_uniform_ks(uniform, top_k=4)
        ks_peaked, entropy_peaked, _mass_peaked, _codes_peaked = topk_branch_uniform_ks(peaked, top_k=4)

        self.assertLess(float(ks_uniform[0]), float(ks_peaked[0]))
        self.assertGreater(float(entropy_uniform[0]), float(entropy_peaked[0]))

    def test_local_ball_uniformity_ks_uses_fixed_volume_neighbors(self):
        features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        targets = np.asarray([0, 1, 2, 3])
        uniform = np.full((4, 4), 0.25)
        peaked = np.asarray(
            [
                [0.90, 0.04, 0.03, 0.03],
                [0.04, 0.90, 0.03, 0.03],
                [0.03, 0.03, 0.90, 0.04],
                [0.03, 0.03, 0.04, 0.90],
            ]
        )

        uniform_stats = local_ball_uniformity_ks(
            features=features,
            probs=uniform,
            target_codes=targets,
            volume=3,
            exclude_self=False,
        )
        peaked_stats = local_ball_uniformity_ks(
            features=features,
            probs=peaked,
            target_codes=targets,
            volume=3,
            exclude_self=False,
        )

        self.assertTrue(np.allclose(uniform_stats["ks"], 0.0))
        self.assertGreater(float(peaked_stats["ks"][0]), float(uniform_stats["ks"][0]))
        self.assertLess(float(peaked_stats["entropy"][0]), float(uniform_stats["entropy"][0]))

    def test_select_singular_mask_prefers_rejected_when_available(self):
        pvalues = np.asarray([0.1, 1e-4, 0.2, 1e-5])
        irregularity = -np.log10(np.maximum(pvalues, 1e-300))

        mask, source = select_singular_mask(
            pvalues=pvalues,
            irregularity=irregularity,
            alpha=1e-2,
            fraction=0.25,
            min_count=2,
        )

        self.assertEqual(source, "fiber_violation_pvalue")
        self.assertEqual(mask.tolist(), [False, True, False, True])

    def test_select_singular_mask_falls_back_to_top_irregularity(self):
        pvalues = np.asarray([0.1, 0.2, 0.3, 0.4])
        irregularity = np.asarray([1.0, 5.0, 2.0, 4.0])

        mask, source = select_singular_mask(
            pvalues=pvalues,
            irregularity=irregularity,
            alpha=1e-2,
            fraction=0.25,
            min_count=2,
        )

        self.assertEqual(source, "top_irregularity_fallback")
        self.assertEqual(mask.tolist(), [False, True, False, True])

    def test_top_fraction_mask_can_balance_by_group(self):
        scores = np.asarray([1.0, 4.0, 2.0, 3.0, 10.0, 9.0])
        groups = np.asarray([0, 0, 0, 1, 1, 1])

        mask = top_fraction_mask(scores, fraction=1 / 3, min_count=1, groups=groups)

        self.assertEqual(mask.tolist(), [False, True, False, False, True, False])

    def test_knn_target_entropy_is_higher_for_mixed_neighbors(self):
        features = np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
                [5.0, 5.0],
                [5.1, 5.0],
                [5.0, 5.1],
            ]
        )
        labels_mixed = np.asarray([1, 1, 2, 7, 7, 7])
        labels_pure = np.asarray([1, 1, 1, 7, 7, 7])

        mixed = knn_target_entropy(features, labels_mixed, k=2)
        pure = knn_target_entropy(features, labels_pure, k=2)

        self.assertGreater(float(mixed[0]), float(pure[0]))

    def test_codebook_singular_masks_map_target_and_previous_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "singular_codes.json"
            path.write_text(json.dumps({"singular_any": [2, 5], "config": {"ignored": True}}), encoding="utf-8")

            code_masks = load_codebook_singular_code_masks(path, vocab_size=8)
            position_masks = codebook_position_masks_from_tokens(
                code_masks,
                np.asarray([[1, 2, 3], [5, 6, 2]], dtype=np.int64),
            )

        self.assertEqual(sorted(code_masks), ["singular_any"])
        self.assertEqual(code_masks["singular_any"].tolist(), [False, False, True, False, False, True, False, False])
        self.assertEqual(
            position_masks["codebook_target_singular_any"].tolist(),
            [False, True, False, True, False, True],
        )
        self.assertEqual(
            position_masks["codebook_prev_singular_any"].tolist(),
            [False, False, True, False, True, False],
        )

    def test_codebook_control_masks_match_reference_size_and_frequency(self):
        code_masks = {
            "large_fiber": np.asarray([False, True, False, True, False, False, False, False], dtype=bool)
        }
        tokens = np.asarray([[1, 1, 3, 4], [3, 5, 6, 7]], dtype=np.int64)

        controls = build_codebook_control_code_masks(
            code_masks,
            tokens,
            source="large_fiber",
            random_controls=1,
            frequency_controls=1,
            seed=123,
        )

        self.assertEqual(sorted(controls), ["freqmatched_large_fiber_00", "random_large_fiber_00"])
        for mask in controls.values():
            self.assertEqual(int(mask.sum()), 2)
            self.assertFalse(bool(np.any(mask & code_masks["large_fiber"])))
        ref_counts = np.bincount(tokens.reshape(-1), minlength=8)[code_masks["large_fiber"]]
        matched_counts = np.bincount(tokens.reshape(-1), minlength=8)[controls["freqmatched_large_fiber_00"]]
        self.assertLessEqual(abs(float(matched_counts.mean()) - float(ref_counts.mean())), 1.0)

    def test_load_class_labels_file_reads_records_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.json"
            records.write_text(
                json.dumps(
                    [
                        {"relative_path": "a.JPEG", "class_label": 7},
                        {"relative_path": "b.JPEG", "class_label": 11},
                        {"relative_path": "c.JPEG", "class_label": 13},
                    ]
                ),
                encoding="utf-8",
            )
            csv_path = root / "labels.csv"
            csv_path.write_text("path,label\na.JPEG,7\nb.JPEG,11\nc.JPEG,13\n", encoding="utf-8")

            self.assertEqual(load_class_labels_file(records, samples=2), [7, 11])
            self.assertEqual(load_class_labels_file(csv_path, samples=3), [7, 11, 13])

    def test_paper_original_estimator_keeps_final_short_segment(self):
        dists = np.linspace(0.0, 1.0, 12)

        result = estimate_stratifications_paper_original(
            dists,
            vol_min=2,
            vol_max=8,
            npts=12,
            ws=10,
            alpha=1e-3,
            nstrat=3,
        )

        self.assertEqual(len(result["dimensions"]), 1)
        self.assertEqual(result["pvalues"], [1.0])

    def test_paper_fiber_test_is_one_sided_for_slope_increase(self):
        volumes = np.arange(10, 90, dtype=np.float64)
        dims_increase = np.where(np.arange(volumes.size) < 40, 1.0, 4.0)
        dims_decrease = np.where(np.arange(volumes.size) < 40, 4.0, 1.0)
        radii_increase = volumes ** (1.0 / dims_increase)
        radii_decrease = volumes ** (1.0 / dims_decrease)

        inc = paper_original_hypothesis_tests(radii_increase, volumes, ws=5, alpha=0.05)
        dec = paper_original_hypothesis_tests(radii_decrease, volumes, ws=5, alpha=0.05)

        self.assertLess(float(inc["fiber_pvalue"]), 0.05)
        self.assertGreater(float(inc["fiber_delta"]), 0.0)
        self.assertEqual(float(dec["fiber_pvalue"]), 1.0)


if __name__ == "__main__":
    unittest.main()
