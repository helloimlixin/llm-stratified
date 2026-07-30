import unittest

import torch
import numpy as np

from scripts.var_shell_experiment import analyze_codebook, compare_groups, compare_groups_by_image


class VarShellExperimentTests(unittest.TestCase):
    def test_codebook_analysis_returns_finite_scores(self):
        generator = torch.Generator().manual_seed(17)
        codebook = torch.randn(256, 12, generator=generator)
        summary, arrays = analyze_codebook(
            codebook,
            neighbors=32,
            alpha=0.05,
            device=torch.device("cpu"),
        )
        self.assertEqual(summary["num_codes"], 256)
        self.assertEqual(summary["tested_inner_radii"], 31)
        self.assertEqual(arrays["scores"].shape, (256,))
        self.assertTrue(np.isfinite(arrays["scores"]).all())

    def test_group_comparison_respects_direction(self):
        values = np.asarray([3.0, 4.0, 0.0, 1.0], dtype=np.float64)
        selected = np.asarray([True, True, False, False])
        result = compare_groups(values, selected, alternative="higher", reps=199, seed=3)
        self.assertAlmostEqual(result["selected_minus_rest"], 3.0)
        self.assertGreaterEqual(result["permutation_p"], 0.0)
        self.assertLessEqual(result["permutation_p"], 1.0)

    def test_image_cluster_comparison_uses_images_as_units(self):
        values = np.asarray([[3.0, 4.0, 0.0, 1.0], [6.0, 5.0, 2.0, 1.0]])
        selected = np.asarray([[True, True, False, False], [True, True, False, False]])
        result = compare_groups_by_image(
            values.reshape(-1),
            selected.reshape(-1),
            num_images=2,
            tokens_per_image=4,
            alternative="higher",
        )
        self.assertEqual(result["num_images"], 2)
        self.assertEqual(result["wins_expected_direction"], 2)
        self.assertAlmostEqual(result["mean_image_difference"], 3.5)
        self.assertGreaterEqual(result["exact_sign_flip_p"], 0.0)
        self.assertLessEqual(result["exact_sign_flip_p"], 1.0)


if __name__ == "__main__":
    unittest.main()
