import unittest

import torch

from scripts.var_polysemy_nn_gallery import (
    _nearest_neighbors,
    _select_singular_anchors,
    build_neighbor_records,
)


class VarPolysemyNearestNeighborGalleryTests(unittest.TestCase):
    def test_select_singular_anchors_ignores_controls_and_sorts(self):
        data = {
            "anchors": [
                {
                    "anchor": {
                        "group": "control",
                        "token_index": 0,
                        "irregularity": 100.0,
                        "entropy_norm": 1.0,
                    }
                },
                {
                    "anchor": {
                        "group": "singular",
                        "token_index": 1,
                        "irregularity": 2.0,
                        "entropy_norm": 0.5,
                    },
                    "mean_pairwise_crop_mse": 0.0,
                },
                {
                    "anchor": {
                        "group": "singular",
                        "token_index": 2,
                        "irregularity": 3.0,
                        "entropy_norm": 0.8,
                    },
                    "mean_pairwise_crop_mse": 0.0,
                },
            ]
        }

        selected = _select_singular_anchors(data, limit=2)

        self.assertEqual([item["anchor"]["token_index"] for item in selected], [2, 1])

    def test_nearest_neighbors_can_filter_same_image_tokens(self):
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.99, 0.02],
                [0.20, 0.98],
                [0.98, 0.03],
            ],
            dtype=torch.float32,
        )
        norm = torch.nn.functional.normalize(embeddings, dim=1)

        neighbors = _nearest_neighbors(
            norm,
            token_index=0,
            k=2,
            patches_per_image=2,
            cross_image_only=True,
        )

        self.assertEqual([idx for idx, _score in neighbors], [3, 2])

    def test_build_neighbor_records_preserves_branch_metadata(self):
        branch_data = {
            "anchors": [
                {
                    "anchor": {
                        "group": "singular",
                        "token_index": 0,
                        "image_id": 0,
                        "patch_id": 0,
                        "row": 0,
                        "col": 0,
                        "irregularity": 1.0,
                        "entropy_norm": 1.0,
                    },
                    "target_code": 11,
                    "branch_codes": [11, 12],
                    "branch_probs": [0.5, 0.4],
                    "branch_prob_entropy": 0.9,
                    "mean_pairwise_crop_mse": 0.01,
                }
            ]
        }
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.95, 0.05],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        records = build_neighbor_records(
            branch_data=branch_data,
            embeddings=embeddings,
            limit=1,
            neighbors=1,
            grid_size=2,
            cross_image_only=False,
        )

        self.assertEqual(records[0]["target_code"], 11)
        self.assertEqual(records[0]["branch_codes"], [11, 12])
        self.assertEqual(records[0]["neighbors"][0]["token_index"], 1)


if __name__ == "__main__":
    unittest.main()
