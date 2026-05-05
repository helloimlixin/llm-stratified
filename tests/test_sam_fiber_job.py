import sys
from pathlib import Path
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.sam_fiber_job import (
    _boxes_to_patch_multihot,
    _grid_bboxes,
    _masks_to_patch_multihot,
    _preview_epoch_summary,
    _preview_rows,
    collect_sam_patch_tokens,
)


class DummySamBackbone(torch.nn.Module):
    embed_dim = 2
    expected_image_size = 4
    patch_size = 2

    def prepare_single_image(self, img, dataset, *, device):
        return {"pixel_values": img.unsqueeze(0).to(device)}, img.detach().cpu().clamp(0, 1)

    def get_image_embedding_map(self, pixel_values):
        base = torch.arange(8, dtype=torch.float32, device=pixel_values.device).reshape(1, 2, 2, 2)
        return base + float(pixel_values.shape[0])

    def predict_masks_for_boxes(self, **kwargs):
        return []


class TestSamFiberHelpers(unittest.TestCase):
    def test_grid_bboxes_cover_expected_patch_cells(self):
        boxes = _grid_bboxes(image_h=8, image_w=8, grid_h=2, grid_w=2)
        self.assertEqual(boxes.shape, (4, 4))
        self.assertEqual(boxes[0].tolist(), [0, 0, 4, 4])
        self.assertEqual(boxes[-1].tolist(), [4, 4, 8, 8])

    def test_boxes_to_patch_multihot_marks_overlapping_cells(self):
        labels = _boxes_to_patch_multihot(
            [(0.0, 0.0, 4.0, 8.0, 1)],
            grid_h=2,
            grid_w=2,
            image_h=8,
            image_w=8,
            num_classes=3,
        )
        self.assertEqual(labels.shape, (4, 3))
        self.assertEqual(labels[:, 1].tolist(), [1.0, 0.0, 1.0, 0.0])

    def test_masks_to_patch_multihot_uses_masks_and_falls_back(self):
        fallback = torch.zeros((4, 2), dtype=torch.float32)
        fallback[:, 1] = 1.0

        mask = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        labels = _masks_to_patch_multihot(
            [mask],
            [0],
            grid_h=2,
            grid_w=2,
            num_classes=2,
            threshold=0.25,
            fallback=fallback,
        )
        self.assertEqual(labels[:, 0].tolist(), [1.0, 0.0, 0.0, 0.0])

        labels_fallback = _masks_to_patch_multihot(
            [],
            [],
            grid_h=2,
            grid_w=2,
            num_classes=2,
            threshold=0.25,
            fallback=fallback,
        )
        self.assertTrue(torch.equal(labels_fallback, fallback))

    def test_preview_rows_emit_richer_descriptions(self):
        preview = {
            "image": torch.zeros((3, 4, 4), dtype=torch.float32),
            "boxes": [(0.0, 0.0, 2.0, 4.0, 0), (2.0, 0.0, 4.0, 4.0, 1)],
            "masks": [
                torch.tensor(
                    [
                        [1.0, 1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                torch.tensor(
                    [
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ),
            ],
            "image_id": 17,
            "categories": [0, 1],
        }

        rows = _preview_rows([preview], class_names=["cat", "dog"], mask_threshold=0.25)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["classes"], "cat x1, dog x1")
        self.assertAlmostEqual(rows[0]["prompt_box_fraction"], 1.0)
        self.assertEqual(rows[0]["coverage_regime"], "dense")
        self.assertIn("Per-class mask coverage", rows[0]["description"])

        summary = _preview_epoch_summary([preview], preview_rows=rows, class_names=["cat", "dog"])
        self.assertEqual(summary["num_preview_images"], 1)
        self.assertIn("Most frequent prompt classes: cat x1, dog x1.", summary["description"])

    def test_collect_sam_patch_tokens_uses_unique_image_buffer(self):
        imgs = torch.rand(2, 3, 4, 4)
        labels = torch.zeros(2, 3, dtype=torch.float32)
        labels[0, 1] = 1.0
        labels[1, 2] = 1.0
        indices = torch.tensor([10, 11], dtype=torch.int64)
        loader = DataLoader(TensorDataset(imgs, labels, indices), batch_size=2, shuffle=False)

        embeddings, patch_labels, images, bboxes, patch_indices, image_ids, pred_labels, previews = collect_sam_patch_tokens(
            model=DummySamBackbone(),
            loader=loader,
            device=torch.device("cpu"),
            dataset="FAKEDATA",
            analysis_patch_size=2,
            num_classes=3,
            max_tokens=6,
            mask_threshold=0.25,
            max_boxes_per_image=0,
            mask_preview_images=0,
            multimask_output=False,
            show_progress=False,
        )

        self.assertEqual(embeddings.shape, (6, 2))
        self.assertEqual(patch_labels.shape, (6, 3))
        self.assertEqual(images.shape[0], 2)
        self.assertEqual(bboxes.shape, (6, 4))
        self.assertEqual(patch_indices.tolist(), [0, 1, 2, 3, 0, 1])
        self.assertEqual(image_ids.tolist(), [0, 0, 0, 0, 1, 1])
        self.assertEqual(pred_labels.tolist(), [1, 1, 1, 1, 2, 2])
        self.assertEqual(previews, [])


if __name__ == "__main__":
    unittest.main()
