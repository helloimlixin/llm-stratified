import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "analyze_pixel_sweep.py"
SPEC = importlib.util.spec_from_file_location("analyze_pixel_sweep", SCRIPT_PATH)
analyze_pixel_sweep = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analyze_pixel_sweep)


def _write_summary(run_dir: Path, payload: dict) -> None:
    target = run_dir / "volume_probe"
    target.mkdir(parents=True, exist_ok=True)
    (target / "volume_summary.json").write_text(json.dumps(payload))


class TestAnalyzePixelSweep(unittest.TestCase):
    def test_collect_rows_and_report_cover_pixel_and_pretrained_runs(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pixel_dir = tmp / "pixel"
            pretrained_dir = tmp / "pretrained"

            _write_summary(
                pixel_dir / "cifar10_ps8_stride4",
                {
                    "config": {
                        "dataset": "CIFAR10",
                        "patch_size": 8,
                        "pixel_patch_stride": 4,
                        "feature_backbone": "model",
                    },
                    "representations": {
                        "patch_pixels": {"summary": {"mean_dim": 6.0, "irregular_ratio": 0.20, "num_tokens": 128}},
                        "patch_pixels_stride_4": {"summary": {"mean_dim": 7.0, "irregular_ratio": 0.35, "num_tokens": 128}},
                        "patch_embeddings": {"summary": {"mean_dim": 5.5, "irregular_ratio": 0.10, "num_tokens": 128}},
                        "tokens": {"summary": {"mean_dim": 4.5, "irregular_ratio": 0.15, "num_tokens": 128}},
                    },
                },
            )
            _write_summary(
                pretrained_dir / "cifar10_dinov2_multilayer_full",
                {
                    "config": {
                        "dataset": "CIFAR10",
                        "patch_size": 14,
                        "pixel_patch_stride": None,
                        "feature_backbone": "dinov2",
                        "dinov2_layers": [3, 6, -1],
                    },
                    "representations": {
                        "patch_pixels": {"summary": {"mean_dim": 8.0, "irregular_ratio": 0.40, "num_tokens": 64}},
                        "patch_embeddings": {"summary": {"mean_dim": 6.5, "irregular_ratio": 0.18, "num_tokens": 64}},
                        "tokens_layer_03": {"summary": {"mean_dim": 5.0, "irregular_ratio": 0.22, "num_tokens": 64}},
                        "tokens_layer_06": {"summary": {"mean_dim": 4.6, "irregular_ratio": 0.17, "num_tokens": 64}},
                        "tokens_layer_last": {"summary": {"mean_dim": 4.1, "irregular_ratio": 0.11, "num_tokens": 64}},
                    },
                },
            )
            _write_summary(
                pretrained_dir / "cifar10_untrained_ps8",
                {
                    "config": {
                        "dataset": "CIFAR10",
                        "patch_size": 8,
                        "pixel_patch_stride": None,
                        "feature_backbone": "model",
                    },
                    "representations": {
                        "patch_pixels": {"summary": {"mean_dim": 6.8, "irregular_ratio": 0.33, "num_tokens": 128}},
                        "patch_embeddings": {"summary": {"mean_dim": 5.2, "irregular_ratio": 0.14, "num_tokens": 128}},
                        "tokens": {"summary": {"mean_dim": 4.9, "irregular_ratio": 0.27, "num_tokens": 128}},
                    },
                },
            )

            pixel_rows, pixel_missing = analyze_pixel_sweep._collect_rows(pixel_dir)
            pretrained_rows, pretrained_missing = analyze_pixel_sweep._collect_rows(pretrained_dir)
            rows = pixel_rows + pretrained_rows

            self.assertEqual(len(rows), 3)
            self.assertEqual(pixel_missing, [])
            self.assertEqual(pretrained_missing, [])

            pixel_row = next(row for row in rows if row["variant"] == "pixel_tinyvit")
            self.assertEqual(analyze_pixel_sweep._primary_pixel_rep_name(pixel_row), "patch_pixels_stride_4")

            dino_row = next(row for row in rows if row["variant"] == "dinov2_multilayer")
            self.assertEqual(analyze_pixel_sweep._primary_token_rep_name(dino_row), "tokens_layer_last")

            report = analyze_pixel_sweep._build_markdown_report(
                rows=rows,
                missing=[],
                sweep_dirs=[pixel_dir, pretrained_dir],
                title="Synthetic Sweep",
                output_json=tmp / "combined.json",
            )

            self.assertIn("# Synthetic Sweep", report)
            self.assertIn("## Pixel Sweep", report)
            self.assertIn("## Pretrained Sweep", report)
            self.assertIn("dinov2_multilayer", report)
            self.assertIn("7.00 / 0.350", report)


if __name__ == "__main__":
    unittest.main()
