import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.volume_probe_logging import (
    build_volume_probe_curve_rows,
    build_volume_probe_log_payload,
    collect_volume_probe_artifact_paths,
    resolve_volume_probe_tags,
)


class DummyWandb:
    @staticmethod
    def Image(path: str):
        return {"image": path}


class TestVolumeProbeLogging(unittest.TestCase):
    def test_resolve_volume_probe_tags_appends_once(self):
        tags = resolve_volume_probe_tags(["baseline", "volume-probe"])
        self.assertEqual(tags.count("volume-probe"), 1)

    def test_build_curve_rows_prefers_tokens_k_axis(self):
        results = {
            "representations": {
                "tokens": {"knn_curve": {"k_values": [1, 2, 4], "radii": {"q50": [0.1, 0.2, 0.4]}}},
                "patch_embeddings": {"knn_curve": {"k_values": [2, 4, 8], "radii": {"q50": [0.3, 0.6, 1.2]}}},
            }
        }

        rows = build_volume_probe_curve_rows(results)

        self.assertEqual([step for step, _row in rows], [1, 2, 4])
        self.assertIn("volume_probe/tokens/radius_q50", rows[0][1])
        self.assertNotIn("volume_probe/patch_embeddings/radius_q50", rows[0][1])
        self.assertEqual(rows[1][1]["volume_probe/patch_embeddings/radius_q50"], 0.3)

    def test_build_log_payload_and_artifacts_use_existing_files_only(self):
        results = {
            "representations": {
                "tokens": {
                    "summary": {"num_tokens": 16},
                    "knn_curve": {"k_values": [4, 8], "radii": {"q50": [0.5, 1.0]}},
                    "dims_path": "tokens_dims.npy",
                    "results_path": "tokens_results.json",
                }
            },
            "viz": {"anchors": "anchors.png", "missing": "missing.png"},
        }

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for filename in ("volume_summary.json", "anchors.png", "tokens_dims.npy"):
                (tmp / filename).write_text("x")

            payload = build_volume_probe_log_payload(results, tmp, DummyWandb())
            artifacts = collect_volume_probe_artifact_paths(results, tmp)

        self.assertEqual(payload["volume_probe/tokens/num_points"], 16)
        self.assertEqual(payload["volume_probe/tokens/k_min"], 4)
        self.assertEqual(payload["volume_probe/tokens/k_max"], 8)
        self.assertIn("volume_probe/viz/anchors", payload)
        self.assertNotIn("volume_probe/viz/missing", payload)
        self.assertEqual({path.name for path in artifacts}, {"volume_summary.json", "anchors.png", "tokens_dims.npy"})


if __name__ == "__main__":
    unittest.main()
