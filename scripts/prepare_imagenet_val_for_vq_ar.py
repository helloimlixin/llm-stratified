"""Prepare ImageNet validation images and labels for VQ-AR probes.

The ImageNet validation tar is flat: ``ILSVRC2012_val_00000001.JPEG`` ...
``ILSVRC2012_val_00050000.JPEG``.  LlamaGen's class-conditional AR model
expects canonical ImageNet-1k class indices, so this script pairs the extracted
images with the validation synset label list and the canonical class-index JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_SYNSET_LABELS_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/slim/datasets/imagenet_2012_validation_synset_labels.txt"
)
DEFAULT_CLASS_INDEX_URL = (
    "https://raw.githubusercontent.com/raghakot/keras-vis/master/"
    "resources/imagenet_class_index.json"
)
EXPECTED_VAL_IMAGES = 50_000


def maybe_download(source: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    if source.startswith(("http://", "https://")):
        print(f"[download] {source} -> {destination}", flush=True)
        urllib.request.urlretrieve(source, destination)
        return destination
    src = Path(source).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    destination.write_bytes(src.read_bytes())
    return destination


def extract_val_tar(val_tar: Path, images_dir: Path, *, skip_extract: bool) -> int:
    images_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(images_dir.glob("ILSVRC2012_val_*.JPEG"))
    if skip_extract or len(existing) >= EXPECTED_VAL_IMAGES:
        return len(existing)
    print(f"[extract] {val_tar} -> {images_dir}", flush=True)
    subprocess.run(["tar", "-xf", str(val_tar), "-C", str(images_dir)], check=True)
    return len(list(images_dir.glob("ILSVRC2012_val_*.JPEG")))


def load_class_index(path: Path) -> dict[str, tuple[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    synset_to_info: dict[str, tuple[int, str]] = {}
    for key, value in payload.items():
        if not isinstance(value, list) or len(value) < 2:
            raise ValueError(f"bad class-index row for {key}: {value!r}")
        label = int(key)
        synset = str(value[0])
        name = str(value[1])
        synset_to_info[synset] = (label, name)
    if len(synset_to_info) != 1000:
        raise ValueError(f"expected 1000 ImageNet classes, got {len(synset_to_info)}")
    return synset_to_info


def write_labels_csv(
    *,
    images_dir: Path,
    synset_labels_path: Path,
    class_index_path: Path,
    out_csv: Path,
) -> list[dict[str, Any]]:
    synsets = [line.strip() for line in synset_labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(synsets) != EXPECTED_VAL_IMAGES:
        raise ValueError(f"expected {EXPECTED_VAL_IMAGES} validation synsets, got {len(synsets)}")
    synset_to_info = load_class_index(class_index_path)

    rows: list[dict[str, Any]] = []
    for idx, synset in enumerate(synsets, start=1):
        if synset not in synset_to_info:
            raise KeyError(f"synset {synset!r} is missing from class-index JSON")
        label, class_name = synset_to_info[synset]
        filename = f"ILSVRC2012_val_{idx:08d}.JPEG"
        rows.append(
            {
                "path": filename,
                "label": int(label),
                "synset": synset,
                "class_name": class_name,
                "exists": (images_dir / filename).exists(),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["path", "label", "synset", "class_name", "exists"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-tar", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--synset-labels", default=DEFAULT_SYNSET_LABELS_URL)
    parser.add_argument("--class-index-json", default=DEFAULT_CLASS_INDEX_URL)
    parser.add_argument("--skip-extract", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    val_tar = Path(args.val_tar).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    images_dir = out_dir / "images"
    meta_dir = out_dir / "metadata"
    labels_path = meta_dir / "imagenet_2012_validation_synset_labels.txt"
    class_index_path = meta_dir / "imagenet_class_index.json"
    labels_csv = out_dir / "imagenet_val_labels.csv"
    summary_path = out_dir / "imagenet_val_setup_summary.json"

    maybe_download(args.synset_labels, labels_path)
    maybe_download(args.class_index_json, class_index_path)
    image_count = extract_val_tar(val_tar, images_dir, skip_extract=bool(args.skip_extract))
    rows = write_labels_csv(
        images_dir=images_dir,
        synset_labels_path=labels_path,
        class_index_path=class_index_path,
        out_csv=labels_csv,
    )
    missing = [row["path"] for row in rows if not row["exists"]]
    summary = {
        "val_tar": str(val_tar),
        "out_dir": str(out_dir),
        "images_dir": str(images_dir),
        "image_count": int(image_count),
        "expected_images": EXPECTED_VAL_IMAGES,
        "labels_csv": str(labels_csv),
        "synset_labels": str(labels_path),
        "class_index_json": str(class_index_path),
        "missing_images": missing[:20],
        "missing_image_count": len(missing),
        "first_rows": rows[:5],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
