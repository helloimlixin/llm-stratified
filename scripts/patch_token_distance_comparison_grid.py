"""Create readable side-by-side patch-token distance comparison grids."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fiber.figure_io import save_figure
from patch_token_distance_heatmaps import (
    _analyze_image,
    _draw_grid,
    _parse_image_ids,
    _torch_load,
)
from utils import denormalize_images

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
except Exception as exc:  # pragma: no cover
    raise ImportError("matplotlib is required for patch-token distance comparison grids") from exc


METRIC_CONFIG = {
    "rank": {
        "row_key": "rank_agreement_grid",
        "mean_key": "mean_rank_spearman",
        "title": "Distance-Rank Agreement",
        "colorbar": "Spearman correlation",
        "vmin": -0.20,
        "vmax": 0.85,
        "cmap": "viridis",
    },
    "overlap": {
        "row_key": "neighbor_overlap_grid",
        "mean_key": "mean_neighbor_overlap",
        "title": "Top-16 Neighbor Overlap",
        "colorbar": "fraction of raw neighbors recovered",
        "vmin": 0.00,
        "vmax": 0.55,
        "cmap": "viridis",
    },
}


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _parse_run(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError("--run must be provided as Label=path/to/epoch_000.pt")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("Run label cannot be empty")
    return label, _resolve(raw_path.strip())


def _load_run_rows(
    *,
    label: str,
    path: Path,
    dataset: str,
    image_ids: list[int],
    patch_size: int | None,
    neighbor_k: int,
    raw_distance_mode: str,
    token_distance_mode: str,
) -> dict[str, Any]:
    artifact = _torch_load(path)
    embeddings = artifact["embeddings"].detach().float().cpu()
    images = artifact["images"].detach().float().cpu()
    token_image_ids = artifact["image_ids"].detach().cpu().long()
    bboxes = artifact["bboxes"].detach().cpu().long()
    images01 = denormalize_images(images, dataset).cpu()

    rows: dict[int, dict[str, Any]] = {}
    for image_id in image_ids:
        row = _analyze_image(
            embeddings=embeddings,
            images01=images01,
            image_ids=token_image_ids,
            bboxes=bboxes,
            image_id=int(image_id),
            patch_size=patch_size,
            neighbor_k=neighbor_k,
            raw_distance_mode=raw_distance_mode,
            token_distance_mode=token_distance_mode,
        )
        if row is not None:
            rows[int(image_id)] = row
    return {
        "label": label,
        "path": str(path),
        "rows": rows,
        "images01": images01,
    }


def _select_representative_ids(
    *,
    runs: list[dict[str, Any]],
    all_image_ids: list[int],
    mean_key: str,
    max_images: int,
) -> list[int]:
    scores: list[tuple[int, float]] = []
    for image_id in all_image_ids:
        values = [
            float(run["rows"][image_id][mean_key])
            for run in runs
            if image_id in run["rows"] and math.isfinite(float(run["rows"][image_id][mean_key]))
        ]
        if values:
            scores.append((int(image_id), float(np.mean(values))))
    if not scores:
        return all_image_ids[:max_images]
    scores.sort(key=lambda item: item[1])
    count = min(max(1, int(max_images)), len(scores))
    if count == 1:
        return [scores[len(scores) // 2][0]]
    positions = np.linspace(0, len(scores) - 1, count)
    selected: list[int] = []
    for pos in positions:
        image_id = scores[int(round(float(pos)))][0]
        if image_id not in selected:
            selected.append(image_id)
    for image_id, _score in scores:
        if len(selected) >= count:
            break
        if image_id not in selected:
            selected.append(image_id)
    return selected[:count]


def _collect_available_ids(path: Path) -> list[int]:
    artifact = _torch_load(path)
    image_ids = artifact["image_ids"].detach().cpu().long()
    return sorted({int(v) for v in image_ids.tolist()})


def _draw_clean_grid(
    *,
    runs: list[dict[str, Any]],
    image_ids: list[int],
    metric: str,
    out_path: Path,
) -> Path:
    cfg = METRIC_CONFIG[metric]
    num_rows = len(image_ids)
    num_cols = len(runs) + 1
    fig = plt.figure(
        figsize=(3.0 * num_cols + 0.95, 2.75 * num_rows + 1.45),
        constrained_layout=False,
    )
    gs = fig.add_gridspec(
        num_rows,
        num_cols + 1,
        width_ratios=[1.05] + [1.0] * len(runs) + [0.06],
        left=0.045,
        right=0.965,
        top=0.90,
        bottom=0.12,
        wspace=0.08,
        hspace=0.22,
    )
    norm = mcolors.Normalize(vmin=float(cfg["vmin"]), vmax=float(cfg["vmax"]))
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=str(cfg["cmap"]))
    image_source = runs[0]["images01"]

    for row_idx, image_id in enumerate(image_ids):
        row_for_grid = next(
            run["rows"][image_id]
            for run in runs
            if image_id in run["rows"]
        )
        grid_h = int(row_for_grid["grid_h"])
        grid_w = int(row_for_grid["grid_w"])
        image = image_source[int(image_id)].permute(1, 2, 0).numpy()

        ax_img = fig.add_subplot(gs[row_idx, 0])
        ax_img.imshow(np.clip(image, 0.0, 1.0), interpolation="bilinear")
        _draw_grid(ax_img, image_h=image.shape[0], image_w=image.shape[1], grid_h=grid_h, grid_w=grid_w)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_ylabel(f"image {image_id}", fontsize=11, rotation=90, labelpad=12)
        if row_idx == 0:
            ax_img.set_title("source", fontsize=12, pad=8)

        for col_idx, run in enumerate(runs, start=1):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            values = np.asarray(run["rows"][image_id][cfg["row_key"]], dtype=float)
            mean_value = float(run["rows"][image_id][cfg["mean_key"]])
            ax.imshow(values, cmap=str(cfg["cmap"]), norm=norm, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_edgecolor("#555555")
            if row_idx == 0:
                ax.set_title(run["label"], fontsize=12, pad=8)
            ax.text(
                0.03,
                0.06,
                f"{mean_value:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                color="white",
                ha="left",
                va="bottom",
                bbox={"facecolor": "black", "alpha": 0.62, "edgecolor": "none", "pad": 2.4},
            )

    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(str(cfg["colorbar"]), fontsize=11, labelpad=9)
    cbar.ax.tick_params(labelsize=9)
    fig.suptitle(
        f"Patch-Token Distance: {cfg['title']}",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.045,
        0.045,
        "Clean score grids use one shared color scale; numbers inside panels are per-image means.",
        fontsize=9.5,
        color="#333333",
    )
    path = save_figure(fig, out_path, dpi=240)
    plt.close(fig)
    return path


def run(args: argparse.Namespace) -> None:
    parsed_runs = [_parse_run(spec) for spec in args.run]
    if len(parsed_runs) < 2:
        raise ValueError("Provide at least two --run entries")
    available_ids = _collect_available_ids(parsed_runs[0][1])
    requested_ids = _parse_image_ids(args.image_ids, available_ids, max(args.max_images, 16))
    runs = [
        _load_run_rows(
            label=label,
            path=path,
            dataset=args.dataset,
            image_ids=requested_ids,
            patch_size=args.patch_size,
            neighbor_k=args.neighbor_k,
            raw_distance_mode=args.raw_distance_mode,
            token_distance_mode=args.token_distance_mode,
        )
        for label, path in parsed_runs
    ]
    cfg = METRIC_CONFIG[args.metric]
    selected_ids = (
        requested_ids[: args.max_images]
        if args.image_ids
        else _select_representative_ids(
            runs=runs,
            all_image_ids=requested_ids,
            mean_key=str(cfg["mean_key"]),
            max_images=args.max_images,
        )
    )
    path = _draw_clean_grid(runs=runs, image_ids=selected_ids, metric=args.metric, out_path=args.out)
    print(f"[patch_token_distance_comparison_grid] selected image ids: {selected_ids}")
    print(f"[patch_token_distance_comparison_grid] wrote {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run specification as Label=path/to/epoch_000.pt. Repeat once per model.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metric", choices=sorted(METRIC_CONFIG), default="rank")
    parser.add_argument("--dataset", default="coco")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--neighbor-k", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--image-ids", default=None)
    parser.add_argument(
        "--raw-distance-mode",
        choices=["raw_l2", "image_standardized", "patch_centered_cosine"],
        default="raw_l2",
    )
    parser.add_argument(
        "--token-distance-mode",
        choices=["raw_l2", "feature_standardized", "l2_normalized"],
        default="raw_l2",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
