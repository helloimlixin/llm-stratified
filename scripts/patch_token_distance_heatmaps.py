"""Compare raw-patch distances with corresponding token distances.

Raw RGB patches and model tokens live in different vector spaces, so this
diagnostic compares their *pairwise distance structure* rather than subtracting
patch vectors from token vectors directly. For each source image, we compute:

- a raw-patch distance matrix over the image grid,
- a token-feature distance matrix over the corresponding patch tokens,
- per-patch distance-rank agreement, projected back to the image grid,
- per-patch top-k neighbor overlap, also projected back to the image grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fiber.figure_io import save_figure
from utils import denormalize_images, to_serializable

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
except Exception as exc:  # pragma: no cover
    raise ImportError("matplotlib is required for patch-token distance heatmaps") from exc


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def _center_l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr - np.mean(arr, axis=1, keepdims=True)
    return _l2_normalize_rows(arr)


def _standardize_columns(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return (arr - np.mean(arr, axis=0, keepdims=True)) / np.maximum(
        np.std(arr, axis=0, keepdims=True),
        1e-6,
    )


def _raw_patch_features(x: np.ndarray, *, mode: str) -> np.ndarray:
    """Prepare raw patch vectors before pairwise-distance comparison."""
    arr = np.asarray(x, dtype=np.float64)
    if mode == "raw_l2":
        return arr
    if mode == "image_standardized":
        return _standardize_columns(arr)
    if mode == "patch_centered_cosine":
        return _center_l2_normalize_rows(arr)
    raise ValueError(f"Unknown raw distance mode: {mode}")


def _token_features(x: np.ndarray, *, mode: str) -> np.ndarray:
    """Prepare token vectors before pairwise-distance comparison."""
    arr = np.asarray(x, dtype=np.float64)
    if mode == "raw_l2":
        return arr
    if mode == "feature_standardized":
        return _standardize_columns(arr)
    if mode == "l2_normalized":
        return _l2_normalize_rows(arr)
    raise ValueError(f"Unknown token distance mode: {mode}")


def _parse_image_ids(spec: str | None, available_ids: list[int], max_images: int) -> list[int]:
    if spec is None or not str(spec).strip():
        return available_ids[: max(1, int(max_images))]
    selected: list[int] = []
    for part in str(spec).split(","):
        text = part.strip()
        if not text:
            continue
        selected.append(int(text))
    missing = sorted(set(selected) - set(available_ids))
    if missing:
        raise ValueError(f"Requested image ids are not present in artifact: {missing}")
    return selected


def _pairwise_distances(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    sq = np.sum(arr * arr, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * (arr @ arr.T), 0.0)
    return np.sqrt(d2)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    ranks = np.empty(order.shape[0], dtype=np.float64)
    ranks[order] = np.arange(order.shape[0], dtype=np.float64)
    return ranks


def _corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_rows(raw_d: np.ndarray, token_d: np.ndarray) -> np.ndarray:
    n = int(raw_d.shape[0])
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        out[i] = _corr_safe(_rankdata(raw_d[i, mask]), _rankdata(token_d[i, mask]))
    return out


def _neighbor_overlap(raw_d: np.ndarray, token_d: np.ndarray, *, k: int) -> np.ndarray:
    n = int(raw_d.shape[0])
    kk = max(1, min(int(k), n - 1))
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        raw_order = np.argsort(raw_d[i], kind="mergesort")
        tok_order = np.argsort(token_d[i], kind="mergesort")
        raw_top = set([int(j) for j in raw_order if int(j) != i][:kk])
        tok_top = set([int(j) for j in tok_order if int(j) != i][:kk])
        out[i] = len(raw_top & tok_top) / float(kk)
    return out


def _normalize_distance_matrix(d: np.ndarray) -> np.ndarray:
    mat = np.asarray(d, dtype=np.float64)
    if mat.size == 0:
        return mat
    mask = np.isfinite(mat) & (mat > 0)
    scale = float(np.quantile(mat[mask], 0.95)) if np.any(mask) else 1.0
    if scale <= 1e-12:
        scale = 1.0
    return np.clip(mat / scale, 0.0, 1.0)


def _grid_shape_from_boxes(boxes: torch.Tensor) -> tuple[int, int]:
    b = boxes.detach().cpu().numpy()
    ys = np.unique(b[:, 1].astype(np.int64))
    xs = np.unique(b[:, 0].astype(np.int64))
    return int(max(1, len(ys))), int(max(1, len(xs)))


def _sort_indices_by_grid(boxes: torch.Tensor, indices: np.ndarray) -> np.ndarray:
    b = boxes[torch.as_tensor(indices, dtype=torch.long)].detach().cpu().numpy()
    order = np.lexsort((b[:, 0], b[:, 1]))
    return np.asarray(indices, dtype=np.int64)[order]


def _extract_patch_vectors(
    *,
    images01: torch.Tensor,
    image_ids: torch.Tensor,
    bboxes: torch.Tensor,
    token_indices: np.ndarray,
    patch_size: int | None,
) -> np.ndarray:
    patches: list[torch.Tensor] = []
    imgs = images01.detach().float().cpu()
    ids = image_ids.detach().cpu().long()
    boxes = bboxes.detach().cpu().long()
    c, h, w = int(imgs.shape[1]), int(imgs.shape[2]), int(imgs.shape[3])
    if patch_size is None:
        widths = (boxes[:, 2] - boxes[:, 0]).float()
        ps = int(max(1, round(float(torch.median(widths).item()))))
    else:
        ps = int(max(1, patch_size))
    for token_idx in np.asarray(token_indices, dtype=np.int64).tolist():
        img_idx = int(ids[token_idx])
        x0, y0, x1, y1 = [int(v) for v in boxes[token_idx].tolist()]
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(x0 + 1, min(w, x1))
        y1 = max(y0 + 1, min(h, y1))
        patch = imgs[img_idx : img_idx + 1, :, y0:y1, x0:x1]
        if int(patch.shape[-2]) != ps or int(patch.shape[-1]) != ps:
            patch = F.interpolate(patch, size=(ps, ps), mode="bilinear", align_corners=False)
        patches.append(patch.reshape(c * ps * ps))
    return torch.stack(patches, dim=0).numpy().astype(np.float64)


def _draw_grid(ax, *, image_h: int, image_w: int, grid_h: int, grid_w: int) -> None:
    for col in range(1, int(grid_w)):
        x = image_w * col / float(grid_w)
        ax.axvline(x, color="white", linewidth=0.45, alpha=0.35)
    for row in range(1, int(grid_h)):
        y = image_h * row / float(grid_h)
        ax.axhline(y, color="white", linewidth=0.45, alpha=0.35)


def _image_heatmap_figure(
    *,
    images01: torch.Tensor,
    rows: list[dict[str, Any]],
    score_key: str,
    title: str,
    colorbar_label: str,
    out_path: Path,
    max_images: int,
    cmap: str,
    vmin: float,
    vmax: float,
    overlay_alpha: float = 0.48,
) -> Path:
    selected = rows[: max(1, int(max_images))]
    cols = 4
    n = len(selected)
    grid_rows = int(math.ceil(n / cols))
    fig = plt.figure(
        figsize=(4.05 * cols + 0.75, 4.08 * grid_rows + 1.25),
        constrained_layout=False,
    )
    gs = fig.add_gridspec(
        grid_rows,
        cols + 1,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.055],
        left=0.03,
        right=0.955,
        top=0.925,
        bottom=0.085,
        wspace=0.08,
        hspace=0.20,
    )
    axes = np.empty((grid_rows, cols), dtype=object)
    for r in range(grid_rows):
        for c in range(cols):
            axes[r, c] = fig.add_subplot(gs[r, c])
    cax = fig.add_subplot(gs[:, -1])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, selected):
        image_id = int(row["image_id"])
        grid_h = int(row["grid_h"])
        grid_w = int(row["grid_w"])
        image = images01[image_id].permute(1, 2, 0).numpy()
        values = np.asarray(row[score_key], dtype=np.float64).reshape(grid_h, grid_w)
        ax.imshow(np.clip(image, 0.0, 1.0), interpolation="bilinear")
        ax.imshow(
            values,
            cmap=cmap,
            norm=norm,
            alpha=overlay_alpha,
            interpolation="nearest",
            extent=(0, image.shape[1], image.shape[0], 0),
        )
        _draw_grid(ax, image_h=image.shape[0], image_w=image.shape[1], grid_h=grid_h, grid_w=grid_w)
        ax.set_title(
            f"image {image_id} | mean {np.nanmean(values):.2f}",
            fontsize=12,
            pad=7,
        )
    fig.suptitle(title, fontsize=20, y=0.982)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(colorbar_label, fontsize=13, labelpad=10)
    cbar.ax.tick_params(labelsize=11)
    fig.text(
        0.03,
        0.035,
        "Each translucent square is one image patch. Scores compare within-image pairwise distances: raw RGB patch geometry versus model-token geometry.",
        fontsize=11,
        color="#333333",
    )
    path = save_figure(fig, out_path, dpi=220)
    plt.close(fig)
    return path


def _matrix_gallery(
    *,
    images01: torch.Tensor,
    rows: list[dict[str, Any]],
    title: str,
    out_path: Path,
    max_images: int,
) -> Path:
    selected = rows[: max(1, int(max_images))]
    num_rows = len(selected)
    fig = plt.figure(
        figsize=(18.5, 3.85 * num_rows + 1.35),
        constrained_layout=False,
    )
    gs = fig.add_gridspec(
        num_rows,
        4,
        left=0.035,
        right=0.985,
        top=0.925,
        bottom=0.155,
        wspace=0.09,
        hspace=0.25,
    )
    axes = np.empty((num_rows, 4), dtype=object)
    for r in range(num_rows):
        for c in range(4):
            axes[r, c] = fig.add_subplot(gs[r, c])
    im2 = None
    im3 = None
    for r, row in enumerate(selected):
        image_id = int(row["image_id"])
        image = images01[image_id].permute(1, 2, 0).numpy()
        raw = np.asarray(row["raw_matrix_norm"], dtype=np.float64)
        tok = np.asarray(row["token_matrix_norm"], dtype=np.float64)
        diff = np.abs(raw - tok)
        axes[r, 0].imshow(np.clip(image, 0.0, 1.0), interpolation="bilinear")
        _draw_grid(
            axes[r, 0],
            image_h=image.shape[0],
            image_w=image.shape[1],
            grid_h=int(row["grid_h"]),
            grid_w=int(row["grid_w"]),
        )
        axes[r, 0].set_title(f"image {image_id}", fontsize=12, pad=7)
        im1 = axes[r, 1].imshow(raw, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[r, 1].set_title("raw patch distances", fontsize=12, pad=7)
        im2 = axes[r, 2].imshow(tok, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[r, 2].set_title("token distances", fontsize=12, pad=7)
        im3 = axes[r, 3].imshow(diff, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
        axes[r, 3].set_title(f"|difference| | corr {row['matrix_spearman']:.2f}", fontsize=12, pad=7)
        for c in range(4):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    fig.suptitle(title, fontsize=20, y=0.982)
    if im2 is not None and im3 is not None:
        cax1 = fig.add_axes([0.34, 0.062, 0.24, 0.018])
        cbar1 = fig.colorbar(im2, cax=cax1, orientation="horizontal")
        cbar1.ax.set_title("normalized distance", fontsize=12, pad=8)
        cbar1.ax.tick_params(labelsize=10)
        cax2 = fig.add_axes([0.72, 0.062, 0.24, 0.018])
        cbar2 = fig.colorbar(im3, cax=cax2, orientation="horizontal")
        cbar2.ax.set_title("absolute normalized difference", fontsize=12, pad=8)
        cbar2.ax.tick_params(labelsize=10)
    fig.text(
        0.035,
        0.022,
        "Each distance matrix is normalized by its 95th percentile off-diagonal distance before comparison.",
        fontsize=11,
        color="#333333",
    )
    path = save_figure(fig, out_path, dpi=220)
    plt.close(fig)
    return path


def _analyze_image(
    *,
    embeddings: torch.Tensor,
    images01: torch.Tensor,
    image_ids: torch.Tensor,
    bboxes: torch.Tensor,
    image_id: int,
    patch_size: int | None,
    neighbor_k: int,
    raw_distance_mode: str,
    token_distance_mode: str,
) -> dict[str, Any] | None:
    indices = torch.nonzero(image_ids.long() == int(image_id), as_tuple=False).flatten().cpu().numpy()
    if indices.size < 4:
        return None
    indices = _sort_indices_by_grid(bboxes, indices)
    image_boxes = bboxes[torch.as_tensor(indices, dtype=torch.long)]
    grid_h, grid_w = _grid_shape_from_boxes(image_boxes)
    expected = grid_h * grid_w
    if expected != int(indices.size):
        side = int(round(math.sqrt(int(indices.size))))
        grid_h = grid_w = max(1, side)
    raw = _extract_patch_vectors(
        images01=images01,
        image_ids=image_ids,
        bboxes=bboxes,
        token_indices=indices,
        patch_size=patch_size,
    )
    token = embeddings[torch.as_tensor(indices, dtype=torch.long)].detach().float().cpu().numpy()
    raw_feat = _raw_patch_features(raw, mode=raw_distance_mode)
    token_feat = _token_features(token, mode=token_distance_mode)
    raw_d = _pairwise_distances(raw_feat)
    token_d = _pairwise_distances(token_feat)
    row_spearman = _spearman_rows(raw_d, token_d)
    rank_disagreement = (1.0 - row_spearman) * 0.5
    rank_disagreement = np.clip(rank_disagreement, 0.0, 1.0)
    overlap = _neighbor_overlap(raw_d, token_d, k=neighbor_k)
    neighbor_disagreement = 1.0 - overlap
    raw_norm = _normalize_distance_matrix(raw_d)
    token_norm = _normalize_distance_matrix(token_d)
    triu = np.triu_indices(int(raw_norm.shape[0]), k=1)
    matrix_spearman = _corr_safe(_rankdata(raw_norm[triu]), _rankdata(token_norm[triu]))
    return {
        "image_id": int(image_id),
        "token_count": int(indices.size),
        "grid_h": int(grid_h),
        "grid_w": int(grid_w),
        "mean_rank_spearman": float(np.nanmean(row_spearman)),
        "mean_rank_disagreement": float(np.nanmean(rank_disagreement)),
        "mean_neighbor_overlap": float(np.nanmean(overlap)),
        "mean_neighbor_disagreement": float(np.nanmean(neighbor_disagreement)),
        "matrix_spearman": float(matrix_spearman),
        "rank_agreement_grid": row_spearman.reshape(grid_h, grid_w),
        "rank_disagreement_grid": rank_disagreement.reshape(grid_h, grid_w),
        "neighbor_overlap_grid": overlap.reshape(grid_h, grid_w),
        "neighbor_disagreement_grid": neighbor_disagreement.reshape(grid_h, grid_w),
        "raw_matrix_norm": raw_norm,
        "token_matrix_norm": token_norm,
    }


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifact = _torch_load(args.embeddings)
    embeddings = artifact["embeddings"].detach().float().cpu()
    images = artifact["images"].detach().float().cpu()
    image_ids = artifact["image_ids"].detach().cpu().long()
    bboxes = artifact["bboxes"].detach().cpu().long()
    images01 = denormalize_images(images, args.dataset).cpu()

    available_ids = sorted({int(v) for v in image_ids.tolist()})
    selected_image_ids = _parse_image_ids(args.image_ids, available_ids, args.max_images)
    rows: list[dict[str, Any]] = []
    for image_id in selected_image_ids:
        row = _analyze_image(
            embeddings=embeddings,
            images01=images01,
            image_ids=image_ids,
            bboxes=bboxes,
            image_id=image_id,
            patch_size=args.patch_size,
            neighbor_k=args.neighbor_k,
            raw_distance_mode=args.raw_distance_mode,
            token_distance_mode=args.token_distance_mode,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No image groups could be analyzed in {args.embeddings}")

    csv_path = args.out_dir / f"{args.slug}_patch_token_distance_summary.csv"
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "image_id",
                "token_count",
                "grid_h",
                "grid_w",
                "mean_rank_spearman",
                "mean_rank_disagreement",
                "mean_neighbor_overlap",
                "mean_neighbor_disagreement",
                "matrix_spearman",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    summary = {
        "label": args.label,
        "source_embeddings": str(args.embeddings),
        "num_images": len(rows),
        "selected_image_ids": [int(v) for v in selected_image_ids],
        "neighbor_k": int(args.neighbor_k),
        "patch_size": int(args.patch_size) if args.patch_size is not None else None,
        "raw_distance_mode": args.raw_distance_mode,
        "token_distance_mode": args.token_distance_mode,
        "mean_rank_spearman": float(np.nanmean([row["mean_rank_spearman"] for row in rows])),
        "mean_rank_disagreement": float(np.nanmean([row["mean_rank_disagreement"] for row in rows])),
        "mean_neighbor_overlap": float(np.nanmean([row["mean_neighbor_overlap"] for row in rows])),
        "mean_neighbor_disagreement": float(np.nanmean([row["mean_neighbor_disagreement"] for row in rows])),
        "mean_matrix_spearman": float(np.nanmean([row["matrix_spearman"] for row in rows])),
        "csv_path": str(csv_path),
    }

    rank_agreement_path = _image_heatmap_figure(
        images01=images01,
        rows=rows,
        score_key="rank_agreement_grid",
        title=f"{args.label}: Raw-Patch vs Token Distance-Rank Agreement",
        colorbar_label="Spearman rank correlation",
        out_path=args.out_dir / f"{args.slug}_patch_token_rank_agreement_heatmaps.png",
        max_images=args.max_images,
        cmap="viridis",
        vmin=-0.25,
        vmax=0.85,
        overlay_alpha=0.44,
    )
    neighbor_overlap_path = _image_heatmap_figure(
        images01=images01,
        rows=rows,
        score_key="neighbor_overlap_grid",
        title=f"{args.label}: Raw-Patch vs Token Top-{args.neighbor_k} Neighbor Overlap",
        colorbar_label=f"top-{args.neighbor_k} neighbor overlap",
        out_path=args.out_dir / f"{args.slug}_patch_token_neighbor_overlap_heatmaps.png",
        max_images=args.max_images,
        cmap="viridis",
        vmin=0.0,
        vmax=0.75,
        overlay_alpha=0.44,
    )
    mismatch_path = _image_heatmap_figure(
        images01=images01,
        rows=rows,
        score_key="rank_disagreement_grid",
        title=f"{args.label}: Raw-Patch vs Token Distance-Rank Disagreement",
        colorbar_label="distance-rank disagreement",
        out_path=args.out_dir / f"{args.slug}_patch_token_rank_disagreement_heatmaps.png",
        max_images=args.max_images,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        overlay_alpha=0.44,
    )
    overlap_path = _image_heatmap_figure(
        images01=images01,
        rows=rows,
        score_key="neighbor_disagreement_grid",
        title=f"{args.label}: Raw-Patch vs Token Top-{args.neighbor_k} Neighbor Disagreement",
        colorbar_label=f"1 - top-{args.neighbor_k} neighbor overlap",
        out_path=args.out_dir / f"{args.slug}_patch_token_neighbor_disagreement_heatmaps.png",
        max_images=args.max_images,
        cmap="plasma",
        vmin=0.0,
        vmax=1.0,
        overlay_alpha=0.44,
    )
    matrix_path = _matrix_gallery(
        images01=images01,
        rows=rows,
        title=f"{args.label}: Patch and Token Pairwise Distance Matrices",
        out_path=args.out_dir / f"{args.slug}_patch_token_distance_matrix_gallery.png",
        max_images=args.matrix_images,
    )
    summary.update(
        {
            "rank_agreement_heatmap": str(rank_agreement_path),
            "neighbor_overlap_heatmap": str(neighbor_overlap_path),
            "rank_disagreement_heatmap": str(mismatch_path),
            "neighbor_disagreement_heatmap": str(overlap_path),
            "distance_matrix_gallery": str(matrix_path),
        }
    )
    summary_path = args.out_dir / f"{args.slug}_patch_token_distance_summary.json"
    with summary_path.open("w") as fp:
        json.dump(to_serializable(summary), fp, indent=2)

    print(
        f"[patch_token_distance] {args.label}: rank_corr={summary['mean_rank_spearman']:.3f} "
        f"neighbor_overlap={summary['mean_neighbor_overlap']:.3f} matrix_corr={summary['mean_matrix_spearman']:.3f}"
    )
    print(f"[patch_token_distance] wrote {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True, help="Saved epoch_000.pt artifact.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--dataset", default="coco")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--neighbor-k", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument(
        "--image-ids",
        default=None,
        help="Comma-separated local image ids to plot, in order. Use this for paired model comparisons.",
    )
    parser.add_argument("--matrix-images", type=int, default=4)
    parser.add_argument(
        "--raw-distance-mode",
        choices=["raw_l2", "image_standardized", "patch_centered_cosine"],
        default="raw_l2",
        help="How to prepare raw RGB patch vectors before pairwise distances.",
    )
    parser.add_argument(
        "--token-distance-mode",
        choices=["raw_l2", "feature_standardized", "l2_normalized"],
        default="raw_l2",
        help="How to prepare model token vectors before pairwise distances.",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
