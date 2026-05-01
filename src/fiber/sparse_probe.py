"""Sparse dictionary probes for local token neighborhoods.

The probe tests whether an epsilon-ball in representation space needs more or
fewer dictionary atoms to reconstruct the corresponding raw image patches at a
fixed residual threshold. Each eligible token gets its own local dictionary
trained from the patches in its fixed-radius neighborhood; there is no global
dictionary and no PCA coordinate used as a fiber coordinate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from utils import to_serializable

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _finite_corr(a: np.ndarray, b: np.ndarray) -> float:
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


def _min_pvalue(result: dict[str, Any] | None) -> float:
    if not result or not result.get("pvalues"):
        return float("nan")
    values = np.asarray(result["pvalues"], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.min(values)) if values.size else float("nan")


def _first_dim(result: dict[str, Any] | None) -> float:
    if not result or not result.get("dimensions"):
        return float("nan")
    try:
        value = float(result["dimensions"][0])
    except Exception:
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def select_probe_tokens(
    embeddings: torch.Tensor,
    *,
    max_tokens: int | None,
) -> np.ndarray:
    """Select token indices to probe without using a geometric projection.

    ``None`` or a non-positive cap means "all tokens".  A positive cap is a
    deterministic uniform subsample in collection order, intended only as a
    runtime control.
    """
    n = int(embeddings.shape[0])
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    if max_tokens is None or int(max_tokens) <= 0 or int(max_tokens) >= n:
        return np.arange(n, dtype=np.int64)

    take = int(max_tokens)
    if take == n:
        tokens = np.arange(n, dtype=np.int64)
    else:
        positions = np.linspace(0, n - 1, num=take)
        tokens = np.unique(np.round(positions).astype(np.int64))
    return tokens.astype(np.int64)


def select_fiber_anchors(
    embeddings: torch.Tensor,
    *,
    max_anchors: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper for older imports.

    The returned coordinate is the token collection index, not PCA.
    """
    tokens = select_probe_tokens(embeddings, max_tokens=max_anchors)
    coord = np.arange(int(embeddings.shape[0]), dtype=np.float64)
    return tokens, coord


def auto_radius_from_knn(
    dists: np.ndarray,
    *,
    neighbor_k: int,
    quantile: float = 0.5,
) -> float:
    """Choose epsilon as a quantile of the k-nearest-neighbor radius."""
    d = np.asarray(dists, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] < 2:
        return float("nan")
    k = max(1, min(int(neighbor_k), d.shape[1] - 1))
    kth = np.partition(d, kth=k, axis=1)[:, k]
    kth = kth[np.isfinite(kth) & (kth > 0)]
    if kth.size == 0:
        return float("nan")
    q = min(1.0, max(0.0, float(quantile)))
    return float(np.quantile(kth, q))


def extract_patch_vectors(
    *,
    images: torch.Tensor,
    image_ids: torch.Tensor,
    bboxes: torch.Tensor,
    token_indices: np.ndarray,
    patch_size: int | None,
) -> np.ndarray:
    """Extract flattened raw image patches for selected token indices."""
    if len(token_indices) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    imgs = images.detach().float().cpu()
    ids = image_ids.detach().cpu().long()
    boxes = bboxes.detach().cpu().long()
    c, h, w = int(imgs.shape[1]), int(imgs.shape[2]), int(imgs.shape[3])
    ps = int(patch_size or max(1, int((boxes[:, 2] - boxes[:, 0]).float().median().item())))
    ps = max(1, ps)

    patches: list[torch.Tensor] = []
    for token_idx in np.asarray(token_indices, dtype=np.int64).tolist():
        if token_idx < 0 or token_idx >= int(ids.numel()):
            continue
        img_idx = int(ids[token_idx])
        if img_idx < 0 or img_idx >= int(imgs.shape[0]):
            continue
        x0, y0, x1, y1 = [int(v) for v in boxes[token_idx].tolist()]
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(x0 + 1, min(w, x1))
        y1 = max(y0 + 1, min(h, y1))
        patch = imgs[img_idx : img_idx + 1, :, y0:y1, x0:x1]
        if int(patch.shape[-2]) != ps or int(patch.shape[-1]) != ps:
            patch = F.interpolate(patch, size=(ps, ps), mode="bilinear", align_corners=False)
        patches.append(patch.reshape(c * ps * ps))

    if not patches:
        return np.zeros((0, c * ps * ps), dtype=np.float32)
    return torch.stack(patches, dim=0).numpy().astype(np.float32)


def _standardize_patch_matrix(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    arr = arr - np.mean(arr, axis=1, keepdims=True)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def fit_pca_dictionary(x: np.ndarray, *, dictionary_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit a local PCA dictionary and return (atoms, mean)."""
    x_std = _standardize_patch_matrix(x)
    if x_std.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float64), np.zeros(0, dtype=np.float64)

    mean = np.mean(x_std, axis=0)
    centered = x_std - mean
    n_atoms = max(1, min(int(dictionary_size), centered.shape[0], centered.shape[1]))
    try:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        atoms = vt[:n_atoms]
    except np.linalg.LinAlgError:
        atoms = centered[:n_atoms].copy()
    atom_norms = np.linalg.norm(atoms, axis=1, keepdims=True)
    atoms = atoms / np.maximum(atom_norms, 1e-12)
    atoms = atoms[np.linalg.norm(atoms, axis=1) > 1e-8]
    return atoms.astype(np.float64), mean.astype(np.float64)


def omp_required_sparsity(
    target: np.ndarray,
    dictionary: np.ndarray,
    *,
    residual_threshold: float,
    max_sparsity: int,
) -> tuple[int, float]:
    """OMP code length needed to hit a relative residual threshold."""
    atoms = np.asarray(dictionary, dtype=np.float64)
    x = np.asarray(target, dtype=np.float64).reshape(-1)
    if atoms.ndim != 2 or atoms.shape[0] == 0 or x.size != atoms.shape[1]:
        return 0, float("nan")

    x_norm = float(np.linalg.norm(x))
    if x_norm <= 1e-12:
        return 0, 0.0

    residual = x.copy()
    selected: list[int] = []
    max_s = max(1, min(int(max_sparsity), int(atoms.shape[0])))
    threshold = max(0.0, float(residual_threshold))

    for _ in range(max_s):
        corr = atoms @ residual
        if selected:
            corr[np.asarray(selected, dtype=np.int64)] = 0.0
        atom_idx = int(np.argmax(np.abs(corr)))
        if atom_idx in selected or not np.isfinite(corr[atom_idx]):
            break
        selected.append(atom_idx)
        basis = atoms[np.asarray(selected, dtype=np.int64)].T
        coeffs, *_ = np.linalg.lstsq(basis, x, rcond=None)
        residual = x - basis @ coeffs
        rel_residual = float(np.linalg.norm(residual) / x_norm)
        if rel_residual <= threshold:
            return len(selected), rel_residual

    rel_residual = float(np.linalg.norm(residual) / x_norm)
    return len(selected), rel_residual


def _code_patch_matrix(
    x: np.ndarray,
    *,
    dictionary_size: int,
    residual_threshold: float,
    max_sparsity: int,
) -> dict[str, Any] | None:
    x_std = _standardize_patch_matrix(x)
    if x_std.shape[0] < 2:
        return None
    dictionary, mean = fit_pca_dictionary(x, dictionary_size=dictionary_size)
    if dictionary.shape[0] == 0:
        return None

    sparsities: list[int] = []
    residuals: list[float] = []
    hits = 0
    for row in x_std:
        sparse, residual = omp_required_sparsity(
            row - mean,
            dictionary,
            residual_threshold=residual_threshold,
            max_sparsity=max_sparsity,
        )
        sparsities.append(int(sparse))
        residuals.append(float(residual))
        if math.isfinite(residual) and residual <= residual_threshold:
            hits += 1

    sparsity_arr = np.asarray(sparsities, dtype=np.float64)
    residual_arr = np.asarray(residuals, dtype=np.float64)
    return {
        "dictionary_atoms": int(dictionary.shape[0]),
        "mean_required_sparsity": float(np.nanmean(sparsity_arr)),
        "median_required_sparsity": float(np.nanmedian(sparsity_arr)),
        "std_required_sparsity": float(np.nanstd(sparsity_arr)),
        "mean_relative_residual": float(np.nanmean(residual_arr)),
        "residual_hit_ratio": float(hits / max(1, len(sparsities))),
        "required_sparsities": [int(v) for v in sparsities],
        "relative_residuals": [float(v) for v in residuals],
    }


def _build_sparse_probe_plot(
    *,
    tokens: list[dict[str, Any]],
    out_path: Path,
    residual_threshold: float,
) -> str | None:
    if plt is None or not tokens:
        return None

    token_index = np.asarray([row["token_index"] for row in tokens], dtype=np.float64)
    sparsity = np.asarray([row["mean_required_sparsity"] for row in tokens], dtype=np.float64)
    patch_count = np.asarray([row["patch_count"] for row in tokens], dtype=np.float64)
    dims = np.asarray([row.get("dimension", np.nan) for row in tokens], dtype=np.float64)
    irregularity = np.asarray([row.get("irregularity", np.nan) for row in tokens], dtype=np.float64)
    mean_residual = np.asarray([row.get("mean_relative_residual", np.nan) for row in tokens], dtype=np.float64)
    hit_ratio = np.asarray([row.get("residual_hit_ratio", np.nan) for row in tokens], dtype=np.float64)
    order = np.argsort(token_index)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    ax0, ax1, ax2, ax3, ax4, ax5 = axes.flatten()

    finite_sparsity = sparsity[np.isfinite(sparsity)]
    if finite_sparsity.size:
        bins = min(24, max(6, int(np.sqrt(finite_sparsity.size))))
        ax0.hist(finite_sparsity, bins=bins, color="#456990", edgecolor="white", linewidth=0.5)
    ax0.set_xlabel("mean required sparsity")
    ax0.set_ylabel("token neighborhoods")
    ax0.set_title("Local Sparse Complexity")
    ax0.grid(alpha=0.25)

    ax1.plot(token_index[order], sparsity[order], ".", color="#2f5d8a", markersize=5)
    ax1.set_xlabel("token index (collection order)")
    ax1.set_ylabel("mean required sparsity")
    ax1.set_title("Per-Token Local Dictionaries")
    ax1.grid(alpha=0.25)

    scatter_count = ax2.scatter(patch_count, sparsity, c=hit_ratio, cmap="viridis", s=32, edgecolors="none")
    plt.colorbar(scatter_count, ax=ax2, fraction=0.046, pad=0.04, label="residual hit ratio")
    ax2.set_xlabel("fixed-radius patch count")
    ax2.set_ylabel("mean required sparsity")
    ax2.set_title("Neighborhood Size vs Sparsity")
    ax2.grid(alpha=0.25)

    scatter_dim = ax3.scatter(dims, sparsity, c=patch_count, cmap="plasma", s=32, edgecolors="none")
    plt.colorbar(scatter_dim, ax=ax3, fraction=0.046, pad=0.04, label="patch count")
    ax3.set_xlabel("local volume dimension")
    ax3.set_ylabel("mean required sparsity")
    ax3.set_title("Dimension vs Sparse Complexity")
    ax3.grid(alpha=0.25)

    scatter_irr = ax4.scatter(irregularity, sparsity, c=patch_count, cmap="plasma", s=32, edgecolors="none")
    plt.colorbar(scatter_irr, ax=ax4, fraction=0.046, pad=0.04, label="patch count")
    ax4.set_xlabel("-log10(min p-value)")
    ax4.set_ylabel("mean required sparsity")
    ax4.set_title("Irregularity vs Sparse Complexity")
    ax4.grid(alpha=0.25)

    scatter_res = ax5.scatter(mean_residual, sparsity, c=hit_ratio, cmap="viridis", s=32, edgecolors="none")
    plt.colorbar(scatter_res, ax=ax5, fraction=0.046, pad=0.04, label="residual hit ratio")
    ax5.axvline(float(residual_threshold), color="#c44e52", linewidth=1.0, linestyle="--")
    ax5.set_xlabel("mean relative residual")
    ax5.set_ylabel("mean required sparsity")
    ax5.set_title("Residual Target Check")
    ax5.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _image_to_display_array(image: torch.Tensor) -> np.ndarray:
    img = image.detach().float().cpu()
    if img.ndim != 3:
        return np.zeros((1, 1, 3), dtype=np.float32)
    arr = img.numpy()
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    elif arr.shape[0] > 3:
        arr = arr[:3]
    arr = np.transpose(arr, (1, 2, 0))
    lo, hi = np.nanpercentile(arr, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        arr = (arr - lo) / (hi - lo)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _build_sparse_probe_heatmaps(
    *,
    tokens: list[dict[str, Any]],
    images: torch.Tensor,
    image_ids: torch.Tensor,
    bboxes: torch.Tensor,
    out_path: Path,
    max_images: int = 8,
) -> str | None:
    if plt is None or not tokens or images.numel() == 0:
        return None

    ids = image_ids.detach().cpu().long()
    boxes = bboxes.detach().cpu().float()
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in tokens:
        token_idx = int(row["token_index"])
        if token_idx < 0 or token_idx >= int(ids.numel()):
            continue
        image_idx = int(ids[token_idx])
        if image_idx < 0 or image_idx >= int(images.shape[0]):
            continue
        grouped.setdefault(image_idx, []).append(row)

    if not grouped:
        return None

    def _score(items: list[dict[str, Any]]) -> tuple[float, int]:
        values = np.asarray([row["mean_required_sparsity"] for row in items], dtype=np.float64)
        values = values[np.isfinite(values)]
        spread = float(np.nanmax(values) - np.nanmin(values)) if values.size else 0.0
        return spread, len(items)

    selected = sorted(grouped, key=lambda idx: _score(grouped[idx]), reverse=True)[: max(1, int(max_images))]
    all_values = np.asarray(
        [row["mean_required_sparsity"] for idx in selected for row in grouped[idx]],
        dtype=np.float64,
    )
    all_values = all_values[np.isfinite(all_values)]
    if all_values.size == 0:
        return None
    vmin = float(np.nanmin(all_values))
    vmax = float(np.nanmax(all_values))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0

    cols = min(4, len(selected))
    rows_n = int(math.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.0 * cols, 4.0 * rows_n), squeeze=False)
    cmap = plt.get_cmap("magma")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for ax, image_idx in zip(axes.flatten(), selected):
        ax.imshow(_image_to_display_array(images[image_idx]))
        ax.set_title(f"image {image_idx}  tokens={len(grouped[image_idx])}", fontsize=9)
        ax.axis("off")
        for row in grouped[image_idx]:
            token_idx = int(row["token_index"])
            if token_idx < 0 or token_idx >= int(boxes.shape[0]):
                continue
            x0, y0, x1, y1 = [float(v) for v in boxes[token_idx].tolist()]
            width = max(1.0, x1 - x0)
            height = max(1.0, y1 - y0)
            color = cmap(norm(float(row["mean_required_sparsity"])))
            rect = plt.Rectangle(
                (x0, y0),
                width,
                height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.45,
            )
            ax.add_patch(rect)

    for ax in axes.flatten()[len(selected):]:
        ax.axis("off")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="mean required sparsity")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def run_sparse_dictionary_probe(
    *,
    epoch: int,
    embeddings: torch.Tensor,
    images: torch.Tensor,
    image_ids: torch.Tensor,
    bboxes: torch.Tensor,
    fiber_results: list[dict[str, Any]],
    dists: np.ndarray,
    out_dir: Path,
    patch_size: int | None,
    radius: float | None = None,
    auto_neighbor_k: int = 32,
    auto_radius_quantile: float = 0.5,
    min_patches: int = 12,
    max_anchors: int | None = None,
    dictionary_size: int = 32,
    residual_threshold: float = 0.15,
    max_sparsity: int = 16,
) -> dict[str, Any]:
    """Run local dictionary/OMP sparse complexity probe over token neighborhoods."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dists_np = np.asarray(dists, dtype=np.float64)
    valid_radius = radius is not None and math.isfinite(float(radius)) and float(radius) > 0
    epsilon = (
        float(radius)
        if valid_radius
        else auto_radius_from_knn(dists_np, neighbor_k=auto_neighbor_k, quantile=auto_radius_quantile)
    )
    radius_source = "configured" if valid_radius else "auto_knn"
    probe_tokens = select_probe_tokens(embeddings, max_tokens=max_anchors)

    token_neighborhoods: list[tuple[int, np.ndarray, np.ndarray]] = []
    skipped_small_neighborhoods = 0
    for token_idx in probe_tokens.tolist():
        if token_idx < 0 or token_idx >= dists_np.shape[0] or not math.isfinite(epsilon):
            continue
        neigh = np.flatnonzero(np.isfinite(dists_np[token_idx]) & (dists_np[token_idx] <= epsilon))
        if neigh.size < int(min_patches):
            skipped_small_neighborhoods += 1
            continue
        patches = extract_patch_vectors(
            images=images,
            image_ids=image_ids,
            bboxes=bboxes,
            token_indices=neigh,
            patch_size=patch_size,
        )
        if patches.shape[0] < 2:
            continue
        token_neighborhoods.append((token_idx, neigh, patches))

    rows: list[dict[str, Any]] = []
    for token_idx, neigh, patches in token_neighborhoods:
        coded = _code_patch_matrix(
            patches,
            dictionary_size=dictionary_size,
            residual_threshold=residual_threshold,
            max_sparsity=max_sparsity,
        )
        if coded is None:
            continue

        res = fiber_results[token_idx] if token_idx < len(fiber_results) else {}
        min_p = _min_pvalue(res)
        irregularity = -math.log10(min_p + 1e-12) if math.isfinite(min_p) else float("nan")
        rows.append({
            "token_index": int(token_idx),
            "anchor": int(token_idx),
            "patch_count": int(neigh.size),
            "dimension": _first_dim(res),
            "min_pvalue": min_p,
            "irregularity": irregularity,
            **coded,
        })

    mean_sparsity = np.asarray([row["mean_required_sparsity"] for row in rows], dtype=np.float64)
    patch_counts = np.asarray([row["patch_count"] for row in rows], dtype=np.float64)
    dims = np.asarray([row["dimension"] for row in rows], dtype=np.float64)
    irregularity = np.asarray([row["irregularity"] for row in rows], dtype=np.float64)
    finite_sparsity = mean_sparsity[np.isfinite(mean_sparsity)]
    sparsity_quantiles = (
        np.nanquantile(finite_sparsity, [0.10, 0.25, 0.75, 0.90])
        if finite_sparsity.size
        else np.asarray([np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
    )
    requested_tokens = None if max_anchors is None or int(max_anchors) <= 0 else int(max_anchors)

    summary: dict[str, Any] = {
        "epoch": int(epoch),
        "enabled": True,
        "dictionary_mode": "local",
        "radius": float(epsilon) if math.isfinite(epsilon) else float("nan"),
        "radius_source": radius_source,
        "auto_neighbor_k": int(auto_neighbor_k),
        "auto_radius_quantile": float(auto_radius_quantile),
        "candidate_tokens": int(embeddings.shape[0]),
        "requested_tokens": requested_tokens,
        "evaluated_tokens": int(len(rows)),
        "skipped_small_neighborhoods": int(skipped_small_neighborhoods),
        "requested_anchors": requested_tokens,
        "evaluated_anchors": int(len(rows)),
        "min_patches": int(min_patches),
        "dictionary_size": int(dictionary_size),
        "residual_threshold": float(residual_threshold),
        "max_sparsity": int(max_sparsity),
        "mean_patch_count": float(np.nanmean(patch_counts)) if patch_counts.size else float("nan"),
        "mean_required_sparsity": float(np.nanmean(mean_sparsity)) if mean_sparsity.size else float("nan"),
        "median_required_sparsity": float(np.nanmedian(mean_sparsity)) if mean_sparsity.size else float("nan"),
        "sparsity_std": float(np.nanstd(mean_sparsity)) if mean_sparsity.size else float("nan"),
        "sparsity_q10": float(sparsity_quantiles[0]),
        "sparsity_q90": float(sparsity_quantiles[3]),
        "sparsity_iqr": float(sparsity_quantiles[2] - sparsity_quantiles[1]),
        "sparsity_range": (
            float(np.nanmax(mean_sparsity) - np.nanmin(mean_sparsity))
            if mean_sparsity.size and np.any(np.isfinite(mean_sparsity))
            else float("nan")
        ),
        "corr_sparsity_patch_count": _finite_corr(mean_sparsity, patch_counts),
        "corr_sparsity_dimension": _finite_corr(mean_sparsity, dims),
        "corr_sparsity_irregularity": _finite_corr(mean_sparsity, irregularity),
    }

    plot_path = _build_sparse_probe_plot(
        tokens=rows,
        out_path=out_dir / f"epoch_{epoch:03d}_sparse_dictionary_probe.png",
        residual_threshold=residual_threshold,
    )
    if plot_path:
        summary["plot_path"] = plot_path
    heatmap_path = _build_sparse_probe_heatmaps(
        tokens=rows,
        images=images,
        image_ids=image_ids,
        bboxes=bboxes,
        out_path=out_dir / f"epoch_{epoch:03d}_sparse_dictionary_heatmaps.png",
    )
    if heatmap_path:
        summary["heatmap_path"] = heatmap_path

    output_path = out_dir / f"epoch_{epoch:03d}_sparse_dictionary_probe.json"
    summary["json_path"] = str(output_path)
    payload = {"summary": summary, "tokens": rows, "anchors": rows}
    with open(output_path, "w") as fp:
        json.dump(to_serializable(payload), fp, indent=2)
    return payload
