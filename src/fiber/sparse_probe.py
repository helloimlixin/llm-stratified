"""Sparse dictionary probes for local fiber neighborhoods.

The probe tests whether an epsilon-ball in representation space needs more or
fewer dictionary atoms to reconstruct the corresponding raw image patches at a
fixed residual threshold. It is intentionally lightweight: dictionaries are
learned by local PCA and codes are found with Orthogonal Matching Pursuit.
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


def select_fiber_anchors(
    embeddings: torch.Tensor,
    *,
    max_anchors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select anchors evenly along the first PCA coordinate."""
    n = int(embeddings.shape[0])
    if n <= 0 or max_anchors <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    emb = embeddings.detach().float().cpu()
    centered = emb - emb.mean(dim=0, keepdim=True)
    rank = min(1, int(centered.shape[0]), int(centered.shape[1]))
    if rank > 0:
        _u, _s, v = torch.pca_lowrank(centered, q=rank)
        coord = (centered @ v[:, :1]).squeeze(1).numpy().astype(np.float64)
    else:
        coord = np.zeros(n, dtype=np.float64)

    order = np.argsort(coord)
    take = min(int(max_anchors), n)
    if take == n:
        anchors = order
    else:
        positions = np.linspace(0, n - 1, num=take)
        anchors = order[np.unique(np.round(positions).astype(np.int64))]
    return anchors.astype(np.int64), coord


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
    global_dictionary: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any] | None:
    x_std = _standardize_patch_matrix(x)
    if x_std.shape[0] < 2:
        return None
    if global_dictionary is not None:
        dictionary, mean = global_dictionary
    else:
        dictionary, mean = fit_pca_dictionary(x_std, dictionary_size=dictionary_size)
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
    anchors: list[dict[str, Any]],
    out_path: Path,
    global_dict: bool = False,
) -> str | None:
    if plt is None or not anchors:
        return None

    coord = np.asarray([row["fiber_coord"] for row in anchors], dtype=np.float64)
    sparsity = np.asarray([row["mean_required_sparsity"] for row in anchors], dtype=np.float64)
    patch_count = np.asarray([row["patch_count"] for row in anchors], dtype=np.float64)
    dims = np.asarray([row.get("dimension", np.nan) for row in anchors], dtype=np.float64)
    irregularity = np.asarray([row.get("irregularity", np.nan) for row in anchors], dtype=np.float64)
    order = np.argsort(coord)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax0, ax1, ax2, ax3 = axes.flatten()

    ax0.plot(coord[order], sparsity[order], "o-", color="#2f5d8a", linewidth=1.5)
    ax0.set_xlabel("fiber coordinate (PCA-1)")
    ax0.set_ylabel("mean required sparsity")
    ax0.set_title("Sparsity Along the Fiber Coordinate (global dict)" if global_dict else "Sparsity Along the Fiber Coordinate (local dict)")
    ax0.grid(alpha=0.25)

    scatter = ax1.scatter(coord, patch_count, c=sparsity, cmap="magma", s=45, edgecolors="none")
    plt.colorbar(scatter, ax=ax1, fraction=0.046, pad=0.04, label="mean sparsity")
    ax1.set_xlabel("fiber coordinate (PCA-1)")
    ax1.set_ylabel("epsilon-ball patch count")
    ax1.set_title("Patch Count and Sparse Complexity")

    scatter_dim = ax2.scatter(dims, sparsity, c=patch_count, cmap="viridis", s=45, edgecolors="none")
    plt.colorbar(scatter_dim, ax=ax2, fraction=0.046, pad=0.04, label="patch count")
    ax2.set_xlabel("local dimension")
    ax2.set_ylabel("mean required sparsity")
    ax2.set_title("Local Dimension vs Sparse Complexity")
    ax2.grid(alpha=0.25)

    scatter_irr = ax3.scatter(irregularity, sparsity, c=patch_count, cmap="viridis", s=45, edgecolors="none")
    plt.colorbar(scatter_irr, ax=ax3, fraction=0.046, pad=0.04, label="patch count")
    ax3.set_xlabel("-log10(min p-value)")
    ax3.set_ylabel("mean required sparsity")
    ax3.set_title("Irregularity vs Sparse Complexity")
    ax3.grid(alpha=0.25)

    fig.tight_layout()
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
    max_anchors: int = 64,
    dictionary_size: int = 32,
    residual_threshold: float = 0.15,
    max_sparsity: int = 16,
    global_dictionary: bool = False,
) -> dict[str, Any]:
    """Run local dictionary/OMP sparse complexity probe over fiber neighborhoods.

    When *global_dictionary* is True a single PCA dictionary is learned from
    **all** collected patches and reused at every anchor.  Sparsity variation
    along the fiber coordinate then reflects genuine structural complexity
    differences rather than dictionary quality differences -- a cleaner test
    of the stratified manifold hypothesis.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dists_np = np.asarray(dists, dtype=np.float64)
    valid_radius = radius is not None and math.isfinite(float(radius)) and float(radius) > 0
    epsilon = (
        float(radius)
        if valid_radius
        else auto_radius_from_knn(dists_np, neighbor_k=auto_neighbor_k, quantile=auto_radius_quantile)
    )
    radius_source = "configured" if valid_radius else "auto_knn"
    anchors_idx, fiber_coord = select_fiber_anchors(embeddings, max_anchors=max_anchors)

    # -- collect per-anchor neighborhoods and their patches -----------------
    anchor_neighborhoods: list[tuple[int, np.ndarray, np.ndarray]] = []
    all_patch_blocks: list[np.ndarray] = []
    for anchor in anchors_idx.tolist():
        if anchor < 0 or anchor >= dists_np.shape[0] or not math.isfinite(epsilon):
            continue
        neigh = np.flatnonzero(np.isfinite(dists_np[anchor]) & (dists_np[anchor] <= epsilon))
        if neigh.size < int(min_patches):
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
        anchor_neighborhoods.append((anchor, neigh, patches))
        if global_dictionary:
            all_patch_blocks.append(patches)

    # -- fit global dictionary if requested ---------------------------------
    shared_dict: tuple[np.ndarray, np.ndarray] | None = None
    if global_dictionary and all_patch_blocks:
        all_patches = np.concatenate(all_patch_blocks, axis=0)
        all_std = _standardize_patch_matrix(all_patches)
        if all_std.shape[0] >= 2:
            shared_dict = fit_pca_dictionary(all_std, dictionary_size=dictionary_size)

    # -- code each anchor's neighborhood ------------------------------------
    rows: list[dict[str, Any]] = []
    for anchor, neigh, patches in anchor_neighborhoods:
        coded = _code_patch_matrix(
            patches,
            dictionary_size=dictionary_size,
            residual_threshold=residual_threshold,
            max_sparsity=max_sparsity,
            global_dictionary=shared_dict,
        )
        if coded is None:
            continue

        res = fiber_results[anchor] if anchor < len(fiber_results) else {}
        min_p = _min_pvalue(res)
        irregularity = -math.log10(min_p + 1e-12) if math.isfinite(min_p) else float("nan")
        rows.append({
            "anchor": int(anchor),
            "fiber_coord": float(fiber_coord[anchor]) if anchor < fiber_coord.size else float("nan"),
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
    coords = np.asarray([row["fiber_coord"] for row in rows], dtype=np.float64)
    order = np.argsort(coords) if coords.size else np.empty(0, dtype=np.int64)
    sorted_sparsity = mean_sparsity[order] if order.size else np.empty(0, dtype=np.float64)
    finite_sorted = sorted_sparsity[np.isfinite(sorted_sparsity)]

    summary: dict[str, Any] = {
        "epoch": int(epoch),
        "enabled": True,
        "dictionary_mode": "global" if global_dictionary else "local",
        "radius": float(epsilon) if math.isfinite(epsilon) else float("nan"),
        "radius_source": radius_source,
        "auto_neighbor_k": int(auto_neighbor_k),
        "auto_radius_quantile": float(auto_radius_quantile),
        "requested_anchors": int(max_anchors),
        "evaluated_anchors": int(len(rows)),
        "min_patches": int(min_patches),
        "dictionary_size": int(dictionary_size),
        "residual_threshold": float(residual_threshold),
        "max_sparsity": int(max_sparsity),
        "mean_patch_count": float(np.nanmean(patch_counts)) if patch_counts.size else float("nan"),
        "mean_required_sparsity": float(np.nanmean(mean_sparsity)) if mean_sparsity.size else float("nan"),
        "median_required_sparsity": float(np.nanmedian(mean_sparsity)) if mean_sparsity.size else float("nan"),
        "sparsity_std": float(np.nanstd(mean_sparsity)) if mean_sparsity.size else float("nan"),
        "sparsity_range": (
            float(np.nanmax(mean_sparsity) - np.nanmin(mean_sparsity))
            if mean_sparsity.size and np.any(np.isfinite(mean_sparsity))
            else float("nan")
        ),
        "sparsity_total_variation": (
            float(np.nansum(np.abs(np.diff(finite_sorted)))) if finite_sorted.size >= 2 else float("nan")
        ),
        "corr_sparsity_patch_count": _finite_corr(mean_sparsity, patch_counts),
        "corr_sparsity_dimension": _finite_corr(mean_sparsity, dims),
        "corr_sparsity_irregularity": _finite_corr(mean_sparsity, irregularity),
    }

    plot_path = _build_sparse_probe_plot(
        anchors=rows,
        out_path=out_dir / f"epoch_{epoch:03d}_sparse_dictionary_probe.png",
        global_dict=global_dictionary,
    )
    if plot_path:
        summary["plot_path"] = plot_path

    output_path = out_dir / f"epoch_{epoch:03d}_sparse_dictionary_probe.json"
    summary["json_path"] = str(output_path)
    payload = {"summary": summary, "anchors": rows}
    with open(output_path, "w") as fp:
        json.dump(to_serializable(payload), fp, indent=2)
    return payload
