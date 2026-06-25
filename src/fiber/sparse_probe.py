"""Sparse dictionary probes for local token neighborhoods.

The local probe tests whether a fixed-radius or fixed-k neighborhood in
representation space needs more or fewer dictionary atoms to reconstruct the
corresponding raw image patches at a fixed residual threshold. Each eligible
token gets its own local dictionary. A separate middle-patch volume curve can
probe one anchor while fitting a fresh dictionary at each neighborhood volume.
"""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from fiber.figure_io import save_figure
from fiber.geometry import min_change_pvalue, min_fiber_violation_pvalue
from utils import to_serializable

try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    Rectangle = None
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
    return min_fiber_violation_pvalue(result)


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
    image_ids: torch.Tensor | None = None,
) -> np.ndarray:
    """Select token indices to probe without using a geometric projection.

    ``None`` or a non-positive cap means "all tokens".  A positive cap is a
    runtime control.  When ``image_ids`` are provided, capped selection keeps
    complete source-image token groups so image-projected heatmaps are not
    missing arbitrary patches.
    """
    n = int(embeddings.shape[0])
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    if max_tokens is None or int(max_tokens) <= 0 or int(max_tokens) >= n:
        return np.arange(n, dtype=np.int64)

    take = int(max_tokens)
    if image_ids is not None and int(image_ids.numel()) >= n:
        ids_np = image_ids.detach().cpu().numpy()[:n]
        selected: list[int] = []
        seen: set[int] = set()
        for image_id_raw in ids_np.tolist():
            image_id = int(image_id_raw)
            if image_id in seen:
                continue
            seen.add(image_id)
            group = np.flatnonzero(ids_np == image_id).astype(np.int64)
            if group.size == 0:
                continue
            if selected and len(selected) + int(group.size) > take:
                break
            if not selected and int(group.size) > take:
                # Prefer one complete image over a partial heatmap when the cap
                # is smaller than a single patch grid.
                selected.extend(group.tolist())
                break
            selected.extend(group.tolist())
            if len(selected) >= take:
                break
        return np.asarray(selected, dtype=np.int64)

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


def _dictionary_is_orthonormal(dictionary: np.ndarray, *, atol: float = 1e-6) -> bool:
    atoms = np.asarray(dictionary, dtype=np.float64)
    if atoms.ndim != 2 or atoms.shape[0] == 0:
        return False
    gram = atoms @ atoms.T
    eye = np.eye(int(atoms.shape[0]), dtype=np.float64)
    return bool(np.allclose(gram, eye, atol=atol, rtol=1e-5))


def omp_required_sparsity(
    target: np.ndarray,
    dictionary: np.ndarray,
    *,
    residual_threshold: float,
    max_sparsity: int,
    orthonormal_dictionary: bool = False,
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
    if orthonormal_dictionary:
        coeffs = atoms @ x
        coeff_energy = np.sort(np.square(coeffs))[::-1][:max_s]
        explained = np.cumsum(coeff_energy)
        rel_residuals = np.sqrt(np.maximum((x_norm * x_norm) - explained, 0.0)) / x_norm
        hit_idx = np.flatnonzero(rel_residuals <= threshold)
        if hit_idx.size:
            idx = int(hit_idx[0])
            return idx + 1, float(rel_residuals[idx])
        return max_s, float(rel_residuals[-1])

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


def _hard_threshold(coeffs: np.ndarray, sparsity: int) -> np.ndarray:
    s = max(1, min(int(sparsity), int(coeffs.size)))
    out = np.zeros_like(coeffs)
    idx = np.argpartition(np.abs(coeffs), -s)[-s:]
    out[idx] = coeffs[idx]
    return out


def iht_required_sparsity(
    target: np.ndarray,
    dictionary: np.ndarray,
    *,
    residual_threshold: float,
    max_sparsity: int,
    steps: int = 80,
    lr: float | None = None,
) -> tuple[int, float]:
    """IHT code length needed to hit a relative residual threshold."""
    atoms = np.asarray(dictionary, dtype=np.float64)
    x = np.asarray(target, dtype=np.float64).reshape(-1)
    if atoms.ndim != 2 or atoms.shape[0] == 0 or x.size != atoms.shape[1]:
        return 0, float("nan")

    x_norm = float(np.linalg.norm(x))
    if x_norm <= 1e-12:
        return 0, 0.0

    max_s = max(1, min(int(max_sparsity), int(atoms.shape[0])))
    threshold = max(0.0, float(residual_threshold))
    n_steps = max(1, int(steps))
    if lr is None or not math.isfinite(float(lr)) or float(lr) <= 0:
        try:
            spectral = float(np.linalg.norm(atoms, ord=2) ** 2)
        except Exception:
            spectral = float("nan")
        step_size = 1.0 / max(spectral, 1e-8) if math.isfinite(spectral) else 0.25
    else:
        step_size = float(lr)

    best_residual = float("inf")
    best_sparsity = max_s
    for sparsity in range(1, max_s + 1):
        coeffs = np.zeros(int(atoms.shape[0]), dtype=np.float64)
        rel_residual = float("inf")
        for _ in range(n_steps):
            residual = x - coeffs @ atoms
            coeffs = _hard_threshold(coeffs + step_size * (atoms @ residual), sparsity)
            rel_residual = float(np.linalg.norm(x - coeffs @ atoms) / x_norm)
            if rel_residual <= threshold:
                return sparsity, rel_residual
        if rel_residual < best_residual:
            best_residual = rel_residual
            best_sparsity = sparsity

    return best_sparsity, best_residual


def sparse_code_required_sparsity(
    target: np.ndarray,
    dictionary: np.ndarray,
    *,
    residual_threshold: float,
    max_sparsity: int,
    algorithm: str = "omp",
    iht_steps: int = 80,
    iht_lr: float | None = None,
    orthonormal_dictionary: bool = False,
) -> tuple[int, float]:
    algo = str(algorithm or "omp").strip().lower()
    if algo == "omp":
        return omp_required_sparsity(
            target,
            dictionary,
            residual_threshold=residual_threshold,
            max_sparsity=max_sparsity,
            orthonormal_dictionary=orthonormal_dictionary,
        )
    if algo == "iht":
        return iht_required_sparsity(
            target,
            dictionary,
            residual_threshold=residual_threshold,
            max_sparsity=max_sparsity,
            steps=iht_steps,
            lr=iht_lr,
        )
    raise ValueError(f"Unsupported sparse coding algorithm '{algorithm}'; expected 'omp' or 'iht'")


def _code_patch_matrix(
    x: np.ndarray,
    *,
    dictionary_size: int,
    residual_threshold: float,
    max_sparsity: int,
    algorithm: str = "omp",
    iht_steps: int = 80,
    iht_lr: float | None = None,
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
    orthonormal_dictionary = _dictionary_is_orthonormal(dictionary) if str(algorithm or "omp").strip().lower() == "omp" else False
    for row in x_std:
        sparse, residual = sparse_code_required_sparsity(
            row - mean,
            dictionary,
            residual_threshold=residual_threshold,
            max_sparsity=max_sparsity,
            algorithm=algorithm,
            iht_steps=iht_steps,
            iht_lr=iht_lr,
            orthonormal_dictionary=orthonormal_dictionary,
        )
        sparsities.append(int(sparse))
        residuals.append(float(residual))
        if math.isfinite(residual) and residual <= residual_threshold:
            hits += 1

    sparsity_arr = np.asarray(sparsities, dtype=np.float64)
    residual_arr = np.asarray(residuals, dtype=np.float64)
    return {
        "coding_algorithm": str(algorithm or "omp").strip().lower(),
        "dictionary_atoms": int(dictionary.shape[0]),
        "mean_required_sparsity": float(np.nanmean(sparsity_arr)),
        "median_required_sparsity": float(np.nanmedian(sparsity_arr)),
        "std_required_sparsity": float(np.nanstd(sparsity_arr)),
        "mean_relative_residual": float(np.nanmean(residual_arr)),
        "residual_hit_ratio": float(hits / max(1, len(sparsities))),
        "required_sparsities": [int(v) for v in sparsities],
        "relative_residuals": [float(v) for v in residuals],
    }


def _caption_for_figure(
    fig: Any,
    caption: str,
    *,
    max_width: int = 155,
    min_width: int = 72,
) -> tuple[str, int]:
    width = int(max(min_width, min(max_width, float(fig.get_figwidth()) * 11.5)))
    text = textwrap.fill(caption, width=width)
    return text, max(1, text.count("\n") + 1)


def _compact_caption(caption: str | None, *, max_chars: int = 220) -> str:
    if not caption:
        return ""
    text = " ".join(str(caption).split())
    conclusion = text.split(" The probe evaluated", 1)[0]
    if len(conclusion) < 40 and "." in text:
        conclusion = text.split(".", 1)[0] + "."
    return textwrap.shorten(conclusion, width=max_chars, placeholder="...")


def _finite_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _style_sparse_axis(ax: Any) -> None:
    ax.set_facecolor("#fbfbfb")
    ax.grid(alpha=0.22, linewidth=0.7)
    ax.tick_params(axis="both", labelsize=12)
    for spine in ax.spines.values():
        spine.set_alpha(0.55)


def _build_sparse_probe_plot(
    *,
    tokens: list[dict[str, Any]],
    out_path: Path,
    residual_threshold: float,
    max_sparsity: int,
    caption: str | None = None,
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

    fig, axes = plt.subplots(2, 3, figsize=(16.4, 9.8))
    ax0, ax1, ax2, ax3, ax4, ax5 = axes.flatten()
    for ax in axes.flatten():
        _style_sparse_axis(ax)

    finite_sparsity = sparsity[np.isfinite(sparsity)]
    y_max = float(max(max_sparsity, np.nanmax(finite_sparsity) if finite_sparsity.size else max_sparsity))
    if finite_sparsity.size:
        bins = np.arange(max(0, math.floor(float(np.nanmin(finite_sparsity))) - 1), math.ceil(y_max) + 2) - 0.5
        ax0.hist(finite_sparsity, bins=bins, color="#456990", edgecolor="white", linewidth=0.5)
        mean_val = float(np.nanmean(finite_sparsity))
        median_val = float(np.nanmedian(finite_sparsity))
        cap_share = float(np.mean(finite_sparsity >= max_sparsity)) if int(max_sparsity) > 0 else float("nan")
        ax0.axvline(mean_val, color="#d55e00", linewidth=1.6, label=f"mean {mean_val:.2f}")
        ax0.axvline(median_val, color="#009e73", linewidth=1.4, linestyle="--", label=f"median {median_val:.2f}")
        if math.isfinite(cap_share):
            ax0.text(
                0.98,
                0.93,
                f"{cap_share:.0%} at cap",
                transform=ax0.transAxes,
                ha="right",
                va="top",
                fontsize=13,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.85},
            )
        ax0.legend(loc="upper left", fontsize=13, frameon=True)
    ax0.set_xlabel("mean atoms required per patch", fontsize=13)
    ax0.set_ylabel("token neighborhoods", fontsize=13)
    ax0.set_title("Distribution", fontsize=15, pad=9)

    ax1.scatter(token_index[order], sparsity[order], color="#2f5d8a", s=8, alpha=0.38, edgecolors="none", rasterized=True)
    ax1.set_xlabel("token index (collection order)", fontsize=13)
    ax1.set_ylabel("mean atoms required", fontsize=13)
    ax1.set_ylim(0, y_max + 0.75)
    ax1.set_title("Per-Token Difficulty", fontsize=15, pad=9)

    x_count, y_count = _finite_pair(patch_count, sparsity)
    if x_count.size:
        hb = ax2.hexbin(x_count, y_count, gridsize=32, mincnt=1, cmap="Blues", linewidths=0.0)
        cbar = fig.colorbar(hb, ax=ax2, fraction=0.046, pad=0.035)
        cbar.set_label("token count", fontsize=13)
        cbar.ax.tick_params(labelsize=11)
    ax2.set_xlabel("fixed-radius patch count", fontsize=13)
    ax2.set_ylabel("mean atoms required", fontsize=13)
    ax2.set_ylim(0, y_max + 0.75)
    ax2.set_title("Neighborhood Size vs Difficulty", fontsize=15, pad=9)

    scatter_dim = ax3.scatter(
        dims,
        sparsity,
        c=patch_count,
        cmap="viridis",
        s=12,
        alpha=0.55,
        edgecolors="none",
        rasterized=True,
    )
    cbar = fig.colorbar(scatter_dim, ax=ax3, fraction=0.046, pad=0.035)
    cbar.set_label("patch count", fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    ax3.set_xlabel("local volume dimension", fontsize=13)
    ax3.set_ylabel("mean atoms required", fontsize=13)
    ax3.set_ylim(0, y_max + 0.75)
    ax3.set_title("Local Dimension vs Sparse Complexity", fontsize=15, pad=9)

    scatter_irr = ax4.scatter(
        irregularity,
        sparsity,
        c=patch_count,
        cmap="viridis",
        s=12,
        alpha=0.55,
        edgecolors="none",
        rasterized=True,
    )
    cbar = fig.colorbar(scatter_irr, ax=ax4, fraction=0.046, pad=0.035)
    cbar.set_label("patch count", fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    ax4.set_xlabel("-log10(min p-value)", fontsize=13)
    ax4.set_ylabel("mean atoms required", fontsize=13)
    ax4.set_ylim(0, y_max + 0.75)
    ax4.set_title("Fiber Irregularity vs Sparse Complexity", fontsize=15, pad=9)

    scatter_res = ax5.scatter(
        mean_residual,
        sparsity,
        c=hit_ratio,
        cmap="viridis",
        s=12,
        alpha=0.55,
        edgecolors="none",
        rasterized=True,
    )
    cbar = fig.colorbar(scatter_res, ax=ax5, fraction=0.046, pad=0.035)
    cbar.set_label("residual hit ratio", fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    ax5.axvline(float(residual_threshold), color="#c44e52", linewidth=1.2, linestyle="--", label="target residual")
    ax5.set_xlabel("mean relative residual", fontsize=13)
    ax5.set_ylabel("mean atoms required", fontsize=13)
    ax5.set_ylim(0, y_max + 0.75)
    ax5.set_title("Residual Target Check", fontsize=15, pad=9)
    ax5.legend(loc="upper left", fontsize=13, frameon=True)

    fig.suptitle("Sparse Dictionary Probe", fontsize=22, y=0.965)
    footer = _compact_caption(caption, max_chars=230)
    if caption:
        caption_text, _caption_lines = _caption_for_figure(fig, footer, max_width=135, min_width=80)
        fig.subplots_adjust(left=0.065, right=0.965, bottom=0.15, top=0.88, wspace=0.34, hspace=0.50)
        fig.text(
            0.5,
            0.035,
            caption_text,
            ha="center",
            va="bottom",
            fontsize=13,
            color="#222222",
        )
    else:
        fig.subplots_adjust(left=0.065, right=0.965, bottom=0.09, top=0.88, wspace=0.34, hspace=0.50)
    save_figure(fig, out_path, dpi=200)
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


def _crop_token_patch_array(
    *,
    images: torch.Tensor,
    image_ids: torch.Tensor,
    bboxes: torch.Tensor,
    token_idx: int,
) -> np.ndarray:
    """Return a display-ready crop for one patch token."""
    fallback = np.zeros((1, 1, 3), dtype=np.float32)
    try:
        token = int(token_idx)
        if token < 0 or token >= int(image_ids.numel()) or token >= int(bboxes.shape[0]):
            return fallback
        image_idx = int(image_ids[token])
        if image_idx < 0 or image_idx >= int(images.shape[0]):
            return fallback

        image_arr = _image_to_display_array(images[image_idx])
        h, w = image_arr.shape[:2]
        x0, y0, x1, y1 = [float(v) for v in bboxes[token].tolist()]
        x0_i = max(0, min(w - 1, int(math.floor(x0))))
        y0_i = max(0, min(h - 1, int(math.floor(y0))))
        x1_i = max(x0_i + 1, min(w, int(math.ceil(x1))))
        y1_i = max(y0_i + 1, min(h, int(math.ceil(y1))))
        return image_arr[y0_i:y1_i, x0_i:x1_i]
    except Exception:
        return fallback


def _smooth_heatmap_array(heatmap: np.ndarray, *, sigma: float) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return heatmap.astype(np.float32, copy=False)
    return gaussian_filter(heatmap.astype(np.float32, copy=False), sigma=float(sigma), mode="nearest").astype(np.float32)


def _full_image_sparse_heatmap(
    *,
    image_shape: tuple[int, int],
    items: list[dict[str, Any]],
    boxes: torch.Tensor,
) -> np.ndarray | None:
    """Rasterize sparse-probe token values as a smooth full-image overlay."""
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0 or not items:
        return None

    value_sum = np.zeros((height, width), dtype=np.float32)
    value_count = np.zeros((height, width), dtype=np.float32)
    centers: list[tuple[float, float]] = []
    values: list[float] = []

    for row in items:
        token_idx = int(row["token_index"])
        if token_idx < 0 or token_idx >= int(boxes.shape[0]):
            continue
        value = float(row.get("mean_required_sparsity", float("nan")))
        if not math.isfinite(value):
            continue

        x0, y0, x1, y1 = [float(v) for v in boxes[token_idx].tolist()]
        x0_i = max(0, min(width - 1, int(math.floor(x0))))
        y0_i = max(0, min(height - 1, int(math.floor(y0))))
        x1_i = max(x0_i + 1, min(width, int(math.ceil(x1))))
        y1_i = max(y0_i + 1, min(height, int(math.ceil(y1))))

        value_sum[y0_i:y1_i, x0_i:x1_i] += value
        value_count[y0_i:y1_i, x0_i:x1_i] += 1.0
        centers.append(((x0_i + x1_i - 1) * 0.5, (y0_i + y1_i - 1) * 0.5))
        values.append(value)

    if not values:
        return None

    heatmap = np.divide(
        value_sum,
        np.maximum(value_count, 1.0),
        out=np.zeros_like(value_sum),
        where=value_count > 0,
    )
    missing = value_count <= 0
    if np.any(missing):
        centers_np = np.asarray(centers, dtype=np.float32)
        values_np = np.asarray(values, dtype=np.float32)
        ys, xs = np.nonzero(missing)
        chunk = 32768
        for start in range(0, int(xs.size), chunk):
            end = min(start + chunk, int(xs.size))
            dx = xs[start:end, None].astype(np.float32) - centers_np[None, :, 0]
            dy = ys[start:end, None].astype(np.float32) - centers_np[None, :, 1]
            nearest = np.argmin(dx * dx + dy * dy, axis=1)
            heatmap[ys[start:end], xs[start:end]] = values_np[nearest]

    sigma = max(1.5, min(height, width) / 28.0)
    return _smooth_heatmap_array(heatmap, sigma=sigma)


def _build_sparse_probe_heatmaps(
    *,
    tokens: list[dict[str, Any]],
    images: torch.Tensor,
    image_ids: torch.Tensor,
    bboxes: torch.Tensor,
    out_path: Path,
    max_images: int = 16,
    caption: str | None = None,
) -> str | None:
    if plt is None or Rectangle is None or not tokens or images.numel() == 0:
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
    vmin = float(np.nanquantile(all_values, 0.05))
    vmax = float(np.nanquantile(all_values, 0.95))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(all_values))
        vmax = vmin + 1.0

    image_count = len(selected)
    cols = min(4, image_count)
    rows_n = int(math.ceil(image_count / cols))
    footer = _compact_caption(caption, max_chars=230)
    bottom = 0.11 if footer else 0.06
    fig_width = max(7.0, 3.62 * cols + 1.05)
    fig_height = max(5.1, 3.65 * rows_n + (1.20 if footer else 0.70))
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        rows_n,
        cols + 1,
        width_ratios=[1.0] * cols + [0.055],
        left=0.035,
        right=0.925,
        bottom=bottom,
        top=0.895,
        wspace=0.08,
        hspace=0.30,
    )
    axes = np.empty((rows_n, cols), dtype=object)
    for row in range(rows_n):
        for col in range(cols):
            axes[row, col] = fig.add_subplot(gs[row, col])
            axes[row, col].axis("off")
    cbar_ax = fig.add_subplot(gs[:, -1])
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for ax, image_idx in zip(axes.flatten(), selected):
        image_arr = _image_to_display_array(images[image_idx])
        ax.imshow(image_arr)
        height, width = image_arr.shape[:2]
        for row in grouped[image_idx]:
            token_idx = int(row["token_index"])
            if token_idx < 0 or token_idx >= int(boxes.shape[0]):
                continue
            value = float(row.get("mean_required_sparsity", float("nan")))
            if not math.isfinite(value):
                continue
            x0, y0, x1, y1 = [float(v) for v in boxes[token_idx].tolist()]
            x0_i = max(0, min(width - 1, int(math.floor(x0))))
            y0_i = max(0, min(height - 1, int(math.floor(y0))))
            x1_i = max(x0_i + 1, min(width, int(math.ceil(x1))))
            y1_i = max(y0_i + 1, min(height, int(math.ceil(y1))))
            face = list(cmap(norm(value)))
            face[3] = 0.46
            edge = list(cmap(norm(value)))
            edge[3] = 0.62
            ax.add_patch(
                Rectangle(
                    (x0_i, y0_i),
                    x1_i - x0_i,
                    y1_i - y0_i,
                    facecolor=tuple(face),
                    edgecolor=tuple(edge),
                    linewidth=0.45,
                )
            )
        ax.set_title(f"image {image_idx}: {len(grouped[image_idx])} probed tokens", fontsize=14, pad=10)
        ax.axis("off")

    for ax in axes.flatten()[len(selected):]:
        ax.axis("off")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.suptitle("Sparse Complexity Projected Back to Image Patches", fontsize=22, y=0.972)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("mean required sparsity", labelpad=10, fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    if footer:
        caption_text, _caption_lines = _caption_for_figure(
            fig,
            "Brighter overlays mark patches whose embedding neighborhoods need more dictionary atoms. " + footer,
            max_width=150,
            min_width=70,
        )
        fig.text(
            0.035,
            0.035,
            caption_text,
            ha="left",
            va="bottom",
            fontsize=13,
            color="#222222",
        )
    save_figure(fig, out_path, dpi=200)
    plt.close(fig)
    return str(out_path)


def _build_volume_curve_plot(
    *,
    rows: list[dict[str, Any]],
    out_path: Path,
    residual_threshold: float,
    max_sparsity: int,
    caption: str | None = None,
) -> str | None:
    if plt is None or not rows:
        return None

    volumes = np.asarray([row["neighborhood_volume"] for row in rows], dtype=np.float64)
    mean_s = np.asarray([row["mean_required_sparsity"] for row in rows], dtype=np.float64)
    median_s = np.asarray([row["median_required_sparsity"] for row in rows], dtype=np.float64)
    hit = np.asarray([row["residual_hit_ratio"] for row in rows], dtype=np.float64)
    residual = np.asarray([row["mean_relative_residual"] for row in rows], dtype=np.float64)
    unique_images = np.asarray([row["unique_image_count"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.9))
    ax0, ax1, ax2, ax3 = axes.flatten()
    for ax in axes.flatten():
        _style_sparse_axis(ax)
        ax.set_xscale("log", base=2)

    ax0.plot(volumes, mean_s, marker="o", linewidth=1.8, label="mean")
    ax0.plot(volumes, median_s, marker="s", linewidth=1.4, linestyle="--", label="median")
    ax0.axhline(float(max_sparsity), color="#c44e52", linewidth=1.0, linestyle=":", label="cap")
    ax0.set_xlabel("neighborhood volume k", fontsize=13)
    ax0.set_ylabel("atoms required", fontsize=13)
    ax0.set_title("Per-Volume Local Dictionary Sparse Complexity", fontsize=15, pad=9)
    ax0.legend(fontsize=13)

    ax1.plot(volumes, residual, marker="o", color="#4c72b0", linewidth=1.6)
    ax1.axhline(float(residual_threshold), color="#c44e52", linewidth=1.2, linestyle="--", label="target")
    ax1.set_xlabel("neighborhood volume k", fontsize=13)
    ax1.set_ylabel("mean relative residual", fontsize=13)
    ax1.set_title("Residual Under Fixed Target", fontsize=15, pad=9)
    ax1.legend(fontsize=13)

    ax2.plot(volumes, hit, marker="o", color="#55a868", linewidth=1.6)
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_xlabel("neighborhood volume k", fontsize=13)
    ax2.set_ylabel("fraction at residual target", fontsize=13)
    ax2.set_title("Target Hit Ratio", fontsize=15, pad=9)

    ax3.plot(volumes, unique_images, marker="o", color="#8172b2", linewidth=1.6)
    ax3.set_xlabel("neighborhood volume k", fontsize=13)
    ax3.set_ylabel("unique source images", fontsize=13)
    ax3.set_title("Neighborhood Breadth", fontsize=15, pad=9)

    fig.suptitle("Middle Patch Local-Dictionary Volume Curve", fontsize=22, y=0.965)
    footer = _compact_caption(caption, max_chars=220)
    if footer:
        caption_text, _caption_lines = _caption_for_figure(fig, footer, max_width=125, min_width=76)
        fig.subplots_adjust(left=0.075, right=0.965, bottom=0.15, top=0.87, wspace=0.31, hspace=0.47)
        fig.text(0.5, 0.035, caption_text, ha="center", va="bottom", fontsize=13, color="#222222")
    else:
        fig.subplots_adjust(left=0.075, right=0.965, bottom=0.09, top=0.87, wspace=0.31, hspace=0.47)
    save_figure(fig, out_path, dpi=200)
    plt.close(fig)
    return str(out_path)


def _select_middle_patch_token(
    *,
    images: torch.Tensor,
    image_ids: torch.Tensor,
    bboxes: torch.Tensor,
) -> int | None:
    if images.numel() == 0 or image_ids.numel() == 0 or bboxes.numel() == 0:
        return None
    ids = image_ids.detach().cpu().long()
    boxes = bboxes.detach().cpu().float()
    target_image = int(images.shape[0]) // 2
    candidates = torch.nonzero(ids == target_image, as_tuple=False).flatten()
    if candidates.numel() == 0:
        target_image = int(ids[int(ids.numel() // 2)].item())
        candidates = torch.nonzero(ids == target_image, as_tuple=False).flatten()
    if candidates.numel() == 0:
        return None
    height, width = float(images.shape[-2]), float(images.shape[-1])
    centers_x = (boxes[candidates, 0] + boxes[candidates, 2]) * 0.5
    centers_y = (boxes[candidates, 1] + boxes[candidates, 3]) * 0.5
    scores = (centers_x - width * 0.5).square() + (centers_y - height * 0.5).square()
    return int(candidates[int(torch.argmin(scores).item())].item())


def _volume_curve_interpretation(summary: dict[str, Any]) -> str:
    rows = summary.get("rows", [])
    if not rows:
        return "Conclusion: the middle-patch volume curve did not have enough valid neighborhoods to summarize."
    first = rows[0]
    last = rows[-1]
    s0 = float(first.get("mean_required_sparsity", float("nan")))
    s1 = float(last.get("mean_required_sparsity", float("nan")))
    r0 = float(first.get("mean_relative_residual", float("nan")))
    r1 = float(last.get("mean_relative_residual", float("nan")))
    if math.isfinite(s0) and math.isfinite(s1):
        if s1 > s0 + 1.0:
            trend = "required sparsity increases as the neighborhood volume grows"
        elif s1 < s0 - 1.0:
            trend = "required sparsity decreases as the neighborhood volume grows"
        else:
            trend = "required sparsity stays roughly stable as the neighborhood volume grows"
    else:
        trend = "the sparsity trend is not well determined"
    return (
        f"Conclusion: with a fresh local dictionary fitted at each middle-patch neighborhood volume, {trend}. "
        f"The curve spans k={first.get('neighborhood_volume')} to k={last.get('neighborhood_volume')} "
        f"with residual target {_fmt_float(summary.get('residual_threshold'), digits=2)}; "
        f"mean residual changes from {_fmt_float(r0, digits=3)} to {_fmt_float(r1, digits=3)}."
    )


def run_neighborhood_dictionary_volume_curve(
    *,
    epoch: int,
    embeddings: torch.Tensor,
    images: torch.Tensor,
    image_ids: torch.Tensor,
    bboxes: torch.Tensor,
    dists: np.ndarray,
    out_dir: Path,
    patch_size: int | None,
    volumes: list[int] | tuple[int, ...] | None = None,
    dictionary_size: int = 64,
    residual_threshold: float = 0.15,
    max_sparsity: int = 16,
    algorithm: str = "omp",
    iht_steps: int = 80,
    iht_lr: float | None = None,
    max_dictionary_patches: int | None = 4096,
) -> dict[str, Any]:
    """Run a per-volume local-dictionary OMP curve around the spatial middle patch token.

    ``max_dictionary_patches`` is ignored and only accepted for older callers
    that used the previous global-dictionary implementation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_tokens = int(embeddings.shape[0])
    volume_values = [8, 16, 32, 64, 128, 256] if volumes is None else [int(v) for v in volumes]
    volume_values = sorted({max(2, min(n_tokens, v)) for v in volume_values if int(v) > 1})
    anchor_idx = _select_middle_patch_token(images=images, image_ids=image_ids, bboxes=bboxes)

    rows: list[dict[str, Any]] = []
    dists_np = np.asarray(dists, dtype=np.float64)
    if anchor_idx is not None and dists_np.ndim == 2:
        anchor_dists = dists_np[int(anchor_idx)]
        finite_order = np.argsort(np.where(np.isfinite(anchor_dists), anchor_dists, np.inf))
        finite_order = finite_order[np.isfinite(anchor_dists[finite_order])]
        ids_np = image_ids.detach().cpu().numpy()
        for volume in volume_values:
            neigh = finite_order[: min(int(volume), int(finite_order.size))]
            if neigh.size < 2:
                continue
            patches = extract_patch_vectors(
                images=images,
                image_ids=image_ids,
                bboxes=bboxes,
                token_indices=neigh,
                patch_size=patch_size,
            )
            coded = _code_patch_matrix(
                patches,
                dictionary_size=dictionary_size,
                residual_threshold=residual_threshold,
                max_sparsity=max_sparsity,
                algorithm=algorithm,
                iht_steps=iht_steps,
                iht_lr=iht_lr,
            )
            if coded is None:
                continue
            rows.append(
                {
                    "anchor_token_index": int(anchor_idx),
                    "neighborhood_volume": int(neigh.size),
                    "max_feature_distance": float(np.nanmax(anchor_dists[neigh])),
                    "unique_image_count": int(np.unique(ids_np[neigh]).size),
                    "dictionary_patch_count": int(patches.shape[0]),
                    **coded,
                }
            )

    dictionary_atoms = np.asarray([row.get("dictionary_atoms", np.nan) for row in rows], dtype=np.float64)
    dictionary_patch_counts = np.asarray(
        [row.get("dictionary_patch_count", np.nan) for row in rows],
        dtype=np.float64,
    )
    summary: dict[str, Any] = {
        "epoch": int(epoch),
        "enabled": True,
        "dictionary_mode": "per_volume_local",
        "anchor_mode": "middle_patch",
        "anchor_token_index": int(anchor_idx) if anchor_idx is not None else None,
        "dictionary_size": int(dictionary_size),
        "mean_dictionary_atoms": float(np.nanmean(dictionary_atoms)) if dictionary_atoms.size else float("nan"),
        "max_dictionary_atoms": int(np.nanmax(dictionary_atoms)) if dictionary_atoms.size else 0,
        "mean_dictionary_patch_count": (
            float(np.nanmean(dictionary_patch_counts)) if dictionary_patch_counts.size else float("nan")
        ),
        "max_dictionary_patch_count": int(np.nanmax(dictionary_patch_counts)) if dictionary_patch_counts.size else 0,
        "residual_threshold": float(residual_threshold),
        "max_sparsity": int(max_sparsity),
        "coding_algorithm": str(algorithm or "omp").strip().lower(),
        "requested_volumes": [int(v) for v in volume_values],
        "evaluated_volumes": int(len(rows)),
        "rows": rows,
    }
    summary["interpretation"] = _volume_curve_interpretation(summary)
    plot_path = _build_volume_curve_plot(
        rows=rows,
        out_path=out_dir / f"epoch_{epoch:03d}_local_dictionary_volume_curve.png",
        residual_threshold=residual_threshold,
        max_sparsity=max_sparsity,
        caption=summary["interpretation"],
    )
    if plot_path:
        summary["plot_path"] = plot_path

    output_path = out_dir / f"epoch_{epoch:03d}_local_dictionary_volume_curve.json"
    summary["json_path"] = str(output_path)
    payload = {"summary": summary, "rows": rows}
    with open(output_path, "w") as fp:
        json.dump(to_serializable(payload), fp, indent=2)
    return payload


# Backward-compatible name for older configs/scripts; behavior is per-volume local.
run_global_dictionary_volume_curve = run_neighborhood_dictionary_volume_curve


def _fmt_float(value: Any, *, digits: int = 2, default: str = "n/a") -> str:
    try:
        value_f = float(value)
    except Exception:
        return default
    if not math.isfinite(value_f):
        return default
    return f"{value_f:.{digits}f}"


def _build_sparse_probe_interpretation(summary: dict[str, Any]) -> str:
    mean_s = float(summary.get("mean_required_sparsity", float("nan")))
    median_s = float(summary.get("median_required_sparsity", float("nan")))
    max_s = float(summary.get("max_sparsity", float("nan")))
    evaluated = int(summary.get("evaluated_tokens", 0) or 0)
    candidate = int(summary.get("candidate_tokens", 0) or 0)
    q10 = summary.get("sparsity_q10", float("nan"))
    q90 = summary.get("sparsity_q90", float("nan"))
    corr_count = float(summary.get("corr_sparsity_patch_count", float("nan")))
    corr_dim = float(summary.get("corr_sparsity_dimension", float("nan")))
    corr_irr = float(summary.get("corr_sparsity_irregularity", float("nan")))

    if math.isfinite(mean_s) and math.isfinite(max_s) and max_s > 0:
        load = mean_s / max_s
        if load >= 0.75 or (math.isfinite(median_s) and median_s >= 0.85 * max_s):
            complexity = "Conclusion: local raw-patch neighborhoods remain hard to explain with very sparse codes under this residual target; many neighborhoods sit near the allowed sparsity cap."
        elif load <= 0.40:
            complexity = "Conclusion: many neighborhoods are reconstructable with relatively few local dictionary atoms, which is evidence for simpler local patch variation."
        else:
            complexity = "Conclusion: sparse complexity is mixed; some neighborhoods admit compact local codes while others need many atoms."
    else:
        complexity = "Conclusion: sparse complexity could not be summarized reliably for this run."

    fixed_k = summary.get("neighborhood_mode") == "knn"
    if fixed_k:
        size_text = "Because the local probe used fixed-k neighborhoods, patch count is controlled rather than a source of variation."
    elif math.isfinite(corr_count):
        if corr_count >= 0.35:
            size_text = "Larger fixed-radius neighborhoods tend to require more atoms, so part of the signal is neighborhood-size driven."
        elif corr_count <= -0.35:
            size_text = "Larger fixed-radius neighborhoods tend to require fewer atoms, suggesting the easiest neighborhoods may be the densest ones."
        else:
            size_text = "Neighborhood size has only a weak linear relationship with sparse complexity."
    else:
        size_text = "The size-complexity relationship is not well determined."

    relation_bits = []
    if math.isfinite(corr_dim):
        relation_bits.append(f"dimension correlation {corr_dim:.2f}")
    if math.isfinite(corr_irr):
        relation_bits.append(f"irregularity correlation {corr_irr:.2f}")
    relation_text = "; ".join(relation_bits) if relation_bits else "dimension and irregularity correlations are unavailable"

    coverage = f"{evaluated}/{candidate}" if candidate else str(evaluated)
    if fixed_k:
        neighborhood_text = f"with k={summary.get('neighbor_k', 'n/a')} nearest tokens"
    else:
        neighborhood_text = f"at radius {_fmt_float(summary.get('radius'), digits=3)}"
    return (
        f"{complexity} The probe evaluated {coverage} candidate tokens {neighborhood_text} "
        f"using {summary.get('coding_algorithm', 'sparse coding')} "
        f"with residual target {_fmt_float(summary.get('residual_threshold'), digits=2)}. "
        f"Mean required sparsity is {_fmt_float(mean_s)} of max {_fmt_float(max_s, digits=0)}, "
        f"with q10-q90 {_fmt_float(q10)} to {_fmt_float(q90)}. {size_text} "
        f"Cross-checks: {relation_text}."
    )


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
    neighbor_k: int | None = None,
    auto_neighbor_k: int = 32,
    auto_radius_quantile: float = 0.5,
    min_patches: int = 12,
    max_anchors: int | None = None,
    dictionary_size: int = 32,
    residual_threshold: float = 0.15,
    max_sparsity: int = 16,
    algorithm: str = "omp",
    iht_steps: int = 80,
    iht_lr: float | None = None,
    heatmap_max_images: int = 16,
) -> dict[str, Any]:
    """Run local dictionary sparse-complexity probe over token neighborhoods."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dists_np = np.asarray(dists, dtype=np.float64)
    fixed_neighbor_k = neighbor_k is not None and int(neighbor_k) > 1
    valid_radius = radius is not None and math.isfinite(float(radius)) and float(radius) > 0
    epsilon = (
        float("nan")
        if fixed_neighbor_k
        else (
            float(radius)
            if valid_radius
            else auto_radius_from_knn(dists_np, neighbor_k=auto_neighbor_k, quantile=auto_radius_quantile)
        )
    )
    radius_source = "fixed_k" if fixed_neighbor_k else ("configured" if valid_radius else "auto_knn")
    probe_tokens = select_probe_tokens(embeddings, max_tokens=max_anchors, image_ids=image_ids)

    token_neighborhoods: list[tuple[int, np.ndarray, np.ndarray]] = []
    skipped_small_neighborhoods = 0
    for token_idx in probe_tokens.tolist():
        if token_idx < 0 or token_idx >= dists_np.shape[0]:
            continue
        finite = np.flatnonzero(np.isfinite(dists_np[token_idx]))
        if fixed_neighbor_k:
            if finite.size == 0:
                continue
            k = max(2, min(int(neighbor_k), int(finite.size)))
            order = finite[np.argsort(dists_np[token_idx, finite])]
            neigh = order[:k]
        else:
            if not math.isfinite(epsilon):
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
            algorithm=algorithm,
            iht_steps=iht_steps,
            iht_lr=iht_lr,
        )
        if coded is None:
            continue

        res = fiber_results[token_idx] if token_idx < len(fiber_results) else {}
        min_p = _min_pvalue(res)
        irregularity = -math.log10(min_p + 1e-12) if math.isfinite(min_p) else 0.0
        rows.append({
            "token_index": int(token_idx),
            "anchor": int(token_idx),
            "patch_count": int(neigh.size),
            "neighborhood_token_indices": [int(v) for v in neigh.tolist()],
            "dimension": _first_dim(res),
            "min_pvalue": min_p,
            "min_change_pvalue": min_change_pvalue(res),
            "min_fiber_violation_pvalue": min_p,
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
        "neighborhood_mode": "knn" if fixed_neighbor_k else "radius",
        "neighbor_k": int(neighbor_k) if fixed_neighbor_k else None,
        "radius": float(epsilon) if math.isfinite(epsilon) else None,
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
        "coding_algorithm": str(algorithm or "omp").strip().lower(),
        "iht_steps": int(iht_steps),
        "iht_lr": float(iht_lr) if iht_lr is not None and math.isfinite(float(iht_lr)) else None,
        "heatmap_max_images": int(heatmap_max_images),
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
    summary["interpretation"] = _build_sparse_probe_interpretation(summary)

    plot_path = _build_sparse_probe_plot(
        tokens=rows,
        out_path=out_dir / f"epoch_{epoch:03d}_sparse_dictionary_probe.png",
        residual_threshold=residual_threshold,
        max_sparsity=max_sparsity,
        caption=summary["interpretation"],
    )
    if plot_path:
        summary["plot_path"] = plot_path
    heatmap_path = _build_sparse_probe_heatmaps(
        tokens=rows,
        images=images,
        image_ids=image_ids,
        bboxes=bboxes,
        out_path=out_dir / f"epoch_{epoch:03d}_sparse_dictionary_heatmaps.png",
        max_images=heatmap_max_images,
        caption=summary["interpretation"],
    )
    if heatmap_path:
        summary["heatmap_path"] = heatmap_path

    output_path = out_dir / f"epoch_{epoch:03d}_sparse_dictionary_probe.json"
    summary["json_path"] = str(output_path)
    payload = {"summary": summary, "tokens": rows, "anchors": rows}
    with open(output_path, "w") as fp:
        json.dump(to_serializable(payload), fp, indent=2)
    return payload
