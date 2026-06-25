"""Probe whether VAR fiber singularities align with generative ambiguity.

This script reuses an existing VAR fiber run and computes generation-side
statistics for the same image-aligned final-scale patch tokens:

* entropy of VAR's teacher-forced next-VQ-token distribution
* negative log likelihood of the observed VQ token
* top-1 probability and top-2 margin

Those quantities are compared with fiber dimension and fiber-violation
irregularity. High entropy is the direct autoregressive analogue of local
polysemy: the model has several plausible visual continuations at that patch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import create_data_loaders  # noqa: E402
from fiber.figure_io import save_figure  # noqa: E402
from fiber.geometry import min_change_pvalue, min_fiber_violation_pvalue  # noqa: E402
from models import VarAutoregressiveImageEncoder  # noqa: E402
from utils import denormalize_images  # noqa: E402


def _finite_corr(x: np.ndarray, y: np.ndarray, *, spearman: bool = False) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if spearman:
        try:
            from scipy.stats import rankdata

            x = rankdata(x)
            y = rankdata(y)
        except Exception:
            x = np.argsort(np.argsort(x)).astype(np.float64)
            y = np.argsort(np.argsort(y)).astype(np.float64)
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 0.0 or sy <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(values)
    if int(mask.sum()) == 0:
        return out
    try:
        from scipy.stats import rankdata

        out[mask] = rankdata(values[mask])
    except Exception:
        order = np.argsort(values[mask])
        ranks = np.empty(order.shape, dtype=np.float64)
        ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
        out[mask] = ranks
    return out


def _residualize(y: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    control_arrays = [np.asarray(control, dtype=np.float64) for control in controls]
    mask = np.isfinite(y)
    for control in control_arrays:
        mask &= np.isfinite(control)
    residual = np.full(y.shape, np.nan, dtype=np.float64)
    if int(mask.sum()) < len(control_arrays) + 3:
        return residual
    x_cols = [np.ones(int(mask.sum()), dtype=np.float64)]
    x_cols.extend(control[mask] for control in control_arrays)
    x = np.column_stack(x_cols)
    beta, *_ = np.linalg.lstsq(x, y[mask], rcond=None)
    residual[mask] = y[mask] - x @ beta
    return residual


def _partial_spearman(x: np.ndarray, y: np.ndarray, controls: list[np.ndarray]) -> float:
    ranked_x = _rank_values(x)
    ranked_y = _rank_values(y)
    ranked_controls = [_rank_values(control) for control in controls]
    return _finite_corr(
        _residualize(ranked_x, ranked_controls),
        _residualize(ranked_y, ranked_controls),
    )


def _permutation_mean_diff_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    *,
    reps: int = 10000,
    seed: int = 1337,
) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan")
    observed = float(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    rng = np.random.default_rng(seed)
    extreme = 0
    n_x = int(x.size)
    for _ in range(int(reps)):
        perm = rng.permutation(pooled)
        diff = float(np.mean(perm[:n_x]) - np.mean(perm[n_x:]))
        if abs(diff) >= abs(observed):
            extreme += 1
    pvalue = (extreme + 1.0) / (float(reps) + 1.0)
    return observed, float(pvalue)


def _cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan")
    pooled_var = ((x.size - 1) * np.var(x, ddof=1) + (y.size - 1) * np.var(y, ddof=1)) / (x.size + y.size - 2)
    if pooled_var <= 0.0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / math.sqrt(float(pooled_var)))


def _mean_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _quantile_or_nan(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else float("nan")


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(values, dtype=np.float64)
    mask = np.isfinite(values)
    if int(mask.sum()) < 2:
        return out
    mean = float(values[mask].mean())
    std = float(values[mask].std())
    if std <= 1e-12:
        return out
    out[mask] = (values[mask] - mean) / std
    return out


def _tail_mask(values: np.ndarray, fraction: float, *, largest: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mask = np.zeros(values.shape, dtype=bool)
    finite_idx = np.flatnonzero(np.isfinite(values))
    if finite_idx.size == 0:
        return mask
    count = max(1, int(math.ceil(float(fraction) * finite_idx.size)))
    order = finite_idx[np.argsort(values[finite_idx])]
    selected = order[-count:] if largest else order[:count]
    mask[selected] = True
    return mask


def _safe_sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def _quantile_bins(values: np.ndarray, *, bins: int = 10) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, -1, dtype=int)
    finite_idx = np.flatnonzero(np.isfinite(values))
    if finite_idx.size == 0:
        return out
    order = finite_idx[np.argsort(values[finite_idx])]
    splits = np.array_split(order, int(bins))
    for bin_idx, split in enumerate(splits):
        out[split] = bin_idx
    return out


def _violation_strength_bins(irregularity: np.ndarray, rejected: np.ndarray) -> tuple[np.ndarray, list[str]]:
    irregularity = np.asarray(irregularity, dtype=np.float64)
    rejected = np.asarray(rejected, dtype=bool)
    groups = np.zeros(irregularity.shape, dtype=int)
    labels = ["no violation", "low violation", "mid violation", "high violation"]
    rejected_idx = np.flatnonzero(rejected & np.isfinite(irregularity))
    if rejected_idx.size:
        ordered = rejected_idx[np.argsort(irregularity[rejected_idx])]
        for group_offset, split in enumerate(np.array_split(ordered, 3), start=1):
            groups[split] = group_offset
    return groups, labels


def _plot_overlay_grid(
    *,
    images: list[np.ndarray],
    maps: np.ndarray,
    out_path: Path,
    title: str,
    colorbar_label: str,
    cmap: str,
    footer: str,
    max_images: int = 16,
) -> None:
    n = min(len(images), int(max_images))
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols + 0.9, 4.45 * rows), squeeze=False)
    axes_flat = axes.ravel()
    finite = maps[:n][np.isfinite(maps[:n])]
    vmin = float(np.quantile(finite, 0.02)) if finite.size else 0.0
    vmax = float(np.quantile(finite, 0.98)) if finite.size else 1.0
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0
    im = None
    for i in range(n):
        ax = axes_flat[i]
        image = np.clip(images[i], 0.0, 1.0)
        h, w = image.shape[:2]
        ax.imshow(image)
        im = ax.imshow(
            maps[i],
            cmap=cmap,
            alpha=0.52,
            interpolation="nearest",
            extent=(0, w, h, 0),
            vmin=vmin,
            vmax=vmax,
        )
        grid = maps.shape[1]
        for edge in range(1, grid):
            x = edge * w / grid
            y = edge * h / grid
            ax.axvline(x, color="white", linewidth=0.35, alpha=0.35)
            ax.axhline(y, color="white", linewidth=0.35, alpha=0.35)
        ax.set_title(f"image {i}", fontsize=12, pad=6)
        ax.set_axis_off()
    for ax in axes_flat[n:]:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=20, y=0.985)
    fig.subplots_adjust(left=0.025, right=0.885, top=0.920, bottom=0.085, wspace=0.08, hspace=0.22)
    if im is not None:
        cax = fig.add_axes([0.910, 0.20, 0.018, 0.60])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(colorbar_label, fontsize=13, labelpad=10)
        cbar.ax.tick_params(labelsize=11)
    fig.text(0.02, 0.018, footer, fontsize=11, color="#333333")
    save_figure(fig, out_path, dpi=180)
    plt.close(fig)


def _plot_scatter(
    *,
    entropy_norm: np.ndarray,
    nll: np.ndarray,
    top1_prob: np.ndarray,
    irregularity: np.ndarray,
    dimension: np.ndarray,
    out_path: Path,
    summary: dict[str, float],
) -> None:
    fig = plt.figure(figsize=(18.8, 5.6))
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.045],
        left=0.055,
        right=0.965,
        top=0.810,
        bottom=0.170,
        wspace=0.34,
    )
    axes = np.asarray([fig.add_subplot(gs[0, idx]) for idx in range(3)], dtype=object)
    cax = fig.add_subplot(gs[0, 3])
    panels = [
        (entropy_norm, "normalized entropy", summary["corr_irregularity_entropy_spearman"]),
        (nll, "true-code NLL", summary["corr_irregularity_nll_spearman"]),
        (top1_prob, "top-1 probability", summary["corr_irregularity_top1_prob_spearman"]),
    ]
    finite_dim = dimension[np.isfinite(dimension)]
    vmin = float(np.quantile(finite_dim, 0.05)) if finite_dim.size else None
    vmax = float(np.quantile(finite_dim, 0.95)) if finite_dim.size else None
    scatter = None
    for ax, (y, ylabel, rho) in zip(axes, panels):
        mask = np.isfinite(irregularity) & np.isfinite(y) & np.isfinite(dimension)
        scatter = ax.scatter(
            irregularity[mask],
            y[mask],
            c=dimension[mask],
            cmap="viridis",
            s=13,
            alpha=0.58,
            linewidths=0,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("fiber-violation irregularity  -log10(p)", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(f"Spearman rho={rho:.3f}" if math.isfinite(rho) else "Spearman rho=nan", fontsize=13, pad=8)
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.25, linewidth=0.6)
    if scatter is not None:
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label("first scaling dimension", fontsize=13, labelpad=10)
        cbar.ax.tick_params(labelsize=12)
    else:
        cax.set_axis_off()
    fig.suptitle("Generation Ambiguity vs Fiber Singularity in VAR", fontsize=20, y=0.960)
    save_figure(fig, out_path, dpi=180)
    plt.close(fig)


def _crop_patch(image: np.ndarray, row: int, col: int, grid_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    y0 = int(round(row * h / grid_size))
    y1 = int(round((row + 1) * h / grid_size))
    x0 = int(round(col * w / grid_size))
    x1 = int(round((col + 1) * w / grid_size))
    return image[y0:y1, x0:x1]


def _select_top(score: np.ndarray, count: int, used: set[int]) -> list[int]:
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf))
    selected: list[int] = []
    for idx in order:
        idx_i = int(idx)
        if idx_i in used:
            continue
        if not np.isfinite(score[idx_i]):
            continue
        selected.append(idx_i)
        used.add(idx_i)
        if len(selected) >= count:
            break
    return selected


def _plot_patch_gallery(
    *,
    images: list[np.ndarray],
    grid_size: int,
    entropy_norm: np.ndarray,
    irregularity: np.ndarray,
    dimension: np.ndarray,
    nll: np.ndarray,
    out_path: Path,
    patches_per_row: int = 8,
) -> None:
    zi = _zscore(irregularity)
    ze = _zscore(entropy_norm)
    zn = _zscore(nll)
    categories = [
        ("singular + ambiguous", zi + ze),
        ("singular + surprising", zi + zn),
        ("ambiguous controls", -zi + ze),
        ("quiet controls", -zi - ze),
    ]
    used: set[int] = set()
    picks = [(label, _select_top(score, patches_per_row, used)) for label, score in categories]
    rows = len(picks)
    fig, axes = plt.subplots(rows, patches_per_row, figsize=(2.35 * patches_per_row, 2.72 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    for r, (label, indices) in enumerate(picks):
        for c in range(patches_per_row):
            ax = axes[r, c]
            ax.set_axis_off()
            if c >= len(indices):
                continue
            token_idx = int(indices[c])
            image_id = token_idx // (grid_size * grid_size)
            patch_id = token_idx % (grid_size * grid_size)
            row = patch_id // grid_size
            col = patch_id % grid_size
            patch = _crop_patch(images[image_id], row, col, grid_size)
            ax.imshow(np.clip(patch, 0.0, 1.0), interpolation="nearest")
            ax.set_title(
                f"i{image_id} p{patch_id}\nI {irregularity[token_idx]:.1f} H {entropy_norm[token_idx]:.2f}\n"
                f"d {dimension[token_idx]:.1f} NLL {nll[token_idx]:.1f}",
                fontsize=11,
                pad=5,
            )
        axes[r, 0].set_ylabel(label, fontsize=13, rotation=0, labelpad=68, va="center")
    fig.suptitle("Patch-Level Generation Polysemy Controls", fontsize=20, y=0.985)
    fig.text(
        0.02,
        0.012,
        "I is fiber-violation irregularity; H is normalized VAR next-token entropy. "
        "The first rows are the patches most aligned with the singularity -> polysemy hypothesis.",
        fontsize=11,
        color="#333333",
    )
    fig.subplots_adjust(left=0.100, right=0.995, top=0.865, bottom=0.090, wspace=0.24, hspace=0.66)
    save_figure(fig, out_path, dpi=180)
    plt.close(fig)


def _plot_aggregate_bins(
    *,
    dimension: np.ndarray,
    irregularity: np.ndarray,
    rejected: np.ndarray,
    entropy_norm: np.ndarray,
    nll: np.ndarray,
    out_path: Path,
) -> None:
    dim_bins = _quantile_bins(dimension, bins=10)
    viol_bins, viol_labels = _violation_strength_bins(irregularity, rejected)

    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.2), squeeze=False)
    axes_flat = axes.ravel()

    for ax, metric, ylabel, title in (
        (axes_flat[0], entropy_norm, "normalized entropy", "Entropy by dimension decile"),
        (axes_flat[1], nll, "true-code NLL", "NLL by dimension decile"),
    ):
        xs = np.arange(10)
        means = []
        sems = []
        labels = []
        for bin_idx in xs:
            mask = dim_bins == int(bin_idx)
            means.append(_mean_or_nan(metric[mask]))
            sems.append(_safe_sem(metric[mask]))
            labels.append(str(int(bin_idx + 1)))
        ax.errorbar(xs, means, yerr=sems, color="#355C7D", marker="o", linewidth=2.0, capsize=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_xlabel("dimension decile (low to high)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, pad=10)
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.24, linewidth=0.7)

    ax = axes_flat[2]
    xs = np.arange(len(viol_labels))
    entropy_means = []
    entropy_sems = []
    nll_means = []
    nll_sems = []
    counts = []
    for group_idx in xs:
        mask = viol_bins == int(group_idx)
        entropy_means.append(_mean_or_nan(entropy_norm[mask]))
        entropy_sems.append(_safe_sem(entropy_norm[mask]))
        nll_means.append(_mean_or_nan(nll[mask]))
        nll_sems.append(_safe_sem(nll[mask]))
        counts.append(int(mask.sum()))
    width = 0.36
    ax.bar(xs - width / 2, entropy_means, width, yerr=entropy_sems, color="#C06C84", alpha=0.82, capsize=3)
    ax.set_ylabel("normalized entropy", fontsize=12, color="#8F3D57")
    ax.tick_params(axis="y", labelcolor="#8F3D57", labelsize=11)
    ax2 = ax.twinx()
    ax2.bar(xs + width / 2, nll_means, width, yerr=nll_sems, color="#6C5B7B", alpha=0.82, capsize=3)
    ax2.set_ylabel("true-code NLL", fontsize=12, color="#4F415E")
    ax2.tick_params(axis="y", labelcolor="#4F415E", labelsize=11)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{label}\n(n={count})" for label, count in zip(viol_labels, counts)], fontsize=10)
    ax.set_title("Generation metrics by fiber-violation strength", fontsize=13, pad=10)
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.7)

    fig.suptitle("Aggregate VAR Generation Diagnostics Across All Tokens", fontsize=18, y=0.985)
    fig.text(
        0.02,
        0.018,
        "Dimension trends use all 2048 teacher-forced visual tokens. Fiber-violation bins separate non-violating tokens "
        "from the 66 corrected slope-increase violations split into tertiles by irregularity.",
        fontsize=11,
        color="#333333",
    )
    fig.subplots_adjust(left=0.055, right=0.945, top=0.790, bottom=0.205, wspace=0.34)
    save_figure(fig, out_path, dpi=180)
    plt.close(fig)


def _load_fiber_results(run_dir: Path, epoch: int) -> list[dict]:
    path = run_dir / "checkpoints" / f"fiber_epoch_{epoch:03d}.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    return data


def _to_token_stats(fiber_results: Iterable[dict], alpha: float) -> tuple[np.ndarray, ...]:
    dims: list[float] = []
    p_change: list[float] = []
    p_violation: list[float] = []
    irregularity: list[float] = []
    rejected: list[bool] = []
    for result in fiber_results:
        dim_values = result.get("dimensions") if isinstance(result, dict) else None
        first_dim = float(dim_values[0]) if dim_values else float("nan")
        change_p = min_change_pvalue(result)
        violation_p = min_fiber_violation_pvalue(result)
        if math.isfinite(violation_p):
            irr = float(-math.log10(violation_p + 1e-12))
        else:
            irr = 0.0
        dims.append(first_dim)
        p_change.append(change_p)
        p_violation.append(violation_p)
        irregularity.append(irr)
        rejected.append(bool(math.isfinite(violation_p) and violation_p < alpha))
    return (
        np.asarray(dims, dtype=np.float64),
        np.asarray(p_change, dtype=np.float64),
        np.asarray(p_violation, dtype=np.float64),
        np.asarray(irregularity, dtype=np.float64),
        np.asarray(rejected, dtype=bool),
    )


def _make_token_records(
    *,
    grid_size: int,
    entropy: np.ndarray,
    entropy_norm: np.ndarray,
    nll: np.ndarray,
    top1_prob: np.ndarray,
    top2_margin: np.ndarray,
    dimension: np.ndarray,
    p_change: np.ndarray,
    p_violation: np.ndarray,
    irregularity: np.ndarray,
    rejected: np.ndarray,
) -> list[dict[str, float | int | bool]]:
    records = []
    for token_idx in range(entropy.shape[0]):
        image_id = token_idx // (grid_size * grid_size)
        patch_id = token_idx % (grid_size * grid_size)
        records.append(
            {
                "token_index": int(token_idx),
                "image_id": int(image_id),
                "patch_id": int(patch_id),
                "row": int(patch_id // grid_size),
                "col": int(patch_id % grid_size),
                "entropy": float(entropy[token_idx]),
                "entropy_norm": float(entropy_norm[token_idx]),
                "nll": float(nll[token_idx]),
                "top1_prob": float(top1_prob[token_idx]),
                "top2_margin": float(top2_margin[token_idx]),
                "dimension": float(dimension[token_idx]),
                "p_change": float(p_change[token_idx]) if math.isfinite(p_change[token_idx]) else None,
                "p_violation": float(p_violation[token_idx]) if math.isfinite(p_violation[token_idx]) else None,
                "irregularity": float(irregularity[token_idx]),
                "fiber_violation_reject": bool(rejected[token_idx]),
            }
        )
    return records


def _arrays_from_records(records: list[dict]) -> dict[str, np.ndarray]:
    keys = [
        "entropy",
        "entropy_norm",
        "nll",
        "top1_prob",
        "top2_margin",
        "dimension",
        "p_change",
        "p_violation",
        "irregularity",
    ]
    arrays = {}
    for key in keys:
        values = []
        for record in records:
            value = record.get(key)
            values.append(float(value) if value is not None else float("nan"))
        arrays[key] = np.asarray(values, dtype=np.float64)
    arrays["rejected"] = np.asarray([bool(record.get("fiber_violation_reject", False)) for record in records])
    return arrays


def _collect_display_images(
    *,
    test_loader,
    num_images: int,
    dataset: str,
    image_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for image_idx, batch in enumerate(test_loader):
        if image_idx >= num_images:
            break
        imgs = batch[0].to(device, non_blocking=True)
        imgs01 = denormalize_images(imgs, dataset)
        if imgs01.shape[-2:] != (image_size, image_size):
            imgs01 = F.interpolate(imgs01, size=(image_size, image_size), mode="bilinear", align_corners=False)
        image = imgs01.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
        images.append(np.clip(image, 0.0, 1.0))
    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "runs/local/coco_var_d30_sparse_fiber/20260506_232532",
        help="Existing VAR fiber run directory.",
    )
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--subset-test", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patches-per-row", type=int, default=8)
    parser.add_argument(
        "--reuse-json",
        action="store_true",
        help="Reuse existing per-token JSON and only refresh summaries/figures.",
    )
    parser.add_argument("--wandb", action="store_true", help="Resume/log the new figures to W&B.")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    cfg_path = run_dir / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(cfg_path) if cfg_path.exists() else OmegaConf.create({})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    fiber_cfg = cfg.get("fiber", {})
    wandb_cfg = cfg.get("wandb", {})

    dataset = args.dataset or data_cfg.get("name", "COCO")
    data_root = args.data_root or data_cfg.get("root", str(REPO_ROOT.parent / "data"))
    img_size = int(args.img_size or data_cfg.get("img_size", 256))
    model_name = args.model_name or model_cfg.get("frozen_backbone_model", "var_d30")
    alpha = float(args.alpha if args.alpha is not None else fiber_cfg.get("alpha", 0.005))
    subset_test = int(args.subset_test or data_cfg.get("subset_test", 64))

    fiber_results = _load_fiber_results(run_dir, args.epoch)
    grid_tokens = int(args.grid_size * args.grid_size)
    if len(fiber_results) % grid_tokens != 0:
        raise ValueError(
            f"Fiber result count {len(fiber_results)} is not divisible by grid size {args.grid_size}^2. "
            "This probe expects image-aligned token collection."
        )
    run_images = len(fiber_results) // grid_tokens
    num_images = min(run_images, int(args.max_images or run_images))
    num_tokens = num_images * grid_tokens
    fiber_results = fiber_results[:num_tokens]
    analysis_dir = run_dir / "checkpoints" / "fiber_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"epoch_{args.epoch:03d}_var_generation_polysemy"
    json_path = analysis_dir / f"{prefix}.json"

    device = torch.device(args.device)
    _, test_loader, _, _, _, _ = create_data_loaders(
        dataset_name=dataset,
        root=data_root,
        img_size=img_size,
        batch_size_train=1,
        batch_size_test=1,
        num_workers=0,
        subset_train=1,
        subset_test=max(subset_test, num_images),
        device=device,
    )

    if args.reuse_json and json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        records = list(cached.get("tokens", []))[:num_tokens]
        if len(records) != num_tokens:
            raise ValueError(f"Cached JSON has {len(records)} tokens, expected {num_tokens}")
        arrays = _arrays_from_records(records)
        entropy = arrays["entropy"]
        entropy_norm = arrays["entropy_norm"]
        nll = arrays["nll"]
        top1_prob = arrays["top1_prob"]
        top2_margin = arrays["top2_margin"]
        dimension = arrays["dimension"]
        p_change = arrays["p_change"]
        p_violation = arrays["p_violation"]
        irregularity = arrays["irregularity"]
        rejected = arrays["rejected"]
        vocab_size = int(cached.get("summary", {}).get("vocab_size", 0))
        images = _collect_display_images(
            test_loader=test_loader,
            num_images=num_images,
            dataset=dataset,
            image_size=img_size,
            device=device,
        )
    else:
        model = VarAutoregressiveImageEncoder(model_name=model_name).to(device).eval()
        images: list[np.ndarray] = []
        entropy_chunks: list[np.ndarray] = []
        entropy_norm_chunks: list[np.ndarray] = []
        nll_chunks: list[np.ndarray] = []
        top1_chunks: list[np.ndarray] = []
        margin_chunks: list[np.ndarray] = []

        vocab_size = None
        with torch.no_grad():
            for image_idx, batch in enumerate(test_loader):
                if image_idx >= num_images:
                    break
                imgs = batch[0].to(device, non_blocking=True)
                pixel_values, imgs01 = model.prepare_images_for_features(imgs, dataset)
                pixel_values = pixel_values.to(device, non_blocking=True)
                pack = model.forward_generation_pack(pixel_values)
                logits = pack["logits"].float()
                targets = pack["targets"].long()
                vocab_size = int(logits.shape[-1])
                log_probs = logits.log_softmax(dim=-1)
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1)
                target_nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                top2 = probs.topk(k=2, dim=-1).values
                entropy_chunks.append(entropy.squeeze(0).detach().cpu().numpy())
                entropy_norm_chunks.append((entropy / math.log(vocab_size)).squeeze(0).detach().cpu().numpy())
                nll_chunks.append(target_nll.squeeze(0).detach().cpu().numpy())
                top1_chunks.append(top2[..., 0].squeeze(0).detach().cpu().numpy())
                margin_chunks.append((top2[..., 0] - top2[..., 1]).squeeze(0).detach().cpu().numpy())
                image = imgs01.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
                images.append(np.clip(image, 0.0, 1.0))

        entropy = np.concatenate(entropy_chunks, axis=0).astype(np.float64)
        entropy_norm = np.concatenate(entropy_norm_chunks, axis=0).astype(np.float64)
        nll = np.concatenate(nll_chunks, axis=0).astype(np.float64)
        top1_prob = np.concatenate(top1_chunks, axis=0).astype(np.float64)
        top2_margin = np.concatenate(margin_chunks, axis=0).astype(np.float64)
        dimension, p_change, p_violation, irregularity, rejected = _to_token_stats(fiber_results, alpha)

    if len(images) != num_images:
        raise RuntimeError(f"Collected {len(images)} images, expected {num_images}")

    high_irregular = _tail_mask(irregularity, 0.1, largest=True)
    low_irregular = _tail_mask(irregularity, 0.1, largest=False)
    high_dimension = _tail_mask(dimension, 0.1, largest=True)
    low_dimension = _tail_mask(dimension, 0.1, largest=False)
    quiet = ~rejected
    image_ids = np.arange(num_tokens, dtype=np.float64) // float(grid_tokens)
    patch_ids = np.arange(num_tokens, dtype=np.float64) % float(grid_tokens)
    patch_rows = np.floor(patch_ids / float(args.grid_size))
    patch_cols = patch_ids % float(args.grid_size)
    entropy_violation_diff, entropy_violation_p = _permutation_mean_diff_pvalue(
        entropy_norm[rejected], entropy_norm[quiet]
    )
    nll_violation_diff, nll_violation_p = _permutation_mean_diff_pvalue(nll[rejected], nll[quiet])
    entropy_high_low_diff, entropy_high_low_p = _permutation_mean_diff_pvalue(
        entropy_norm[high_irregular], entropy_norm[low_irregular]
    )
    nll_high_low_diff, nll_high_low_p = _permutation_mean_diff_pvalue(nll[high_irregular], nll[low_irregular])
    entropy_dim_high_low_diff, entropy_dim_high_low_p = _permutation_mean_diff_pvalue(
        entropy_norm[high_dimension], entropy_norm[low_dimension]
    )
    nll_dim_high_low_diff, nll_dim_high_low_p = _permutation_mean_diff_pvalue(
        nll[high_dimension], nll[low_dimension]
    )

    summary = {
        "run_dir": str(run_dir),
        "epoch": int(args.epoch),
        "dataset": str(dataset),
        "model_name": str(model_name),
        "num_images": int(num_images),
        "num_tokens": int(num_tokens),
        "grid_size": int(args.grid_size),
        "vocab_size": int(vocab_size or 0),
        "alpha": float(alpha),
        "fiber_violation_reject_count": int(rejected.sum()),
        "high_irregular_decile_size": int(high_irregular.sum()),
        "low_irregular_decile_size": int(low_irregular.sum()),
        "mean_entropy": _mean_or_nan(entropy),
        "mean_entropy_norm": _mean_or_nan(entropy_norm),
        "mean_nll": _mean_or_nan(nll),
        "mean_top1_prob": _mean_or_nan(top1_prob),
        "mean_top2_margin": _mean_or_nan(top2_margin),
        "mean_entropy_rejected": _mean_or_nan(entropy_norm[rejected]),
        "mean_entropy_nonrejected": _mean_or_nan(entropy_norm[quiet]),
        "mean_nll_rejected": _mean_or_nan(nll[rejected]),
        "mean_nll_nonrejected": _mean_or_nan(nll[quiet]),
        "mean_entropy_high_irregular_decile": _mean_or_nan(entropy_norm[high_irregular]),
        "mean_entropy_low_irregular_decile": _mean_or_nan(entropy_norm[low_irregular]),
        "mean_nll_high_irregular_decile": _mean_or_nan(nll[high_irregular]),
        "mean_nll_low_irregular_decile": _mean_or_nan(nll[low_irregular]),
        "mean_entropy_high_dimension_decile": _mean_or_nan(entropy_norm[high_dimension]),
        "mean_entropy_low_dimension_decile": _mean_or_nan(entropy_norm[low_dimension]),
        "mean_nll_high_dimension_decile": _mean_or_nan(nll[high_dimension]),
        "mean_nll_low_dimension_decile": _mean_or_nan(nll[low_dimension]),
        "diff_entropy_rejected_minus_nonrejected": entropy_violation_diff,
        "perm_p_entropy_rejected_vs_nonrejected": entropy_violation_p,
        "cohen_d_entropy_rejected_vs_nonrejected": _cohen_d(entropy_norm[rejected], entropy_norm[quiet]),
        "diff_nll_rejected_minus_nonrejected": nll_violation_diff,
        "perm_p_nll_rejected_vs_nonrejected": nll_violation_p,
        "cohen_d_nll_rejected_vs_nonrejected": _cohen_d(nll[rejected], nll[quiet]),
        "diff_entropy_high_minus_low_irregular_decile": entropy_high_low_diff,
        "perm_p_entropy_high_vs_low_irregular_decile": entropy_high_low_p,
        "cohen_d_entropy_high_vs_low_irregular_decile": _cohen_d(
            entropy_norm[high_irregular], entropy_norm[low_irregular]
        ),
        "diff_nll_high_minus_low_irregular_decile": nll_high_low_diff,
        "perm_p_nll_high_vs_low_irregular_decile": nll_high_low_p,
        "cohen_d_nll_high_vs_low_irregular_decile": _cohen_d(nll[high_irregular], nll[low_irregular]),
        "diff_entropy_high_minus_low_dimension_decile": entropy_dim_high_low_diff,
        "perm_p_entropy_high_vs_low_dimension_decile": entropy_dim_high_low_p,
        "cohen_d_entropy_high_vs_low_dimension_decile": _cohen_d(
            entropy_norm[high_dimension], entropy_norm[low_dimension]
        ),
        "diff_nll_high_minus_low_dimension_decile": nll_dim_high_low_diff,
        "perm_p_nll_high_vs_low_dimension_decile": nll_dim_high_low_p,
        "cohen_d_nll_high_vs_low_dimension_decile": _cohen_d(nll[high_dimension], nll[low_dimension]),
        "corr_irregularity_entropy_pearson": _finite_corr(irregularity, entropy_norm),
        "corr_irregularity_entropy_spearman": _finite_corr(irregularity, entropy_norm, spearman=True),
        "corr_irregularity_nll_pearson": _finite_corr(irregularity, nll),
        "corr_irregularity_nll_spearman": _finite_corr(irregularity, nll, spearman=True),
        "corr_irregularity_top1_prob_pearson": _finite_corr(irregularity, top1_prob),
        "corr_irregularity_top1_prob_spearman": _finite_corr(irregularity, top1_prob, spearman=True),
        "corr_dimension_entropy_spearman": _finite_corr(dimension, entropy_norm, spearman=True),
        "corr_dimension_nll_spearman": _finite_corr(dimension, nll, spearman=True),
        "partial_corr_irregularity_entropy_given_dimension_position_spearman": _partial_spearman(
            irregularity, entropy_norm, [dimension, image_ids, patch_rows, patch_cols]
        ),
        "partial_corr_irregularity_nll_given_dimension_position_spearman": _partial_spearman(
            irregularity, nll, [dimension, image_ids, patch_rows, patch_cols]
        ),
        "partial_corr_dimension_entropy_given_irregularity_position_spearman": _partial_spearman(
            dimension, entropy_norm, [irregularity, image_ids, patch_rows, patch_cols]
        ),
        "partial_corr_dimension_nll_given_irregularity_position_spearman": _partial_spearman(
            dimension, nll, [irregularity, image_ids, patch_rows, patch_cols]
        ),
    }

    entropy_maps = entropy_norm.reshape(num_images, args.grid_size, args.grid_size)
    nll_maps = nll.reshape(num_images, args.grid_size, args.grid_size)
    irregularity_maps = irregularity.reshape(num_images, args.grid_size, args.grid_size)

    entropy_path = analysis_dir / f"{prefix}_entropy_heatmaps.png"
    nll_path = analysis_dir / f"{prefix}_nll_heatmaps.png"
    scatter_path = analysis_dir / f"{prefix}_scatter.png"
    gallery_path = analysis_dir / f"{prefix}_patch_gallery.png"
    aggregate_path = analysis_dir / f"{prefix}_aggregate_bins.png"

    _plot_overlay_grid(
        images=images,
        maps=entropy_maps,
        out_path=entropy_path,
        title="VAR Generation Ambiguity Projected Back to Image Patches",
        colorbar_label="normalized next-token entropy",
        cmap="magma",
        footer="Brighter patches are locations where VAR assigns mass to more possible VQ tokens under teacher forcing.",
        max_images=num_images,
    )
    _plot_overlay_grid(
        images=images,
        maps=nll_maps,
        out_path=nll_path,
        title="VAR True-Code Surprise Projected Back to Image Patches",
        colorbar_label="negative log likelihood",
        cmap="viridis",
        footer="Brighter patches are observed VQ tokens that VAR found less probable under the prefix/context.",
        max_images=num_images,
    )
    _plot_scatter(
        entropy_norm=entropy_norm,
        nll=nll,
        top1_prob=top1_prob,
        irregularity=irregularity,
        dimension=dimension,
        out_path=scatter_path,
        summary=summary,
    )
    _plot_patch_gallery(
        images=images,
        grid_size=args.grid_size,
        entropy_norm=entropy_norm,
        irregularity=irregularity,
        dimension=dimension,
        nll=nll,
        out_path=gallery_path,
        patches_per_row=int(args.patches_per_row),
    )
    _plot_aggregate_bins(
        dimension=dimension,
        irregularity=irregularity,
        rejected=rejected,
        entropy_norm=entropy_norm,
        nll=nll,
        out_path=aggregate_path,
    )

    payload = {
        "summary": summary,
        "figures": {
            "entropy_heatmaps": str(entropy_path),
            "nll_heatmaps": str(nll_path),
            "scatter": str(scatter_path),
            "patch_gallery": str(gallery_path),
            "aggregate_bins": str(aggregate_path),
        },
        "tokens": _make_token_records(
            grid_size=args.grid_size,
            entropy=entropy,
            entropy_norm=entropy_norm,
            nll=nll,
            top1_prob=top1_prob,
            top2_margin=top2_margin,
            dimension=dimension,
            p_change=p_change,
            p_violation=p_violation,
            irregularity=irregularity,
            rejected=rejected,
        ),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if args.wandb:
        import wandb

        project = args.wandb_project or wandb_cfg.get("project", "stratified-manifold-learning")
        run_id = args.wandb_run_id
        if run_id is None:
            wandb_dir = run_dir / "wandb" / "wandb"
            matches = sorted(wandb_dir.glob("run-*-*")) if wandb_dir.exists() else []
            if matches:
                run_id = matches[-1].name.rsplit("-", 1)[-1]
        run = wandb.init(project=project, id=run_id, resume="allow", dir=str(run_dir / "wandb"))
        wandb.log(
            {
                "generation_polysemy/summary": summary,
                "generation_polysemy/entropy_heatmaps": wandb.Image(str(entropy_path)),
                "generation_polysemy/nll_heatmaps": wandb.Image(str(nll_path)),
                "generation_polysemy/scatter": wandb.Image(str(scatter_path)),
                "generation_polysemy/patch_gallery": wandb.Image(str(gallery_path)),
                "generation_polysemy/aggregate_bins": wandb.Image(str(aggregate_path)),
            }
        )
        run.finish()

    print(json.dumps({"summary": summary, "json": str(json_path), "figures": payload["figures"]}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
