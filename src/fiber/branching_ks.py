"""Branch-flattening and robust sliced-KS diagnostics for vision tokens."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class KSTestResult:
    statistic: float
    pvalue: float
    n_a: int
    n_b: int


@dataclass(frozen=True)
class SlicedKSTestResult:
    median_statistic: float
    trimmed_mean_statistic: float
    max_statistic: float
    permutation_pvalue: float
    projection_statistics: np.ndarray
    null_statistics: np.ndarray


def standardize_features(features: np.ndarray) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("features must be a rank-2 array")
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True)
    return (arr - mean) / np.maximum(std, 1e-8)


def l2_normalize_rows(features: np.ndarray) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def fit_kmeans(
    features: np.ndarray,
    *,
    n_clusters: int,
    seed: int = 0,
    iters: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Small dependency-free k-means for probe prototypes."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("features must be non-empty and rank-2")
    k = max(1, min(int(n_clusters), int(x.shape[0])))
    rng = np.random.default_rng(seed)
    init = rng.choice(x.shape[0], size=k, replace=False)
    centers = x[init].copy()
    labels = np.zeros(x.shape[0], dtype=np.int64)
    for _ in range(max(1, int(iters))):
        d2 = (
            np.sum(x * x, axis=1, keepdims=True)
            + np.sum(centers * centers, axis=1)[None, :]
            - 2.0 * (x @ centers.T)
        )
        next_labels = np.argmin(d2, axis=1)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        for cid in range(k):
            mask = labels == cid
            if np.any(mask):
                centers[cid] = x[mask].mean(axis=0)
            else:
                centers[cid] = x[int(rng.integers(0, x.shape[0]))]
    return centers, labels


def branch_posteriors(
    features: np.ndarray,
    prototypes: np.ndarray,
    *,
    temperature: float = 0.08,
    top_k: int | None = None,
) -> np.ndarray:
    """Convert token-prototype similarities into a local branch posterior."""
    x = l2_normalize_rows(standardize_features(features))
    c = l2_normalize_rows(standardize_features(prototypes))
    logits = x @ c.T
    if top_k is not None and 0 < int(top_k) < logits.shape[1]:
        k = int(top_k)
        keep = np.argpartition(logits, -k, axis=1)[:, -k:]
        masked = np.full_like(logits, -1e9)
        rows = np.arange(logits.shape[0])[:, None]
        masked[rows, keep] = logits[rows, keep]
        logits = masked
    logits = logits / max(float(temperature), 1e-6)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)


def branch_metrics(posteriors: np.ndarray) -> dict[str, np.ndarray]:
    probs = np.asarray(posteriors, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] == 0:
        raise ValueError("posteriors must have shape (n, k)")
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    entropy_norm = entropy / max(math.log(probs.shape[1]), 1e-12)
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    top1 = sorted_probs[:, 0]
    top2 = sorted_probs[:, 1] if probs.shape[1] > 1 else np.zeros_like(top1)
    return {
        "branch_entropy": entropy,
        "branch_entropy_norm": entropy_norm,
        "branch_margin": top1 - top2,
        "branch_flatness": 1.0 - (top1 - top2),
        "effective_branches": np.exp(entropy),
        "top_branch_prob": top1,
        "top_branch": np.argmax(probs, axis=1).astype(np.int64),
    }


def ks_2samp(sample_a: Iterable[float], sample_b: Iterable[float]) -> KSTestResult:
    a = np.sort(np.asarray(list(sample_a), dtype=np.float64))
    b = np.sort(np.asarray(list(sample_b), dtype=np.float64))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return KSTestResult(float("nan"), float("nan"), int(a.size), int(b.size))
    values = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, values, side="right") / a.size
    cdf_b = np.searchsorted(b, values, side="right") / b.size
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    n_eff = (a.size * b.size) / max(1, a.size + b.size)
    lam = (math.sqrt(n_eff) + 0.12 + 0.11 / max(math.sqrt(n_eff), 1e-12)) * d
    terms = [(-1) ** (j - 1) * math.exp(-2.0 * (j * j) * lam * lam) for j in range(1, 101)]
    pvalue = float(max(0.0, min(1.0, 2.0 * sum(terms))))
    return KSTestResult(d, pvalue, int(a.size), int(b.size))


def _trimmed_mean(values: np.ndarray, trim_fraction: float = 0.10) -> float:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    trim = int(math.floor(arr.size * max(0.0, min(0.45, trim_fraction))))
    if trim > 0 and arr.size > 2 * trim:
        arr = arr[trim:-trim]
    return float(np.mean(arr))


def sliced_ks_test(
    features: np.ndarray,
    group_mask: np.ndarray,
    *,
    projections: int = 128,
    permutations: int = 200,
    seed: int = 0,
) -> SlicedKSTestResult:
    x = l2_normalize_rows(standardize_features(features))
    mask = np.asarray(group_mask, dtype=np.bool_)
    if x.shape[0] != mask.size:
        raise ValueError("group_mask must have one entry per feature row")
    if np.sum(mask) == 0 or np.sum(~mask) == 0:
        empty = np.asarray([], dtype=np.float64)
        return SlicedKSTestResult(float("nan"), float("nan"), float("nan"), float("nan"), empty, empty)

    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(max(1, int(projections)), x.shape[1]))
    dirs = l2_normalize_rows(dirs)
    stats = np.zeros(dirs.shape[0], dtype=np.float64)
    for idx, direction in enumerate(dirs):
        projected = x @ direction
        stats[idx] = ks_2samp(projected[mask], projected[~mask]).statistic

    median_stat = float(np.median(stats))
    null = np.zeros(max(0, int(permutations)), dtype=np.float64)
    if null.size:
        for idx in range(null.size):
            perm = rng.permutation(mask)
            perm_stats = [
                ks_2samp((x @ direction)[perm], (x @ direction)[~perm]).statistic
                for direction in dirs
            ]
            null[idx] = float(np.median(perm_stats))
        pvalue = float((1.0 + np.sum(null >= median_stat)) / (null.size + 1.0))
    else:
        pvalue = float("nan")

    return SlicedKSTestResult(
        median_statistic=median_stat,
        trimmed_mean_statistic=_trimmed_mean(stats),
        max_statistic=float(np.max(stats)),
        permutation_pvalue=pvalue,
        projection_statistics=stats,
        null_statistics=null,
    )


def fiber_singularity_scores(
    fiber_results: list[dict[str, Any]],
    *,
    alpha: float = 1e-2,
) -> dict[str, np.ndarray]:
    dims = np.full(len(fiber_results), np.nan, dtype=np.float64)
    pvals = np.full(len(fiber_results), np.nan, dtype=np.float64)
    irregularity = np.zeros(len(fiber_results), dtype=np.float64)
    rejected = np.zeros(len(fiber_results), dtype=np.bool_)
    for idx, result in enumerate(fiber_results):
        slopes = [float(v) for v in result.get("dimensions", []) if v is not None]
        changes = [float(v) for v in result.get("pvalues", []) if v is not None]
        if slopes:
            dims[idx] = slopes[0]
        candidates: list[float] = []
        for change_idx in range(min(len(changes), max(0, len(slopes) - 1))):
            if slopes[change_idx + 1] > slopes[change_idx] and math.isfinite(changes[change_idx]):
                candidates.append(changes[change_idx])
        if candidates:
            pvals[idx] = min(candidates)
            if pvals[idx] < float(alpha):
                rejected[idx] = True
                irregularity[idx] = -math.log10(max(float(pvals[idx]), 1e-300))
    return {
        "dimension": dims,
        "min_fiber_violation_pvalue": pvals,
        "irregularity": irregularity,
        "rejected": rejected,
    }


def quantile_group_mask(scores: np.ndarray, *, upper_quantile: float = 0.85) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape[0], dtype=np.bool_)
    threshold = np.nanquantile(arr[finite], float(upper_quantile))
    return finite & (arr >= threshold)


def make_preserving_augmentations(image: Image.Image, *, seed: int, count: int) -> list[Image.Image]:
    rng = np.random.default_rng(seed)
    base = image.convert("RGB")
    out = [base]
    for idx in range(max(0, int(count) - 1)):
        img = base.copy()
        if idx % 4 == 0:
            img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.82, 1.18)))
            img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.84, 1.22)))
        elif idx % 4 == 1:
            img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.35, 1.15))))
        elif idx % 4 == 2:
            img = ImageEnhance.Color(img).enhance(float(rng.uniform(0.72, 1.28)))
        else:
            arr = np.asarray(img).astype(np.float64)
            arr += rng.normal(scale=float(rng.uniform(2.0, 8.0)), size=arr.shape)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
        out.append(img)
    return out


def _patch_descriptor(patch: np.ndarray) -> np.ndarray:
    arr = np.asarray(patch, dtype=np.float64) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    mean = arr.mean(axis=(0, 1))
    std = arr.std(axis=(0, 1))
    q25 = np.quantile(arr, 0.25, axis=(0, 1))
    q75 = np.quantile(arr, 0.75, axis=(0, 1))
    gray = arr.mean(axis=2)
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy)
    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=4, range=(0.0, 1.0), density=False)
        hist_parts.append(hist / max(1, hist.sum()))
    return np.concatenate(
        [
            mean,
            std,
            q25,
            q75,
            np.asarray([gray.mean(), gray.std(), grad.mean(), grad.std(), np.percentile(grad, 90)]),
            *hist_parts,
        ]
    )


def extract_image_folder_patch_features(
    image_dir: str | Path,
    *,
    image_size: int = 224,
    grid: int = 8,
    augmentations: int = 6,
    seed: int = 0,
) -> dict[str, Any]:
    paths = sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    )
    if not paths:
        raise FileNotFoundError(f"No images found under {image_dir}")

    base_features: list[np.ndarray] = []
    all_features: list[np.ndarray] = []
    variant_groups: list[list[int]] = []
    images: list[Image.Image] = []
    image_ids: list[int] = []
    bboxes: list[tuple[int, int, int, int]] = []
    patch_indices: list[int] = []
    names: list[str] = []
    patch_px = max(1, int(image_size) // max(1, int(grid)))
    for image_idx, path in enumerate(paths):
        img = Image.open(path).convert("RGB").resize((int(image_size), int(image_size)))
        images.append(img)
        variants = make_preserving_augmentations(img, seed=seed + image_idx * 1009, count=augmentations)
        for row in range(int(grid)):
            for col in range(int(grid)):
                x0, y0 = col * patch_px, row * patch_px
                x1 = int(image_size) if col == int(grid) - 1 else (col + 1) * patch_px
                y1 = int(image_size) if row == int(grid) - 1 else (row + 1) * patch_px
                group: list[int] = []
                for variant_idx, variant in enumerate(variants):
                    patch = np.asarray(variant.crop((x0, y0, x1, y1)))
                    descriptor = _patch_descriptor(patch)
                    all_features.append(descriptor)
                    group.append(len(all_features) - 1)
                    if variant_idx == 0:
                        base_features.append(descriptor)
                variant_groups.append(group)
                image_ids.append(image_idx)
                bboxes.append((x0, y0, x1, y1))
                patch_indices.append(row * int(grid) + col)
                names.append(path.name)

    return {
        "features": np.vstack(base_features).astype(np.float64),
        "all_variant_features": np.vstack(all_features).astype(np.float64),
        "variant_groups": variant_groups,
        "images": images,
        "image_ids": np.asarray(image_ids, dtype=np.int64),
        "bboxes": np.asarray(bboxes, dtype=np.int64),
        "patch_indices": np.asarray(patch_indices, dtype=np.int64),
        "image_names": names,
        "grid": int(grid),
        "image_size": int(image_size),
    }


def augmentation_branch_instability(
    variant_posteriors: np.ndarray,
    variant_groups: list[list[int]],
) -> np.ndarray:
    probs = np.asarray(variant_posteriors, dtype=np.float64)
    instability = np.zeros(len(variant_groups), dtype=np.float64)
    for idx, group in enumerate(variant_groups):
        if not group:
            continue
        assignments = np.argmax(probs[np.asarray(group, dtype=np.int64)], axis=1)
        counts = np.bincount(assignments, minlength=probs.shape[1]).astype(np.float64)
        instability[idx] = 1.0 - float(np.max(counts) / max(1.0, counts.sum()))
    return instability
