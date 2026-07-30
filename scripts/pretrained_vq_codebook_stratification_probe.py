"""Paper-style stratification probe for pretrained VQ codebooks.

This script treats visual token IDs as the points of interest.  It loads a
matched pretrained VQ tokenizer, extracts the codebook embeddings, and runs the
radius-volume sliding Welch tests used by the original token-embedding
stratification paper on the codebook geometry itself.

The output labels singular visual token IDs.  Downstream AR/polysemy probes can
then ask whether contexts involving those token IDs have flatter or more
ambiguous next-token branches.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

try:
    from scipy.special import stdtr as scipy_stdtr
except ImportError:  # pragma: no cover
    scipy_stdtr = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from fiber.figure_io import save_figure  # noqa: E402
from pretrained_vq_ar_pipeline import (  # noqa: E402
    LLAMAGEN_PROFILES,
    llamagen_import_context,
    load_weight_payload,
    resolve_llamagen_repo,
)


@dataclass(frozen=True)
class CodebookPayload:
    embeddings: np.ndarray
    source_key: str
    normalized: bool
    quantizer_l2_norm: bool
    checkpoint_path: str
    repo_path: str


@dataclass(frozen=True)
class StratificationBand:
    name: str
    vol_min: int
    vol_max: int
    volumes: np.ndarray
    manifold_pvalue: np.ndarray
    manifold_index: np.ndarray
    manifold_delta: np.ndarray
    fiber_pvalue: np.ndarray
    fiber_index: np.ndarray
    fiber_delta: np.ndarray
    dimension: np.ndarray
    dimvec: np.ndarray
    manifold_adjusted_pvalue: np.ndarray
    manifold_rejected: np.ndarray
    fiber_adjusted_pvalue: np.ndarray
    fiber_rejected: np.ndarray


def mean_or_nan(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def min_or_nan(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.min()) if arr.size else float("nan")


def finite_minimum(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.asarray([], dtype=np.float64)
    stacked = np.vstack([np.asarray(x, dtype=np.float64) for x in arrays])
    finite = np.isfinite(stacked)
    safe = np.where(finite, stacked, np.inf)
    out = np.min(safe, axis=0)
    out[~np.any(finite, axis=0)] = np.nan
    return out


def resolve_distance_device(text: str) -> torch.device:
    value = str(text).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def holm_bonferroni(pvalues: np.ndarray, *, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Return Holm-Bonferroni adjusted p-values and rejection decisions."""
    arr = np.asarray(pvalues, dtype=np.float64)
    adjusted = np.full(arr.shape, np.nan, dtype=np.float64)
    rejected = np.zeros(arr.shape, dtype=bool)
    finite_idx = np.flatnonzero(np.isfinite(arr))
    if finite_idx.size == 0:
        return adjusted, rejected

    order = finite_idx[np.argsort(arr[finite_idx])]
    sorted_p = arr[order]
    m = int(sorted_p.size)
    running_adj = 0.0
    stopped = False
    for rank, idx in enumerate(order):
        multiplier = m - rank
        corrected = min(1.0, float(sorted_p[rank]) * float(multiplier))
        running_adj = max(running_adj, corrected)
        adjusted[idx] = running_adj
        if not stopped and float(sorted_p[rank]) <= float(alpha) / float(multiplier):
            rejected[idx] = True
        else:
            stopped = True
    return adjusted, rejected


def _welch_ttest_pvalue_alt(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    *,
    alternative: str = "two-sided",
) -> float:
    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return 1.0

    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    term_a, term_b = var_a / a.size, var_b / b.size
    denom = term_a + term_b
    if not np.isfinite(denom) or denom <= 0.0:
        return 1.0

    t_stat = (mean_a - mean_b) / math.sqrt(float(denom))
    df_denom = 0.0
    if a.size > 1:
        df_denom += (term_a ** 2) / (a.size - 1)
    if b.size > 1:
        df_denom += (term_b ** 2) / (b.size - 1)
    if not np.isfinite(df_denom) or df_denom <= 0.0:
        return 1.0
    df = (denom ** 2) / df_denom
    if not np.isfinite(df) or df <= 0.0:
        return 1.0

    if scipy_stdtr is not None:
        cdf = float(scipy_stdtr(df, t_stat))
    else:
        cdf = 0.5 * (1.0 + math.erf(t_stat / math.sqrt(2.0)))
    cdf = max(0.0, min(1.0, cdf)) if np.isfinite(cdf) else 0.5

    if alternative == "two-sided":
        return float(max(0.0, min(1.0, 2.0 * min(cdf, 1.0 - cdf))))
    if alternative == "less":
        return float(cdf)
    if alternative == "greater":
        return float(1.0 - cdf)
    raise ValueError(f"unknown Welch alternative {alternative!r}")


def paper_style_sliding_welch_tests(
    radii: np.ndarray,
    volumes: np.ndarray,
    *,
    ws: int,
) -> dict[str, float | int | None]:
    """Scan the paper's left/right dimension windows and keep best p-values.

    The public estimator reports the first p-value below a threshold.  For a
    codebook-wide multiple-testing correction, this helper exposes the minimum
    p-value observed by the same window scan.
    """
    radii_safe = np.clip(np.asarray(radii, dtype=np.float64), 1e-12, None)
    volumes_safe = np.maximum(np.asarray(volumes, dtype=np.float64), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dimvec = np.gradient(np.log(volumes_safe)) / np.gradient(np.log(radii_safe))

    manifold_idx: int | None = None
    fiber_idx: int | None = None
    manifold_pvalue = 1.0
    fiber_pvalue = 1.0
    manifold_delta = 0.0
    fiber_delta = 0.0

    for w in range(2 * ws, dimvec.shape[0] - 2 * ws):
        left = dimvec[w - 2 * ws : w - ws]
        right = dimvec[w + ws : w + 2 * ws]
        left = left[np.logical_and(np.abs(left) > 1e-5, np.isfinite(left))]
        right = right[np.logical_and(np.abs(right) > 1e-5, np.isfinite(right))]
        if left.size < 2 or right.size < 2:
            continue
        delta = float(np.mean(right) - np.mean(left))

        p_manifold = _welch_ttest_pvalue_alt(left, right, alternative="two-sided")
        if p_manifold < manifold_pvalue:
            manifold_pvalue = float(p_manifold)
            manifold_idx = int(w)
            manifold_delta = delta

        if delta > 0.0:
            p_fiber = _welch_ttest_pvalue_alt(left, right, alternative="less")
            if p_fiber < fiber_pvalue:
                fiber_pvalue = float(p_fiber)
                fiber_idx = int(w)
                fiber_delta = delta

    return {
        "manifold_index": manifold_idx,
        "manifold_pvalue": float(manifold_pvalue),
        "manifold_delta": float(manifold_delta),
        "fiber_index": fiber_idx,
        "fiber_pvalue": float(fiber_pvalue),
        "fiber_delta": float(fiber_delta),
    }


def paper_style_geo_dimension(radii: np.ndarray, volumes: np.ndarray) -> float:
    radii = np.asarray(radii, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    valid = np.isfinite(radii) & np.isfinite(volumes) & (radii > 1e-10) & (volumes > 0)
    if int(valid.sum()) < 2:
        return float("nan")
    matrix = np.column_stack((np.ones(int(valid.sum()), dtype=np.float64), np.log(radii[valid])))
    coeffs, _residuals, _rank, _sv = np.linalg.lstsq(matrix, np.log(volumes[valid]), rcond=None)
    return float(coeffs[1])


def local_dimension_curves(sorted_distances: np.ndarray, *, vol_min: int, vol_max: int) -> tuple[np.ndarray, np.ndarray]:
    distances = np.asarray(sorted_distances, dtype=np.float64)
    upper = min(int(vol_max), int(distances.shape[1]))
    lower = max(1, int(vol_min))
    if upper <= lower + 2:
        return np.arange(lower, upper, dtype=np.float64), np.empty((distances.shape[0], 0), dtype=np.float64)
    volumes = np.arange(lower, upper, dtype=np.float64)
    radii = np.clip(distances[:, lower:upper], 1e-12, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        dimvec = np.gradient(np.log(volumes), axis=0)[None, :] / np.gradient(np.log(radii), axis=1)
    dimvec[~np.isfinite(dimvec)] = np.nan
    return volumes, dimvec


def scan_stratification_band(
    *,
    name: str,
    sorted_distances: np.ndarray,
    vol_min: int,
    vol_max: int,
    window_size: int,
    alpha: float,
) -> StratificationBand:
    distances = np.asarray(sorted_distances, dtype=np.float64)
    n_codes = int(distances.shape[0])
    lower = max(1, int(vol_min))
    upper = min(int(vol_max), int(distances.shape[1]))
    if upper <= lower:
        raise ValueError(f"{name} volume band is empty after clipping: {lower}..{upper}")

    volumes = np.arange(lower, upper, dtype=np.float64)
    radii = distances[:, lower:upper]
    manifold_pvalue = np.ones(n_codes, dtype=np.float64)
    manifold_index = np.full(n_codes, np.nan, dtype=np.float64)
    manifold_delta = np.zeros(n_codes, dtype=np.float64)
    fiber_pvalue = np.ones(n_codes, dtype=np.float64)
    fiber_index = np.full(n_codes, np.nan, dtype=np.float64)
    fiber_delta = np.zeros(n_codes, dtype=np.float64)
    dimension = np.full(n_codes, np.nan, dtype=np.float64)
    _curve_volumes, dimvec = local_dimension_curves(distances, vol_min=lower, vol_max=upper)

    for code_id in range(n_codes):
        tests = paper_style_sliding_welch_tests(radii[code_id], volumes, ws=int(window_size))
        manifold_pvalue[code_id] = float(tests["manifold_pvalue"])
        fiber_pvalue[code_id] = float(tests["fiber_pvalue"])
        manifold_delta[code_id] = float(tests["manifold_delta"])
        fiber_delta[code_id] = float(tests["fiber_delta"])
        if tests["manifold_index"] is not None:
            manifold_index[code_id] = float(tests["manifold_index"])
        if tests["fiber_index"] is not None:
            fiber_index[code_id] = float(tests["fiber_index"])
        dimension[code_id] = paper_style_geo_dimension(radii[code_id], volumes)

    manifold_adjusted, manifold_rejected = holm_bonferroni(manifold_pvalue, alpha=alpha)
    fiber_adjusted, fiber_rejected = holm_bonferroni(fiber_pvalue, alpha=alpha)
    return StratificationBand(
        name=name,
        vol_min=lower,
        vol_max=upper,
        volumes=volumes,
        manifold_pvalue=manifold_pvalue,
        manifold_index=manifold_index,
        manifold_delta=manifold_delta,
        fiber_pvalue=fiber_pvalue,
        fiber_index=fiber_index,
        fiber_delta=fiber_delta,
        dimension=dimension,
        dimvec=dimvec,
        manifold_adjusted_pvalue=manifold_adjusted,
        manifold_rejected=manifold_rejected,
        fiber_adjusted_pvalue=fiber_adjusted,
        fiber_rejected=fiber_rejected,
    )


def extract_codebook_embeddings(
    vq_model: Any,
    *,
    expected_size: int,
    expected_dim: int,
    geometry: str,
) -> tuple[np.ndarray, str, bool, bool]:
    quantizer = getattr(vq_model, "quantize", None)
    quantizer_l2_norm = bool(getattr(quantizer, "l2_norm", False))
    source_key = ""
    tensor: torch.Tensor | None = None
    if quantizer is not None and hasattr(quantizer, "embedding"):
        embedding = getattr(quantizer, "embedding")
        if hasattr(embedding, "weight"):
            tensor = embedding.weight.detach()
            source_key = "quantize.embedding.weight"

    if tensor is None:
        candidates: list[tuple[str, torch.Tensor]] = []
        for key, value in vq_model.state_dict().items():
            if isinstance(value, torch.Tensor) and tuple(value.shape) == (int(expected_size), int(expected_dim)):
                candidates.append((key, value.detach()))
        if not candidates:
            raise ValueError(
                f"Could not find a codebook tensor with shape {(int(expected_size), int(expected_dim))}"
            )
        candidates.sort(key=lambda item: (0 if "quant" in item[0] or "embed" in item[0] else 1, item[0]))
        source_key, tensor = candidates[0]

    if tuple(tensor.shape) != (int(expected_size), int(expected_dim)):
        raise ValueError(f"{source_key} has shape {tuple(tensor.shape)}, expected {(expected_size, expected_dim)}")

    normalized = False
    if geometry == "quantizer":
        normalized = quantizer_l2_norm
    elif geometry == "l2":
        normalized = True
    elif geometry == "raw":
        normalized = False
    else:
        raise ValueError(f"unknown codebook geometry {geometry!r}")

    weight = tensor.float().cpu()
    if normalized:
        weight = F.normalize(weight, p=2, dim=-1)
    return weight.numpy().astype(np.float32), source_key, normalized, quantizer_l2_norm


def load_llamagen_codebook(
    *,
    profile_name: str,
    llamagen_repo: str,
    geometry: str,
) -> CodebookPayload:
    profile = LLAMAGEN_PROFILES[profile_name]
    repo_path = resolve_llamagen_repo(llamagen_repo or None)
    checkpoint_path = hf_hub_download(repo_id=profile["repo_id"], filename=profile["vq_file"])
    with llamagen_import_context(repo_path):
        from tokenizer.tokenizer_image.vq_model import VQ_models

        vq_model = VQ_models[profile["vq_model"]](
            codebook_size=int(profile["codebook_size"]),
            codebook_embed_dim=int(profile["codebook_embed_dim"]),
        )
        vq_model.load_state_dict(load_weight_payload(checkpoint_path), strict=True)
        vq_model.eval()
        embeddings, source_key, normalized, quantizer_l2_norm = extract_codebook_embeddings(
            vq_model,
            expected_size=int(profile["codebook_size"]),
            expected_dim=int(profile["codebook_embed_dim"]),
            geometry=geometry,
        )
    return CodebookPayload(
        embeddings=embeddings,
        source_key=source_key,
        normalized=normalized,
        quantizer_l2_norm=quantizer_l2_norm,
        checkpoint_path=str(checkpoint_path),
        repo_path=str(repo_path),
    )


def chunked_sorted_neighbor_distances(
    features: np.ndarray,
    *,
    max_neighbors: int,
    chunk_size: int,
    device: torch.device,
    include_self: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted local neighbor distances without building an n-by-n matrix."""
    x = torch.as_tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32, device=device)
    n = int(x.shape[0])
    k = min(max(1, int(max_neighbors)), n)
    distances = np.empty((n, k), dtype=np.float32)
    indices = np.empty((n, k), dtype=np.int64)
    x_norm = torch.sum(x * x, dim=1)
    x_t = x.T.contiguous()
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        chunk = x[start:end]
        d2 = x_norm[start:end, None] + x_norm[None, :] - 2.0 * (chunk @ x_t)
        d2 = torch.clamp(d2, min=0.0)
        if not include_self:
            rows = torch.arange(end - start, device=device)
            cols = torch.arange(start, end, device=device)
            d2[rows, cols] = float("inf")
        vals, idx = torch.topk(d2, k=k, dim=1, largest=False, sorted=True)
        distances[start:end] = torch.sqrt(vals).detach().cpu().numpy().astype(np.float32)
        indices[start:end] = idx.detach().cpu().numpy().astype(np.int64)
    return distances, indices


def pca_project(features: np.ndarray, *, dims: int = 2) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must be a 2D array")
    if int(dims) <= 0 or int(dims) >= int(x.shape[1]):
        return x[:, : int(dims)]
    centered = x - x.mean(axis=0, keepdims=True)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[: int(dims)].T


def neg_log10(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return -np.log10(np.clip(arr, 1e-300, 1.0))


def codebook_record_value(values: np.ndarray, idx: int) -> float | None:
    value = float(values[idx])
    return value if math.isfinite(value) else None


def build_codebook_records(
    *,
    small: StratificationBand,
    large: StratificationBand,
    singular_any: np.ndarray,
    singular_manifold_any: np.ndarray,
    singular_fiber_any: np.ndarray,
) -> list[dict[str, Any]]:
    n_codes = int(singular_any.size)
    best_raw = finite_minimum(
        [small.manifold_pvalue, small.fiber_pvalue, large.manifold_pvalue, large.fiber_pvalue]
    )
    best_adjusted = finite_minimum(
        [
            small.manifold_adjusted_pvalue,
            small.fiber_adjusted_pvalue,
            large.manifold_adjusted_pvalue,
            large.fiber_adjusted_pvalue,
        ]
    )
    records: list[dict[str, Any]] = []
    for code_id in range(n_codes):
        records.append(
            {
                "code_id": code_id,
                "singular_any": bool(singular_any[code_id]),
                "singular_manifold_any": bool(singular_manifold_any[code_id]),
                "singular_fiber_any": bool(singular_fiber_any[code_id]),
                "best_raw_pvalue": codebook_record_value(best_raw, code_id),
                "best_adjusted_pvalue": codebook_record_value(best_adjusted, code_id),
                "small_dimension": codebook_record_value(small.dimension, code_id),
                "large_dimension": codebook_record_value(large.dimension, code_id),
                "small_manifold_pvalue": codebook_record_value(small.manifold_pvalue, code_id),
                "small_manifold_adjusted_pvalue": codebook_record_value(small.manifold_adjusted_pvalue, code_id),
                "small_manifold_rejected": bool(small.manifold_rejected[code_id]),
                "small_manifold_index": codebook_record_value(small.manifold_index, code_id),
                "small_manifold_delta": codebook_record_value(small.manifold_delta, code_id),
                "small_fiber_pvalue": codebook_record_value(small.fiber_pvalue, code_id),
                "small_fiber_adjusted_pvalue": codebook_record_value(small.fiber_adjusted_pvalue, code_id),
                "small_fiber_rejected": bool(small.fiber_rejected[code_id]),
                "small_fiber_index": codebook_record_value(small.fiber_index, code_id),
                "small_fiber_delta": codebook_record_value(small.fiber_delta, code_id),
                "large_manifold_pvalue": codebook_record_value(large.manifold_pvalue, code_id),
                "large_manifold_adjusted_pvalue": codebook_record_value(large.manifold_adjusted_pvalue, code_id),
                "large_manifold_rejected": bool(large.manifold_rejected[code_id]),
                "large_manifold_index": codebook_record_value(large.manifold_index, code_id),
                "large_manifold_delta": codebook_record_value(large.manifold_delta, code_id),
                "large_fiber_pvalue": codebook_record_value(large.fiber_pvalue, code_id),
                "large_fiber_adjusted_pvalue": codebook_record_value(large.fiber_adjusted_pvalue, code_id),
                "large_fiber_rejected": bool(large.fiber_rejected[code_id]),
                "large_fiber_index": codebook_record_value(large.fiber_index, code_id),
                "large_fiber_delta": codebook_record_value(large.fiber_delta, code_id),
            }
        )
    return records


def plot_codebook_pca(
    *,
    pca: np.ndarray,
    singular: np.ndarray,
    out_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    pca = np.asarray(pca, dtype=np.float64)
    singular = np.asarray(singular, dtype=bool)
    ax.scatter(pca[~singular, 0], pca[~singular, 1], s=8, alpha=0.32, color="#4c78a8", label=f"regular n={(~singular).sum()}")
    if int(singular.sum()):
        ax.scatter(pca[singular, 0], pca[singular, 1], s=16, alpha=0.82, color="#f58518", label=f"singular n={singular.sum()}")
    ax.set_title("VQ codebook PCA with paper-style singular token IDs")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = save_figure(fig, out_path, dpi=190)
    plt.close(fig)
    return path


def plot_pvalue_histogram(
    *,
    best_adjusted_pvalue: np.ndarray,
    singular: np.ndarray,
    alpha: float,
    out_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    singular = np.asarray(singular, dtype=bool)
    values = neg_log10(best_adjusted_pvalue)
    finite = np.isfinite(values)
    if finite.any():
        hi = float(np.nanquantile(values[finite], 0.995))
        hi = max(3.0, hi)
        bins = np.linspace(0.0, hi, 42)
        regular_values = values[finite & ~singular]
        singular_values = values[finite & singular]
        regular_weights = np.full(regular_values.shape, 1.0 / max(1, regular_values.size), dtype=np.float64)
        singular_weights = np.full(singular_values.shape, 1.0 / max(1, singular_values.size), dtype=np.float64)
        ax.hist(
            regular_values,
            bins=bins,
            weights=regular_weights,
            alpha=0.62,
            color="#4c78a8",
            label=f"regular n={regular_values.size}",
        )
        ax.hist(
            singular_values,
            bins=bins,
            weights=singular_weights,
            alpha=0.72,
            color="#f58518",
            label=f"singular n={singular_values.size}",
        )
        ax.axvline(-math.log10(float(alpha)), color="#333333", linestyle="--", linewidth=1.5, label=f"alpha={alpha:g}")
    else:
        ax.text(0.5, 0.5, "no finite p-values", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Best adjusted codebook stratification p-values")
    ax.set_xlabel("-log10(adjusted p-value)")
    ax.set_ylabel("fraction within group")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = save_figure(fig, out_path, dpi=190)
    plt.close(fig)
    return path


def plot_dimension_curves(
    *,
    small: StratificationBand,
    large: StratificationBand,
    singular: np.ndarray,
    out_path: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    singular = np.asarray(singular, dtype=bool)
    for ax, band in zip(axes, [small, large]):
        dimvec = np.asarray(band.dimvec, dtype=np.float64)
        volumes = np.asarray(band.volumes, dtype=np.float64)
        if dimvec.size == 0 or not singular.any() or not (~singular).any():
            ax.text(0.5, 0.5, "insufficient groups", ha="center", va="center", transform=ax.transAxes)
        else:
            reg = dimvec[~singular]
            sing = dimvec[singular]
            reg_median = np.nanmedian(reg, axis=0)
            sing_median = np.nanmedian(sing, axis=0)
            ax.plot(volumes, reg_median, color="#4c78a8", linewidth=2, label="regular median")
            ax.plot(volumes, sing_median, color="#f58518", linewidth=2, label="singular median")
            ax.fill_between(
                volumes,
                np.nanpercentile(reg, 25, axis=0),
                np.nanpercentile(reg, 75, axis=0),
                color="#4c78a8",
                alpha=0.13,
                linewidth=0,
            )
            ax.fill_between(
                volumes,
                np.nanpercentile(sing, 25, axis=0),
                np.nanpercentile(sing, 75, axis=0),
                color="#f58518",
                alpha=0.17,
                linewidth=0,
            )
        ax.set_title(f"{band.name} radius band")
        ax.set_xlabel("neighbor volume")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("local dimension estimate")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    path = save_figure(fig, out_path, dpi=190)
    plt.close(fig)
    return path


def plot_radius_volume_examples(
    *,
    sorted_distances: np.ndarray,
    records: list[dict[str, Any]],
    singular: np.ndarray,
    vol_min: int,
    vol_max: int,
    out_path: Path,
    max_examples: int = 8,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    lower = max(1, int(vol_min))
    upper = min(int(vol_max), int(sorted_distances.shape[1]))
    volumes = np.arange(lower, upper, dtype=np.float64)
    best = np.asarray(
        [record["best_adjusted_pvalue"] if record["best_adjusted_pvalue"] is not None else 1.0 for record in records],
        dtype=np.float64,
    )
    singular_ids = np.flatnonzero(singular)
    regular_ids = np.flatnonzero(~singular)
    if singular_ids.size:
        order = singular_ids[np.argsort(best[singular_ids])[: int(max_examples)]]
        for code_id in order:
            radii = np.clip(sorted_distances[code_id, lower:upper], 1e-12, None)
            ax.plot(np.log(volumes), np.log(radii), color="#f58518", alpha=0.52, linewidth=1.4)
    if regular_ids.size:
        regular_scores = best[regular_ids]
        keep = regular_ids[np.argsort(np.abs(regular_scores - np.nanmedian(regular_scores)))[: int(max_examples)]]
        for code_id in keep:
            radii = np.clip(sorted_distances[code_id, lower:upper], 1e-12, None)
            ax.plot(np.log(volumes), np.log(radii), color="#4c78a8", alpha=0.35, linewidth=1.1)
    ax.set_title("Example codebook radius-volume curves")
    ax.set_xlabel("log(neighbor volume)")
    ax.set_ylabel("log(radius)")
    ax.plot([], [], color="#f58518", label="top singular")
    ax.plot([], [], color="#4c78a8", label="typical regular")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = save_figure(fig, out_path, dpi=190)
    plt.close(fig)
    return path


def summarize_band(band: StratificationBand) -> dict[str, Any]:
    return {
        "name": band.name,
        "vol_min": int(band.vol_min),
        "vol_max": int(band.vol_max),
        "manifold_rejected_count": int(band.manifold_rejected.sum()),
        "fiber_rejected_count": int(band.fiber_rejected.sum()),
        "min_manifold_adjusted_pvalue": min_or_nan(band.manifold_adjusted_pvalue),
        "min_fiber_adjusted_pvalue": min_or_nan(band.fiber_adjusted_pvalue),
        "mean_dimension": mean_or_nan(band.dimension),
        "median_dimension": float(np.nanmedian(band.dimension)) if np.isfinite(band.dimension).any() else float("nan"),
        "mean_positive_fiber_delta": mean_or_nan(band.fiber_delta[band.fiber_delta > 0.0]),
    }


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = LLAMAGEN_PROFILES[args.profile]
    payload = load_llamagen_codebook(
        profile_name=args.profile,
        llamagen_repo=args.llamagen_repo,
        geometry=args.codebook_geometry,
    )
    features = payload.embeddings
    max_vol = max(int(args.small_vol_max), int(args.large_vol_max))
    distance_device = resolve_distance_device(args.distance_device)
    sorted_distances, sorted_indices = chunked_sorted_neighbor_distances(
        features,
        max_neighbors=max_vol,
        chunk_size=int(args.knn_chunk_size),
        device=distance_device,
        include_self=True,
    )
    np.savez_compressed(
        out_dir / "vq_codebook_neighbors.npz",
        distances=sorted_distances,
        indices=sorted_indices,
    )

    small = scan_stratification_band(
        name="small",
        sorted_distances=sorted_distances,
        vol_min=int(args.small_vol_min),
        vol_max=int(args.small_vol_max),
        window_size=int(args.window_size),
        alpha=float(args.alpha),
    )
    large = scan_stratification_band(
        name="large",
        sorted_distances=sorted_distances,
        vol_min=int(args.large_vol_min),
        vol_max=int(args.large_vol_max),
        window_size=int(args.window_size),
        alpha=float(args.alpha),
    )
    singular_manifold_any = small.manifold_rejected | large.manifold_rejected
    singular_fiber_any = small.fiber_rejected | large.fiber_rejected
    singular_any = singular_manifold_any | singular_fiber_any
    records = build_codebook_records(
        small=small,
        large=large,
        singular_any=singular_any,
        singular_manifold_any=singular_manifold_any,
        singular_fiber_any=singular_fiber_any,
    )
    best_adjusted = finite_minimum(
        [
            small.manifold_adjusted_pvalue,
            small.fiber_adjusted_pvalue,
            large.manifold_adjusted_pvalue,
            large.fiber_adjusted_pvalue,
        ]
    )
    pca = pca_project(features, dims=2)
    np.save(out_dir / "vq_codebook_pca.npy", pca.astype(np.float32))

    figures = {
        "codebook_pca": str(plot_codebook_pca(
            pca=pca,
            singular=singular_any,
            out_path=out_dir / "vq_codebook_pca_singular.png",
        )),
        "adjusted_pvalue_hist": str(plot_pvalue_histogram(
            best_adjusted_pvalue=best_adjusted,
            singular=singular_any,
            alpha=float(args.alpha),
            out_path=out_dir / "vq_codebook_adjusted_pvalue_hist.png",
        )),
        "dimension_curves": str(plot_dimension_curves(
            small=small,
            large=large,
            singular=singular_any,
            out_path=out_dir / "vq_codebook_dimension_curves.png",
        )),
        "radius_volume_examples": str(plot_radius_volume_examples(
            sorted_distances=sorted_distances,
            records=records,
            singular=singular_any,
            vol_min=int(args.large_vol_min),
            vol_max=int(args.large_vol_max),
            out_path=out_dir / "vq_codebook_radius_volume_examples.png",
        )),
    }

    singular_codes = {
        "singular_any": [int(x) for x in np.flatnonzero(singular_any).tolist()],
        "singular_manifold_any": [int(x) for x in np.flatnonzero(singular_manifold_any).tolist()],
        "singular_fiber_any": [int(x) for x in np.flatnonzero(singular_fiber_any).tolist()],
        "small_manifold": [int(x) for x in np.flatnonzero(small.manifold_rejected).tolist()],
        "small_fiber": [int(x) for x in np.flatnonzero(small.fiber_rejected).tolist()],
        "large_manifold": [int(x) for x in np.flatnonzero(large.manifold_rejected).tolist()],
        "large_fiber": [int(x) for x in np.flatnonzero(large.fiber_rejected).tolist()],
    }
    config = {
        "profile": args.profile,
        "profile_metadata": profile,
        "codebook_geometry": args.codebook_geometry,
        "distance_device": str(distance_device),
        "knn_chunk_size": int(args.knn_chunk_size),
        "small_vol_min": int(args.small_vol_min),
        "small_vol_max": int(args.small_vol_max),
        "large_vol_min": int(args.large_vol_min),
        "large_vol_max": int(args.large_vol_max),
        "window_size": int(args.window_size),
        "alpha": float(args.alpha),
    }
    summary: dict[str, Any] = {
        "analysis": "vq_codebook_paper_style_stratification",
        "config": config,
        "codebook": {
            "shape": [int(features.shape[0]), int(features.shape[1])],
            "source_key": payload.source_key,
            "normalized": bool(payload.normalized),
            "quantizer_l2_norm": bool(payload.quantizer_l2_norm),
            "checkpoint_path": payload.checkpoint_path,
            "repo_path": payload.repo_path,
            "mean_norm": float(np.linalg.norm(features, axis=1).mean()),
            "std_norm": float(np.linalg.norm(features, axis=1).std()),
        },
        "small_band": summarize_band(small),
        "large_band": summarize_band(large),
        "singular_counts": {
            "singular_any": int(singular_any.sum()),
            "singular_any_fraction": float(singular_any.mean()),
            "singular_manifold_any": int(singular_manifold_any.sum()),
            "singular_fiber_any": int(singular_fiber_any.sum()),
            "small_manifold": int(small.manifold_rejected.sum()),
            "small_fiber": int(small.fiber_rejected.sum()),
            "large_manifold": int(large.manifold_rejected.sum()),
            "large_fiber": int(large.fiber_rejected.sum()),
        },
        "best_adjusted_pvalue": {
            "min": min_or_nan(best_adjusted),
            "median": float(np.nanmedian(best_adjusted)) if np.isfinite(best_adjusted).any() else float("nan"),
        },
        "figures": figures,
    }

    summary_path = write_json(out_dir / "vq_codebook_stratification_summary.json", summary)
    records_path = write_json(out_dir / "vq_codebook_records.json", records)
    singular_path = write_json(out_dir / "vq_codebook_singular_codes.json", {"config": config, **singular_codes})
    summary["artifacts"] = {
        "summary": str(summary_path),
        "records": str(records_path),
        "singular_codes": str(singular_path),
        "neighbors": str(out_dir / "vq_codebook_neighbors.npz"),
        "pca": str(out_dir / "vq_codebook_pca.npy"),
    }
    write_json(summary_path, summary)

    if args.wandb:
        import wandb

        tags = [part.strip() for part in str(args.wandb_tags).split(",") if part.strip()]
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=tags,
            config=config,
        )
        wandb.log(
            {
                "codebook/singular_any_count": int(singular_any.sum()),
                "codebook/singular_any_fraction": float(singular_any.mean()),
                "codebook/singular_manifold_any_count": int(singular_manifold_any.sum()),
                "codebook/singular_fiber_any_count": int(singular_fiber_any.sum()),
                "codebook/small_manifold_rejected_count": int(small.manifold_rejected.sum()),
                "codebook/small_fiber_rejected_count": int(small.fiber_rejected.sum()),
                "codebook/large_manifold_rejected_count": int(large.manifold_rejected.sum()),
                "codebook/large_fiber_rejected_count": int(large.fiber_rejected.sum()),
                "codebook/min_best_adjusted_pvalue": min_or_nan(best_adjusted),
                "codebook/mean_small_dimension": mean_or_nan(small.dimension),
                "codebook/mean_large_dimension": mean_or_nan(large.dimension),
                **{f"codebook/{key}": wandb.Image(path) for key, path in figures.items()},
            }
        )
        artifact = wandb.Artifact(f"{args.wandb_name}_outputs", type="analysis")
        for path in [summary_path, records_path, singular_path, out_dir / "vq_codebook_neighbors.npz", out_dir / "vq_codebook_pca.npy"]:
            artifact.add_file(str(path))
        for path in figures.values():
            artifact.add_file(path)
        run.log_artifact(artifact)
        run.finish()

    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(LLAMAGEN_PROFILES), default="c2i-B-256")
    parser.add_argument("--llamagen-repo", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--codebook-geometry", choices=["quantizer", "l2", "raw"], default="quantizer")
    parser.add_argument("--distance-device", default="auto")
    parser.add_argument("--knn-chunk-size", type=int, default=512)
    parser.add_argument("--small-vol-min", type=int, default=10)
    parser.add_argument("--small-vol-max", type=int, default=50)
    parser.add_argument("--large-vol-min", type=int, default=50)
    parser.add_argument("--large-vol-max", type=int, default=200)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="stratified-manifold-learning")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default="llamagen-c2i-B-256-codebook-stratification")
    parser.add_argument("--wandb-tags", default="vq-codebook,llamagen,stratification,singularity")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = run(args)
    printable = {key: value for key, value in summary.items() if key not in {"figures"}}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
