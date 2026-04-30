"""Canonical geometric estimators for stratification and volume probes.

This is the single implementation of kNN volume-scaling dimension estimation,
stratification testing, and fiber bundle analysis. All other modules delegate here.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch

try:
    from scipy.special import stdtr as scipy_stdtr
except ImportError:  # pragma: no cover
    scipy_stdtr = None

__all__ = [
    "analyze_stratification",
    "analyze_stratification_from_sorted_distances",
    "normalize_volume_range",
    "run_fiber_bundle_test",
    "run_fiber_bundle_test_from_sorted_dists",
    "sorted_distance_matrix",
    "summarize_stratification",
    "summarize_stratifications",
]


# ---------------------------------------------------------------------------
# Low-level estimators
# ---------------------------------------------------------------------------

def _welch_ttest_pvalue(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Two-sided Welch t-test p-value without requiring scipy.stats."""
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

    t_stat = (mean_a - mean_b) / np.sqrt(denom)
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

    abs_t = abs(t_stat)
    if scipy_stdtr is not None:
        cdf = float(scipy_stdtr(df, abs_t))
    else:
        cdf = 0.5 * (1.0 + math.erf(abs_t / math.sqrt(2.0)))
    if not np.isfinite(cdf):
        return 1.0
    return float(max(0.0, min(1.0, 2.0 * (1.0 - cdf))))


def _geo_estimator(
    radii: np.ndarray,
    volumes: np.ndarray,
    npts: int,
    args: SimpleNamespace,
) -> tuple[float, float, float]:
    """Estimate local dimension via log-log linear regression of volume vs radius.

    Uses weighted least squares with weights proportional to volume index,
    which reduces bias from the high-variance small-radius regime.
    """
    log_r = np.log(radii)
    log_v = np.log(volumes)

    # WLS: weight by volume index (stabilises the small-radius regime where
    # variance of log(V) is highest due to low counts).
    w = np.sqrt(volumes.astype(np.float64))
    rstack = np.column_stack((np.ones_like(log_r), log_r))
    Xw = rstack * w[:, None]
    yw = log_v * w
    coeffs, residuals, _rank, _sv = np.linalg.lstsq(Xw, yw, rcond=None)

    scaling_coeff = float(np.exp(coeffs[0]) / npts)
    dimension = float(coeffs[1])
    residual = float(residuals[0]) if residuals.size else 0.0
    if args.miller:
        scaling_coeff *= float(np.exp(min(0.5 * residual ** 2, 50.0)))
    ricci = (
        float(np.mean(-residual * 6 * (dimension + 2) / radii ** 2))
        if args.ricci
        else 0.0
    )
    return scaling_coeff, dimension, ricci


def _smooth_1d(x: np.ndarray, window: int) -> np.ndarray:
    """Simple uniform moving-average smoothing (preserves length via padding)."""
    if window < 2 or x.size < window:
        return x
    kernel = np.ones(window, dtype=np.float64) / window
    # Pad with edge values to preserve length
    pad = window // 2
    padded = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    return np.convolve(padded, kernel, mode="valid")[:x.size]


def _stratification_test(
    radii: np.ndarray,
    volumes: np.ndarray,
    ws: int = 10,
    alpha: float = 1e-3,
    smooth_window: int = 3,
) -> tuple[int | None, float]:
    """Detect a dimension change-point using a sliding Welch t-test.

    Improvements over naive gradient ratio:
    - Smooth the local dimension vector before testing (reduces false positives
      from noise amplification in the finite-difference gradient ratio).
    """
    radii_safe = np.clip(radii, 1e-12, None)
    volumes_safe = np.maximum(volumes, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        grad_v = np.gradient(np.log(volumes_safe))
        grad_r = np.gradient(np.log(radii_safe))
        grad_r = np.where(np.abs(grad_r) < 1e-12, np.nan, grad_r)
        dimvec = np.divide(
            grad_v, grad_r, out=np.full_like(grad_v, np.nan), where=~np.isnan(grad_r)
        )
    # Smooth to reduce noise before the t-test
    finite_mask = np.isfinite(dimvec)
    if finite_mask.any():
        smoothed = _smooth_1d(
            np.where(finite_mask, dimvec, 0.0), max(1, smooth_window)
        )
        dimvec = np.where(finite_mask, smoothed, dimvec)

    for w in range(2 * ws, dimvec.shape[0] - 2 * ws):
        left = dimvec[w - 2 * ws : w - ws]
        left = left[np.logical_and(np.abs(left) > 1e-5, np.isfinite(left))]
        right = dimvec[w + ws : w + 2 * ws]
        right = right[np.logical_and(np.abs(right) > 1e-5, np.isfinite(right))]
        pvalue = _welch_ttest_pvalue(left, right)
        if pvalue < alpha:
            return w, pvalue
    return None, 1.0


def _estimate_stratifications(
    dists_sorted: np.ndarray,
    vol_min: int,
    vol_max: int,
    npts: int,
    args: SimpleNamespace,
    ws: int = 10,
    alpha: float = 1e-3,
) -> dict[str, list[float]]:
    radii = dists_sorted[vol_min:vol_max]
    volumes = np.arange(vol_min, vol_max)
    output: dict[str, list[float]] = {
        "scaling_coeffs": [],
        "dimensions": [],
        "riccis": [],
        "strat_radii": [],
        "strat_volumes": [],
        "pvalues": [],
    }
    vol_min_current = int(np.argmax(radii > 1e-10))
    for _ in range(args.nstrat):
        vol_max_current = radii.shape[0]
        strat_idx, pvalue = _stratification_test(
            radii[vol_min_current:vol_max_current],
            volumes[vol_min_current:vol_max_current],
            ws,
            alpha / max(1, args.nstrat),
        )
        if strat_idx is not None:
            vol_max_current = strat_idx + vol_min_current
        scaling_coeff, dimension, ricci = _geo_estimator(
            radii[vol_min_current:vol_max_current],
            volumes[vol_min_current:vol_max_current],
            npts,
            args,
        )
        output["scaling_coeffs"].append(scaling_coeff)
        output["dimensions"].append(dimension)
        output["riccis"].append(ricci)
        output["strat_volumes"].append(float(vol_min + vol_min_current))
        output["strat_radii"].append(float(radii[vol_min_current]))
        output["pvalues"].append(float(pvalue))
        if strat_idx is None:
            break
        vol_min_current = strat_idx + vol_min_current
    return output


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_volume_range(npts: int, vol_min: int, vol_max: int) -> tuple[int, int]:
    """Clamp (vol_min, vol_max) to a valid kNN range given npts."""
    if npts < 2:
        return 1, 1
    vol_max = min(int(vol_max), npts - 1)
    vol_min = min(int(vol_min), max(1, vol_max - 2))
    if vol_max - vol_min < 5:
        vol_min = max(1, vol_max - 5)
    return vol_min, vol_max


def sorted_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute all-pairs distances and return the per-point sorted distance matrix.

    Shape: (npts, npts), where row 0 is always 0 (self-distance).
    """
    coords64 = np.asarray(coords, dtype=np.float64)
    sq_norms = np.sum(coords64 * coords64, axis=1, keepdims=True)
    d2 = np.maximum(sq_norms + sq_norms.T - 2.0 * (coords64 @ coords64.T), 0.0)
    np.fill_diagonal(d2, 0.0)
    dists = np.sqrt(d2, out=d2)
    return np.sort(dists, axis=0)


def analyze_stratification_from_sorted_distances(
    dists_sorted: np.ndarray,
    *,
    vol_min: int = 8,
    vol_max: int = 64,
    ws: int = 8,
    alpha: float = 1e-2,
    nstrat: int = 3,
) -> list[dict[str, list[float]]]:
    """Analyze stratification using a precomputed sorted distance matrix."""
    npts = int(dists_sorted.shape[0])
    if npts < 2:
        return []
    vol_min, vol_max = normalize_volume_range(npts, vol_min, vol_max)
    args = SimpleNamespace(nstrat=nstrat, miller=True, ricci=False)
    return [
        _estimate_stratifications(
            dists_sorted[:, i], vol_min, vol_max, npts, args, ws=ws, alpha=alpha
        )
        for i in range(npts)
    ]


def analyze_stratification(
    embeddings: torch.Tensor,
    vol_min: int = 8,
    vol_max: int = 64,
    ws: int = 8,
    alpha: float = 1e-2,
    nstrat: int = 3,
) -> tuple[list[dict[str, list[float]]], np.ndarray, np.ndarray]:
    """Analyze embedding stratification and return (results, sorted_dists, unsorted_dists).

    Returns both distance matrices so callers can reuse them
    (e.g. hypothesis metrics needs the unsorted matrix) without recomputation.
    """
    coords = embeddings.cpu().numpy().astype(np.float64)
    sq_norms = np.sum(coords * coords, axis=1, keepdims=True)
    d2 = np.maximum(sq_norms + sq_norms.T - 2.0 * (coords @ coords.T), 0.0)
    np.fill_diagonal(d2, 0.0)
    unsorted_dists = np.sqrt(d2, out=d2)
    dists_sorted = np.sort(unsorted_dists, axis=0)
    results = analyze_stratification_from_sorted_distances(
        dists_sorted,
        vol_min=vol_min,
        vol_max=vol_max,
        ws=ws,
        alpha=alpha,
        nstrat=nstrat,
    )
    return results, dists_sorted, unsorted_dists


def summarize_stratification(
    results: list[dict[str, list[float]]], alpha: float = 1e-2
) -> dict[str, float]:
    first_dims, min_pvals, irr_scores, irregular_tokens = [], [], [], 0
    for res in results:
        if not res or not res["dimensions"]:
            continue
        first_dims.append(res["dimensions"][0])
        min_p = min(res["pvalues"])
        min_pvals.append(min_p)
        irr_scores.append(-np.log10(min_p + 1e-12))
        if min_p < alpha:
            irregular_tokens += 1
    return {
        "num_tokens": len(results),
        "tokens_with_strata": len(first_dims),
        "mean_dim": float(np.mean(first_dims)) if first_dims else float("nan"),
        "median_dim": float(np.median(first_dims)) if first_dims else float("nan"),
        "min_pvalue": float(np.min(min_pvals)) if min_pvals else float("nan"),
        "max_pvalue": float(np.max(min_pvals)) if min_pvals else float("nan"),
        "mean_irregularity": float(np.mean(irr_scores)) if irr_scores else float("nan"),
        "max_irregularity": float(np.max(irr_scores)) if irr_scores else float("nan"),
        "irregular_ratio": irregular_tokens / len(results) if results else float("nan"),
    }


def run_fiber_bundle_test_from_sorted_dists(
    dists_sorted: np.ndarray,
    *,
    vol_min: int = 8,
    vol_max: int = 64,
    ws: int = 8,
    alpha: float = 1e-2,
    nstrat: int = 3,
) -> list[dict[str, list[float]]]:
    return analyze_stratification_from_sorted_distances(
        dists_sorted,
        vol_min=vol_min,
        vol_max=vol_max,
        ws=ws,
        alpha=alpha,
        nstrat=nstrat,
    )


def run_fiber_bundle_test(
    embeddings: torch.Tensor,
    vol_min: int = 8,
    vol_max: int = 64,
    ws: int = 8,
    alpha: float = 1e-2,
    nstrat: int = 3,
) -> tuple[list[dict[str, list[float]]], np.ndarray, np.ndarray]:
    return analyze_stratification(
        embeddings,
        vol_min=vol_min,
        vol_max=vol_max,
        ws=ws,
        alpha=alpha,
        nstrat=nstrat,
    )


def summarize_stratifications(
    results: list[dict[str, list[float]]], alpha: float = 1e-2
) -> dict[str, float]:
    return summarize_stratification(results, alpha=alpha)
