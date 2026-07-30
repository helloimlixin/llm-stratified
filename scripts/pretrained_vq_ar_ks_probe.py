"""Robust KS probe for pretrained VQ-token autoregressive models.

This tests visual analogues of language-side polysemy on a matched tokenizer/AR
pair. It includes AR branch-uniformity diagnostics and an embedding-ball radial
KS test for the local geometry of singular hidden-state positions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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
from fiber.geometry import (  # noqa: E402
    analyze_stratification_from_sorted_distances,
    _welch_ttest_pvalue,
    min_change_pvalue,
    min_fiber_violation_pvalue,
    sorted_distance_matrix,
    summarize_stratification,
)
from pretrained_vq_ar_pipeline import (  # noqa: E402
    LLAMAGEN_PROFILES,
    load_weight_payload,
    parse_class_labels,
    resolve_device,
    resolve_llamagen_repo,
    resolve_precision,
    save_grid,
    tensor_to_pil,
    llamagen_import_context,
)


def _trimmed_mean(values: np.ndarray, *, axis: int = 0, trim: float = 0.10) -> np.ndarray:
    arr = np.sort(np.asarray(values, dtype=np.float64), axis=axis)
    n = arr.shape[axis]
    cut = int(math.floor(max(0.0, min(0.45, trim)) * n))
    if cut > 0 and n > 2 * cut:
        slc = [slice(None)] * arr.ndim
        slc[axis] = slice(cut, n - cut)
        arr = arr[tuple(slc)]
    return np.mean(arr, axis=axis)


def mean_or_nan(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def min_or_nan(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.min()) if arr.size else float("nan")


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = ((a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1)) / (a.size + b.size - 2)
    if pooled <= 0.0:
        return float("nan")
    return float((a.mean() - b.mean()) / math.sqrt(float(pooled)))


def holm_bonferroni(pvalues: np.ndarray, *, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni adjusted p-values and step-down rejection mask."""
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


def permutation_mean_diff_pvalue(a: np.ndarray, b: np.ndarray, *, reps: int = 5000, seed: int = 0) -> tuple[float, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    observed = float(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    extreme = 0
    n_a = int(a.size)
    for _ in range(max(1, int(reps))):
        perm = rng.permutation(pooled)
        diff = float(perm[:n_a].mean() - perm[n_a:].mean())
        if abs(diff) >= abs(observed):
            extreme += 1
    return observed, float((extreme + 1.0) / (float(reps) + 1.0))


def _welch_ttest_pvalue_alt(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    *,
    alternative: str = "two-sided",
) -> float:
    """Welch t-test p-value with the same filtering semantics as the paper code."""
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
    if not np.isfinite(cdf):
        return 1.0
    cdf = max(0.0, min(1.0, cdf))

    if alternative == "two-sided":
        return float(max(0.0, min(1.0, 2.0 * min(cdf, 1.0 - cdf))))
    if alternative == "less":
        return float(cdf)
    if alternative == "greater":
        return float(1.0 - cdf)
    raise ValueError(f"unknown Welch alternative {alternative!r}")


def paper_original_hypothesis_tests(
    radii: np.ndarray,
    volumes: np.ndarray,
    *,
    ws: int = 10,
    alpha: float = 1e-3,
) -> dict[str, float | int | None]:
    """Algorithm-1 p-values for manifold and fiber-bundle tests.

    The public estimator scans sliding windows from small to large radius and
    returns the first significant Welch p-value. The manifold test is two-sided;
    the fiber-bundle test is the one-sided slope-increase violation described in
    the paper.
    """
    radii = np.asarray(radii, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        dimvec = np.gradient(np.log(volumes)) / np.gradient(np.log(radii))

    manifold_idx: int | None = None
    fiber_idx: int | None = None
    manifold_pvalue = 1.0
    fiber_pvalue = 1.0
    manifold_delta = 0.0
    fiber_delta = 0.0

    for w in range(2 * ws, dimvec.shape[0] - 2 * ws):
        left = dimvec[w - 2 * ws : w - ws]
        left = left[np.logical_and(np.abs(left) > 1e-5, np.isfinite(left))]
        right = dimvec[w + ws : w + 2 * ws]
        right = right[np.logical_and(np.abs(right) > 1e-5, np.isfinite(right))]
        if left.size < 2 or right.size < 2:
            continue
        delta = float(np.mean(right) - np.mean(left))

        if manifold_idx is None:
            pvalue = _welch_ttest_pvalue_alt(left, right, alternative="two-sided")
            if pvalue < alpha:
                manifold_idx = int(w)
                manifold_pvalue = float(pvalue)
                manifold_delta = delta

        if fiber_idx is None and delta > 0.0:
            pvalue = _welch_ttest_pvalue_alt(left, right, alternative="less")
            if pvalue < alpha:
                fiber_idx = int(w)
                fiber_pvalue = float(pvalue)
                fiber_delta = delta

        if manifold_idx is not None and fiber_idx is not None:
            break

    return {
        "manifold_index": manifold_idx,
        "manifold_pvalue": float(manifold_pvalue),
        "manifold_delta": float(manifold_delta),
        "fiber_index": fiber_idx,
        "fiber_pvalue": float(fiber_pvalue),
        "fiber_delta": float(fiber_delta),
    }


def pca_project(features: torch.Tensor, *, dims: int) -> torch.Tensor:
    x = features.detach().float().cpu()
    if dims <= 0 or int(dims) >= int(x.shape[1]):
        return x
    x = x - x.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    basis = vh[: int(dims)].T.contiguous()
    return x @ basis


def paper_original_stratification_test(
    radii: np.ndarray,
    volumes: np.ndarray,
    *,
    ws: int = 10,
    alpha: float = 1e-3,
) -> tuple[int | None, float]:
    """Reference-style sliding Welch test from stratified_estimator."""
    radii_safe = np.clip(np.asarray(radii, dtype=np.float64), 1e-12, None)
    volumes_safe = np.maximum(np.asarray(volumes, dtype=np.float64), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dimvec = np.gradient(np.log(volumes_safe)) / np.gradient(np.log(radii_safe))
    for w in range(2 * ws, dimvec.shape[0] - 2 * ws):
        left = dimvec[w - 2 * ws : w - ws]
        left = left[np.logical_and(np.abs(left) > 1e-5, np.isfinite(left))]
        right = dimvec[w + ws : w + 2 * ws]
        right = right[np.logical_and(np.abs(right) > 1e-5, np.isfinite(right))]
        pvalue = _welch_ttest_pvalue(left, right)
        if pvalue < alpha:
            return int(w), float(pvalue)
    return None, 1.0


def paper_original_geo_estimator(
    radii: np.ndarray,
    volumes: np.ndarray,
    npts: int,
) -> tuple[float, float, float]:
    radii = np.asarray(radii, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    valid = np.isfinite(radii) & np.isfinite(volumes) & (radii > 1e-10) & (volumes > 0)
    if int(valid.sum()) < 2:
        return float("nan"), float("nan"), 0.0
    rstack = np.column_stack((np.ones(int(valid.sum()), dtype=np.float64), np.log(radii[valid])))
    coeffs, residuals, _rank, _sv = np.linalg.lstsq(rstack, np.log(volumes[valid]), rcond=None)
    scaling_coeff = float(np.exp(coeffs[0]) / max(1, int(npts)))
    if residuals.size:
        scaling_coeff *= float(np.exp(min(0.5 * residuals[0] ** 2, 50.0)))
    return scaling_coeff, float(coeffs[1]), 0.0


def estimate_stratifications_paper_original(
    dists_sorted: np.ndarray,
    vol_min: int,
    vol_max: int,
    npts: int,
    *,
    ws: int,
    alpha: float,
    nstrat: int,
) -> dict[str, list[float]]:
    radii = np.asarray(dists_sorted[vol_min:vol_max], dtype=np.float64)
    volumes = np.arange(vol_min, vol_max, dtype=np.float64)
    output: dict[str, list[float]] = {
        "scaling_coeffs": [],
        "dimensions": [],
        "riccis": [],
        "strat_radii": [],
        "strat_volumes": [],
        "pvalues": [],
    }
    positive = np.isfinite(radii) & (radii > 1e-10)
    if not positive.any():
        return output
    vol_min_current = int(np.argmax(positive))
    hypothesis = paper_original_hypothesis_tests(
        radii[vol_min_current:],
        volumes[vol_min_current:],
        ws=ws,
        alpha=float(alpha) / max(1, int(nstrat)),
    )
    output["paper_manifold_pvalue"] = [float(hypothesis["manifold_pvalue"])]
    output["paper_fiber_pvalue"] = [float(hypothesis["fiber_pvalue"])]
    output["paper_manifold_index"] = [
        float(hypothesis["manifold_index"]) if hypothesis["manifold_index"] is not None else float("nan")
    ]
    output["paper_fiber_index"] = [
        float(hypothesis["fiber_index"]) if hypothesis["fiber_index"] is not None else float("nan")
    ]
    output["paper_manifold_delta"] = [float(hypothesis["manifold_delta"])]
    output["paper_fiber_delta"] = [float(hypothesis["fiber_delta"])]
    for _ in range(max(1, int(nstrat))):
        vol_max_current = int(radii.shape[0])
        strat_idx, pvalue = paper_original_stratification_test(
            radii[vol_min_current:vol_max_current],
            volumes[vol_min_current:vol_max_current],
            ws=ws,
            alpha=float(alpha) / max(1, int(nstrat)),
        )
        if strat_idx is not None:
            vol_max_current = int(strat_idx) + vol_min_current
        if vol_max_current - vol_min_current < 2:
            break
        scaling_coeff, dimension, ricci = paper_original_geo_estimator(
            radii[vol_min_current:vol_max_current],
            volumes[vol_min_current:vol_max_current],
            npts,
        )
        output["scaling_coeffs"].append(scaling_coeff)
        output["dimensions"].append(dimension)
        output["riccis"].append(ricci)
        output["strat_volumes"].append(float(vol_min + vol_min_current))
        output["strat_radii"].append(float(radii[vol_min_current]))
        output["pvalues"].append(float(pvalue))
        if strat_idx is None:
            break
        vol_min_current = int(strat_idx) + vol_min_current
    return output


def analyze_stratification_paper_original(
    dists_sorted: np.ndarray,
    *,
    vol_min: int,
    vol_max: int,
    ws: int,
    alpha: float,
    nstrat: int,
) -> list[dict[str, list[float]]]:
    npts = int(dists_sorted.shape[0])
    return [
        estimate_stratifications_paper_original(
            dists_sorted[:, idx],
            int(vol_min),
            min(int(vol_max), max(1, npts - 1)),
            npts,
            ws=ws,
            alpha=alpha,
            nstrat=nstrat,
        )
        for idx in range(int(dists_sorted.shape[1]))
    ]


def ranked_probability_uniform_ks(probs: np.ndarray) -> np.ndarray:
    """Order-free KS-style distance between sorted code mass and uniform mass."""
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("probs must have shape (n_contexts, vocab_size)")
    arr = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1e-12)
    sorted_probs = np.sort(arr, axis=1)[:, ::-1]
    cdf = np.cumsum(sorted_probs, axis=1)
    uniform_cdf = np.arange(1, arr.shape[1] + 1, dtype=np.float64) / float(arr.shape[1])
    return np.max(np.abs(cdf - uniform_cdf[None, :]), axis=1)


def one_sample_uniform_ks(values: np.ndarray) -> float:
    """One-sample KS statistic against Uniform(0, 1)."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    arr = np.sort(np.clip(arr, 0.0, 1.0))
    n = int(arr.size)
    cdf_hi = np.arange(1, n + 1, dtype=np.float64) / float(n)
    cdf_lo = np.arange(0, n, dtype=np.float64) / float(n)
    return float(max(np.max(cdf_hi - arr), np.max(arr - cdf_lo)))


def asymptotic_uniform_ks_pvalue(statistic: float, n: int) -> float:
    """Kolmogorov one-sample asymptotic p-value for Uniform(0, 1)."""
    if not np.isfinite(statistic) or int(n) <= 0:
        return float("nan")
    n_float = float(n)
    lam = (math.sqrt(n_float) + 0.12 + 0.11 / math.sqrt(n_float)) * float(statistic)
    if lam <= 0.0:
        return 1.0
    total = 0.0
    for j in range(1, 101):
        total += ((-1.0) ** (j - 1)) * math.exp(-2.0 * (j ** 2) * (lam ** 2))
    return float(max(0.0, min(1.0, 2.0 * total)))


def _trimmed_vector(values: np.ndarray, *, trim: float) -> np.ndarray:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr
    cut = int(math.floor(max(0.0, min(0.45, float(trim))) * int(arr.size)))
    if cut > 0 and int(arr.size) > 2 * cut:
        arr = arr[cut:-cut]
    return arr


def radial_uniformity_from_distances(
    *,
    inner_distances: np.ndarray,
    boundary_radius: float,
    ambient_dim: int,
    dimension_trim: float,
    min_inner: int,
    eps: float = 1e-12,
) -> dict[str, float]:
    r_boundary = float(boundary_radius)
    if not np.isfinite(r_boundary) or r_boundary <= eps:
        return {"ks": float("nan"), "pvalue": float("nan"), "dimension": float("nan"), "mean_u": float("nan")}
    inner = np.asarray(inner_distances, dtype=np.float64)
    inner = inner[np.isfinite(inner)]
    inner = inner[(inner > eps) & (inner < r_boundary + eps)]
    if inner.size < int(min_inner):
        return {"ks": float("nan"), "pvalue": float("nan"), "dimension": float("nan"), "mean_u": float("nan")}

    logs = np.log(np.maximum(r_boundary, eps) / np.maximum(inner, eps))
    logs = _trimmed_vector(logs[logs > 0.0], trim=dimension_trim)
    if logs.size == 0:
        return {"ks": float("nan"), "pvalue": float("nan"), "dimension": float("nan"), "mean_u": float("nan")}
    local_dim = 1.0 / max(float(np.mean(logs)), eps)
    local_dim = float(np.clip(local_dim, 0.25, max(1.0, float(ambient_dim))))

    u = np.clip((inner / r_boundary) ** local_dim, 0.0, 1.0)
    ks = one_sample_uniform_ks(u)
    return {
        "ks": float(ks),
        "pvalue": asymptotic_uniform_ks_pvalue(float(ks), int(u.size)),
        "dimension": local_dim,
        "mean_u": float(np.mean(u)),
    }


def embedding_ball_radial_uniformity_ks(
    *,
    features: np.ndarray,
    volume: int,
    exclude_self: bool = True,
    dimension_trim: float = 0.10,
    min_inner: int = 8,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Robust embedding-ball uniformity test on token embeddings.

    For each center, choose a fixed-volume kNN ball using embedding geometry
    only. The kth neighbor gives the local radius R. Conditional on R, points
    drawn uniformly from a d-dimensional ball have radial CDF values
    u=(r/R)^d distributed as Uniform(0, 1), so we estimate local intrinsic
    dimension with a trimmed MLE and run a one-sample KS test on the inner
    radii. This radial reduction is intentionally less sensitive to ambient
    high dimensionality than a full multivariate goodness-of-fit test.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must have shape (n_contexts, dim)")

    n, ambient_dim = int(x.shape[0]), int(x.shape[1])
    max_neighbors = n - 1 if exclude_self else n
    k = max(2, min(int(volume), max_neighbors))
    sq_norms = np.sum(x * x, axis=1, keepdims=True)
    d2 = np.maximum(sq_norms + sq_norms.T - 2.0 * (x @ x.T), 0.0)
    if exclude_self:
        np.fill_diagonal(d2, np.inf)
    neighbor_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    neighbor_d2 = np.take_along_axis(d2, neighbor_idx, axis=1)
    order = np.argsort(neighbor_d2, axis=1)
    neighbor_idx = np.take_along_axis(neighbor_idx, order, axis=1)
    neighbor_d2 = np.take_along_axis(neighbor_d2, order, axis=1)
    neighbor_dist = np.sqrt(np.maximum(neighbor_d2, 0.0))

    radius = neighbor_dist[:, -1]
    ks = np.full(n, np.nan, dtype=np.float64)
    pvalue = np.full(n, np.nan, dtype=np.float64)
    dimension = np.full(n, np.nan, dtype=np.float64)
    mean_u = np.full(n, np.nan, dtype=np.float64)
    var_u = np.full(n, np.nan, dtype=np.float64)
    inner_count = np.zeros(n, dtype=np.float64)

    upper_dim = max(1.0, float(ambient_dim))
    for idx in range(n):
        r_boundary = float(radius[idx])
        if not np.isfinite(r_boundary) or r_boundary <= eps:
            continue
        inner = neighbor_dist[idx, :-1]
        inner = inner[np.isfinite(inner)]
        inner = inner[(inner > eps) & (inner < r_boundary + eps)]
        if inner.size < int(min_inner):
            continue

        logs = np.log(np.maximum(r_boundary, eps) / np.maximum(inner, eps))
        logs = _trimmed_vector(logs[logs > 0.0], trim=dimension_trim)
        if logs.size == 0:
            continue
        local_dim = 1.0 / max(float(np.mean(logs)), eps)
        local_dim = float(np.clip(local_dim, 0.25, upper_dim))

        u = np.clip((inner / r_boundary) ** local_dim, 0.0, 1.0)
        inner_count[idx] = float(u.size)
        dimension[idx] = local_dim
        mean_u[idx] = float(np.mean(u))
        var_u[idx] = float(np.var(u))
        ks[idx] = one_sample_uniform_ks(u)
        pvalue[idx] = asymptotic_uniform_ks_pvalue(float(ks[idx]), int(u.size))

    return {
        "ks": ks,
        "pvalue": pvalue,
        "dimension": dimension,
        "radius": radius,
        "inner_count": inner_count,
        "mean_u": mean_u,
        "var_u": var_u,
        "neighbor_indices": neighbor_idx,
    }


def embedding_radius_uniformity_threshold(
    *,
    features: np.ndarray,
    volume_min: int,
    volume_max: int,
    volume_step: int,
    uniform_pvalue: float,
    max_ks: float,
    consecutive: int,
    exclude_self: bool = True,
    dimension_trim: float = 0.10,
    min_inner: int = 8,
) -> dict[str, np.ndarray]:
    """Find the first radius where an embedding ball is nearly uniform."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must have shape (n_contexts, dim)")

    n, ambient_dim = int(x.shape[0]), int(x.shape[1])
    max_neighbors = n - 1 if exclude_self else n
    k_min = max(2, min(int(volume_min), max_neighbors))
    k_max = max(k_min, min(int(volume_max), max_neighbors))
    step = max(1, int(volume_step))
    candidate_volumes = np.unique(np.arange(k_min, k_max + 1, step, dtype=np.int64))
    if int(candidate_volumes[-1]) != k_max:
        candidate_volumes = np.unique(np.concatenate([candidate_volumes, np.asarray([k_max], dtype=np.int64)]))

    sq_norms = np.sum(x * x, axis=1, keepdims=True)
    d2 = np.maximum(sq_norms + sq_norms.T - 2.0 * (x @ x.T), 0.0)
    if exclude_self:
        np.fill_diagonal(d2, np.inf)
    neighbor_idx = np.argpartition(d2, kth=k_max - 1, axis=1)[:, :k_max]
    neighbor_d2 = np.take_along_axis(d2, neighbor_idx, axis=1)
    order = np.argsort(neighbor_d2, axis=1)
    neighbor_idx = np.take_along_axis(neighbor_idx, order, axis=1)
    neighbor_d2 = np.take_along_axis(neighbor_d2, order, axis=1)
    neighbor_dist = np.sqrt(np.maximum(neighbor_d2, 0.0))

    num_candidates = int(candidate_volumes.size)
    ks_by_volume = np.full((n, num_candidates), np.nan, dtype=np.float64)
    pvalue_by_volume = np.full((n, num_candidates), np.nan, dtype=np.float64)
    radius_by_volume = np.full((n, num_candidates), np.nan, dtype=np.float64)
    dimension_by_volume = np.full((n, num_candidates), np.nan, dtype=np.float64)

    threshold_radius = np.full(n, np.nan, dtype=np.float64)
    threshold_volume = np.full(n, np.nan, dtype=np.float64)
    threshold_ks = np.full(n, np.nan, dtype=np.float64)
    threshold_pvalue = np.full(n, np.nan, dtype=np.float64)
    threshold_dimension = np.full(n, np.nan, dtype=np.float64)
    best_radius = np.full(n, np.nan, dtype=np.float64)
    best_volume = np.full(n, np.nan, dtype=np.float64)
    best_ks = np.full(n, np.nan, dtype=np.float64)
    best_pvalue = np.full(n, np.nan, dtype=np.float64)

    p_cutoff = float(uniform_pvalue)
    ks_cutoff = float(max_ks)
    use_ks_cutoff = np.isfinite(ks_cutoff) and ks_cutoff > 0.0
    run_length = max(1, int(consecutive))

    for idx in range(n):
        distances = neighbor_dist[idx]
        for j, k in enumerate(candidate_volumes):
            k_int = int(k)
            radius = float(distances[k_int - 1])
            stats = radial_uniformity_from_distances(
                inner_distances=distances[: k_int - 1],
                boundary_radius=radius,
                ambient_dim=ambient_dim,
                dimension_trim=dimension_trim,
                min_inner=min_inner,
            )
            radius_by_volume[idx, j] = radius
            ks_by_volume[idx, j] = float(stats["ks"])
            pvalue_by_volume[idx, j] = float(stats["pvalue"])
            dimension_by_volume[idx, j] = float(stats["dimension"])

        finite = np.flatnonzero(np.isfinite(ks_by_volume[idx]))
        if finite.size:
            best_j = int(finite[np.argmin(ks_by_volume[idx, finite])])
            best_radius[idx] = radius_by_volume[idx, best_j]
            best_volume[idx] = float(candidate_volumes[best_j])
            best_ks[idx] = ks_by_volume[idx, best_j]
            best_pvalue[idx] = pvalue_by_volume[idx, best_j]

        near_uniform = np.isfinite(ks_by_volume[idx]) & np.isfinite(pvalue_by_volume[idx])
        near_uniform &= pvalue_by_volume[idx] >= p_cutoff
        if use_ks_cutoff:
            near_uniform &= ks_by_volume[idx] <= ks_cutoff
        threshold_j: int | None = None
        for start in range(0, max(0, num_candidates - run_length + 1)):
            if bool(np.all(near_uniform[start : start + run_length])):
                threshold_j = int(start)
                break
        if threshold_j is None:
            continue
        threshold_radius[idx] = radius_by_volume[idx, threshold_j]
        threshold_volume[idx] = float(candidate_volumes[threshold_j])
        threshold_ks[idx] = ks_by_volume[idx, threshold_j]
        threshold_pvalue[idx] = pvalue_by_volume[idx, threshold_j]
        threshold_dimension[idx] = dimension_by_volume[idx, threshold_j]

    return {
        "candidate_volumes": candidate_volumes.astype(np.float64),
        "ks_by_volume": ks_by_volume,
        "pvalue_by_volume": pvalue_by_volume,
        "radius_by_volume": radius_by_volume,
        "dimension_by_volume": dimension_by_volume,
        "threshold_radius": threshold_radius,
        "threshold_volume": threshold_volume,
        "threshold_ks": threshold_ks,
        "threshold_pvalue": threshold_pvalue,
        "threshold_dimension": threshold_dimension,
        "threshold_found": np.isfinite(threshold_radius),
        "best_radius": best_radius,
        "best_volume": best_volume,
        "best_ks": best_ks,
        "best_pvalue": best_pvalue,
        "neighbor_indices": neighbor_idx,
    }


def topk_branch_uniform_ks(probs: np.ndarray, *, top_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Uniformity over the most plausible next patch-token continuations."""
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("probs must have shape (n_contexts, vocab_size)")
    k = max(2, min(int(top_k), int(arr.shape[1])))
    arr = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1e-12)
    top_indices = np.argpartition(arr, -k, axis=1)[:, -k:]
    top = np.take_along_axis(arr, top_indices, axis=1)
    order = np.argsort(top, axis=1)[:, ::-1]
    top = np.take_along_axis(top, order, axis=1)
    top_indices = np.take_along_axis(top_indices, order, axis=1)
    top_mass = np.sum(top, axis=1)
    local = top / np.maximum(top_mass[:, None], 1e-12)
    cdf = np.cumsum(local, axis=1)
    uniform_cdf = np.arange(1, k + 1, dtype=np.float64) / float(k)
    ks = np.max(np.abs(cdf - uniform_cdf[None, :]), axis=1)
    entropy = -np.sum(local * np.log(np.clip(local, 1e-12, 1.0)), axis=1) / math.log(k)
    return ks, entropy, top_mass, top_indices


def local_ball_uniformity_ks(
    *,
    features: np.ndarray,
    probs: np.ndarray,
    target_codes: np.ndarray,
    volume: int,
    exclude_self: bool = True,
) -> dict[str, np.ndarray]:
    """Uniformity over a fixed-volume local neighborhood in embedding space.

    The ball is chosen from geometry alone. The model distribution for the
    center context is restricted to the target codes observed among the k
    nearest neighbor contexts, then normalized and compared to a uniform
    distribution over neighbor instances with an order-free one-sample KS
    statistic. Duplicate target codes split their probability mass across the
    duplicate neighbor instances.
    """
    x = np.asarray(features, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(target_codes, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError("features must have shape (n_contexts, dim)")
    if p.ndim != 2:
        raise ValueError("probs must have shape (n_contexts, vocab_size)")
    if y.ndim != 1 or y.shape[0] != x.shape[0] or p.shape[0] != x.shape[0]:
        raise ValueError("features, probs, and target_codes must share the same context count")

    n = int(x.shape[0])
    max_neighbors = n - 1 if exclude_self else n
    k = max(1, min(int(volume), max_neighbors))
    sq_norms = np.sum(x * x, axis=1, keepdims=True)
    d2 = np.maximum(sq_norms + sq_norms.T - 2.0 * (x @ x.T), 0.0)
    if exclude_self:
        np.fill_diagonal(d2, np.inf)
    neighbor_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    neighbor_d2 = np.take_along_axis(d2, neighbor_idx, axis=1)
    order = np.argsort(neighbor_d2, axis=1)
    neighbor_idx = np.take_along_axis(neighbor_idx, order, axis=1)
    neighbor_d2 = np.take_along_axis(neighbor_d2, order, axis=1)
    neighbor_codes = y[neighbor_idx]

    ks = np.full(n, np.nan, dtype=np.float64)
    entropy = np.full(n, np.nan, dtype=np.float64)
    mass = np.full(n, np.nan, dtype=np.float64)
    unique_count = np.zeros(n, dtype=np.float64)
    radius = np.sqrt(np.maximum(neighbor_d2[:, -1], 0.0))
    denom = math.log(k) if k > 1 else 1.0
    uniform_cdf = np.arange(1, k + 1, dtype=np.float64) / float(k)

    for idx in range(n):
        codes = neighbor_codes[idx]
        unique, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
        unique_count[idx] = float(unique.size)
        unique_probs = p[idx, unique]
        instance_mass = unique_probs[inverse] / np.maximum(counts[inverse].astype(np.float64), 1.0)
        total = float(np.sum(instance_mass))
        mass[idx] = total
        if not np.isfinite(total) or total <= 0.0:
            continue
        q = instance_mass / total
        sorted_q = np.sort(q)[::-1]
        cdf = np.cumsum(sorted_q)
        ks[idx] = float(np.max(np.abs(cdf - uniform_cdf)))
        entropy[idx] = float(-np.sum(q * np.log(np.clip(q, 1e-12, 1.0))) / max(denom, 1e-12))

    return {
        "ks": ks,
        "entropy": entropy,
        "mass": mass,
        "unique_count": unique_count,
        "radius": radius,
        "neighbor_indices": neighbor_idx,
        "neighbor_codes": neighbor_codes,
    }


def permuted_categorical_uniform_ks(
    probs: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Code-order robust categorical KS via random codebook permutations."""
    arr = np.asarray(probs, dtype=np.float64)
    arr = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(seed)
    uniform_cdf = np.arange(1, arr.shape[1] + 1, dtype=np.float64) / float(arr.shape[1])
    stats = np.zeros((max(1, int(permutations)), arr.shape[0]), dtype=np.float64)
    for perm_idx in range(stats.shape[0]):
        order = rng.permutation(arr.shape[1])
        cdf = np.cumsum(arr[:, order], axis=1)
        stats[perm_idx] = np.max(np.abs(cdf - uniform_cdf[None, :]), axis=1)
    return {
        "median": np.median(stats, axis=0),
        "trimmed_mean": _trimmed_mean(stats, axis=0),
        "max": np.max(stats, axis=0),
    }


def load_llamagen_gpt(
    *,
    profile_name: str,
    repo_path: Path,
    device: torch.device,
    dtype: torch.dtype,
):
    profile = dict(LLAMAGEN_PROFILES[profile_name])
    gpt_path = hf_hub_download(repo_id=profile["repo_id"], filename=profile["gpt_file"])
    latent_size = int(profile["image_size"]) // int(profile["downsample_size"])
    with llamagen_import_context(repo_path):
        from autoregressive.models.gpt import GPT_models

        model = GPT_models[profile["gpt_model"]](
            vocab_size=int(profile["codebook_size"]),
            block_size=latent_size ** 2,
            num_classes=1000,
            cls_token_num=1,
            model_type="c2i",
        ).to(device=device, dtype=dtype)
        missing, unexpected = model.load_state_dict(load_weight_payload(gpt_path), strict=False)
        model.eval()
    return model, profile, list(missing), list(unexpected)


@torch.no_grad()
def teacher_force_logits_and_hidden(
    model,
    tokens: torch.Tensor,
    class_labels: list[int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = tokens.to(device=device, dtype=torch.long)
    labels = torch.tensor(class_labels, dtype=torch.long, device=device)
    if labels.numel() != tokens.shape[0]:
        raise ValueError("number of class labels must match token batch")
    shifted = tokens[:, :-1].contiguous()
    input_pos = torch.arange(tokens.shape[1], dtype=torch.long, device=device)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["hidden"] = output.detach().float().cpu()

    handle = model.norm.register_forward_hook(hook)
    try:
        logits, _ = model(shifted, labels, input_pos=input_pos)
    finally:
        handle.remove()
    hidden = captured["hidden"]
    if logits.shape[1] != tokens.shape[1]:
        raise RuntimeError(f"expected logits length {tokens.shape[1]}, got {logits.shape[1]}")
    return logits.detach().float().cpu(), hidden


def select_singular_mask(
    *,
    pvalues: np.ndarray,
    irregularity: np.ndarray,
    alpha: float,
    fraction: float,
    min_count: int,
) -> tuple[np.ndarray, str]:
    rejected = np.isfinite(pvalues) & (pvalues < float(alpha))
    if int(rejected.sum()) >= int(min_count):
        return rejected, "fiber_violation_pvalue"
    finite = np.isfinite(irregularity)
    mask = np.zeros(irregularity.shape, dtype=bool)
    idx = np.flatnonzero(finite)
    if idx.size == 0:
        return mask, "none"
    count = max(int(min_count), int(math.ceil(float(fraction) * idx.size)))
    count = min(count, int(idx.size))
    order = idx[np.argsort(irregularity[idx])]
    mask[order[-count:]] = True
    return mask, "top_irregularity_fallback"


def finite_minimum(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.asarray([], dtype=np.float64)
    stacked = np.stack([np.asarray(arr, dtype=np.float64) for arr in arrays], axis=0)
    finite = np.isfinite(stacked)
    safe = np.where(finite, stacked, np.inf)
    out = np.min(safe, axis=0)
    out[~np.any(finite, axis=0)] = np.nan
    return out


def _paper_result_scalar(result: dict[str, Any], key: str, fallback: float) -> float:
    values = result.get(key)
    if isinstance(values, (list, tuple, np.ndarray)) and len(values) > 0:
        try:
            value = float(values[0])
            return value if math.isfinite(value) else fallback
        except Exception:
            return fallback
    try:
        value = float(values)
        return value if math.isfinite(value) else fallback
    except Exception:
        return fallback


def _paper_radius_summary(
    *,
    label: str,
    results: list[dict[str, list[float]]],
    alpha: float,
) -> dict[str, Any]:
    manifold_pvalues = np.asarray(
        [
            _paper_result_scalar(result, "paper_manifold_pvalue", min_change_pvalue(result))
            for result in results
        ],
        dtype=np.float64,
    )
    fiber_pvalues = np.asarray(
        [
            _paper_result_scalar(result, "paper_fiber_pvalue", min_fiber_violation_pvalue(result))
            for result in results
        ],
        dtype=np.float64,
    )
    manifold_delta = np.asarray(
        [_paper_result_scalar(result, "paper_manifold_delta", 0.0) for result in results],
        dtype=np.float64,
    )
    fiber_delta = np.asarray(
        [_paper_result_scalar(result, "paper_fiber_delta", 0.0) for result in results],
        dtype=np.float64,
    )
    fiber_adjusted, fiber_rejected = holm_bonferroni(fiber_pvalues, alpha=alpha)
    return {
        "label": label,
        "results": results,
        "manifold_pvalues": manifold_pvalues,
        "manifold_adjusted_pvalues": np.full(manifold_pvalues.shape, np.nan, dtype=np.float64),
        "manifold_rejected": np.zeros(manifold_pvalues.shape, dtype=bool),
        "fiber_pvalues": fiber_pvalues,
        "manifold_delta": manifold_delta,
        "fiber_delta": fiber_delta,
        "fiber_adjusted_pvalues": fiber_adjusted,
        "fiber_rejected": fiber_rejected,
        "summary": summarize_stratification(results, alpha=alpha),
    }


def run_paper_protocol(
    features: torch.Tensor,
    *,
    small_vol_min: int,
    small_vol_max: int,
    large_vol_min: int,
    large_vol_max: int,
    window_size: int,
    alpha: float,
    nstrat: int,
    geometry_mode: str,
) -> dict[str, Any]:
    """Apply Algorithm-1 manifold/fiber tests at two neighborhood radii."""
    coords = features.detach().cpu().numpy().astype(np.float64)
    dists_sorted = sorted_distance_matrix(coords)
    radius_configs = {
        "small": (int(small_vol_min), int(small_vol_max)),
        "large": (int(large_vol_min), int(large_vol_max)),
    }
    radii: dict[str, Any] = {}
    for label, (vol_min, vol_max) in radius_configs.items():
        if geometry_mode == "original":
            results = analyze_stratification_paper_original(
                dists_sorted,
                vol_min=vol_min,
                vol_max=vol_max,
                ws=window_size,
                alpha=alpha,
                nstrat=nstrat,
            )
        elif geometry_mode == "robust":
            results = analyze_stratification_from_sorted_distances(
                dists_sorted,
                vol_min=vol_min,
                vol_max=vol_max,
                ws=window_size,
                alpha=alpha,
                nstrat=nstrat,
            )
        else:
            raise ValueError(f"unknown paper geometry mode: {geometry_mode}")
        radius = _paper_radius_summary(label=label, results=results, alpha=alpha)
        radius["vol_min"] = int(vol_min)
        radius["vol_max"] = int(vol_max)
        radii[label] = radius

    manifold_pvalues_all = np.concatenate(
        [radii["small"]["manifold_pvalues"], radii["large"]["manifold_pvalues"]]
    )
    manifold_adjusted_all, manifold_rejected_all = holm_bonferroni(manifold_pvalues_all, alpha=alpha)
    n_small = int(radii["small"]["manifold_pvalues"].shape[0])
    radii["small"]["manifold_adjusted_pvalues"] = manifold_adjusted_all[:n_small]
    radii["small"]["manifold_rejected"] = manifold_rejected_all[:n_small]
    radii["large"]["manifold_adjusted_pvalues"] = manifold_adjusted_all[n_small:]
    radii["large"]["manifold_rejected"] = manifold_rejected_all[n_small:]

    reject_arrays = [
        radii["small"]["manifold_rejected"],
        radii["small"]["fiber_rejected"],
        radii["large"]["manifold_rejected"],
        radii["large"]["fiber_rejected"],
    ]
    adjusted_arrays = [
        radii["small"]["manifold_adjusted_pvalues"],
        radii["small"]["fiber_adjusted_pvalues"],
        radii["large"]["manifold_adjusted_pvalues"],
        radii["large"]["fiber_adjusted_pvalues"],
    ]
    raw_arrays = [
        radii["small"]["manifold_pvalues"],
        radii["small"]["fiber_pvalues"],
        radii["large"]["manifold_pvalues"],
        radii["large"]["fiber_pvalues"],
    ]
    any_rejected = np.logical_or.reduce(reject_arrays)
    best_adjusted = finite_minimum(adjusted_arrays)
    best_raw = finite_minimum(raw_arrays)
    irregularity = np.where(
        np.isfinite(best_adjusted),
        -np.log10(np.maximum(best_adjusted, 1e-300)),
        0.0,
    )
    return {
        "radii": radii,
        "any_rejected": any_rejected,
        "manifold_any_rejected": radii["small"]["manifold_rejected"] | radii["large"]["manifold_rejected"],
        "fiber_any_rejected": radii["small"]["fiber_rejected"] | radii["large"]["fiber_rejected"],
        "best_adjusted_pvalue": best_adjusted,
        "best_raw_pvalue": best_raw,
        "irregularity": irregularity,
    }


def choose_paper_singular_mask(protocol: dict[str, Any], source: str) -> tuple[np.ndarray, str]:
    small = protocol["radii"]["small"]
    large = protocol["radii"]["large"]
    choices = {
        "paper_any": protocol["any_rejected"],
        "paper_stratified_any": protocol["any_rejected"],
        "paper_manifold_any": protocol["manifold_any_rejected"],
        "paper_fiber_any": protocol["fiber_any_rejected"],
        "paper_stratified_small": small["manifold_rejected"] | small["fiber_rejected"],
        "paper_stratified_large": large["manifold_rejected"] | large["fiber_rejected"],
        "paper_fiber_small": small["fiber_rejected"],
        "paper_fiber_large": large["fiber_rejected"],
        "paper_manifold_small": small["manifold_rejected"],
        "paper_manifold_large": large["manifold_rejected"],
    }
    if source not in choices:
        raise ValueError(f"unknown singular source {source!r}; expected one of {sorted(choices)}")
    return np.asarray(choices[source], dtype=bool), source


def load_codebook_singular_code_masks(path: str | Path, *, vocab_size: int) -> dict[str, np.ndarray]:
    """Load code-ID singular masks emitted by pretrained_vq_codebook_stratification_probe."""
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    masks: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
            continue
        mask = np.zeros(int(vocab_size), dtype=bool)
        indices = np.asarray(value, dtype=np.int64)
        indices = indices[(indices >= 0) & (indices < int(vocab_size))]
        mask[indices] = True
        masks[str(key)] = mask
    return masks


def codebook_position_masks_from_tokens(
    code_masks: dict[str, np.ndarray],
    tokens: np.ndarray,
) -> dict[str, np.ndarray]:
    """Map code-ID masks to target-token and previous-token position masks."""
    if not code_masks:
        return {}
    token_arr = np.asarray(tokens, dtype=np.int64)
    flat_targets = token_arr.reshape(-1)
    out: dict[str, np.ndarray] = {}
    for name, code_mask in code_masks.items():
        target_mask = code_mask[flat_targets]
        prev_mask = np.zeros(token_arr.shape, dtype=bool)
        if token_arr.shape[1] > 1:
            prev_mask[:, 1:] = code_mask[token_arr[:, :-1]]
        out[f"codebook_target_{name}"] = target_mask
        out[f"codebook_prev_{name}"] = prev_mask.reshape(-1)
    return out


def build_codebook_control_code_masks(
    code_masks: dict[str, np.ndarray],
    tokens: np.ndarray,
    *,
    source: str,
    random_controls: int,
    frequency_controls: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Build non-singular random and frequency-matched code-ID controls."""
    if not code_masks or not source or source not in code_masks:
        return {}
    reference = np.asarray(code_masks[source], dtype=bool)
    vocab_size = int(reference.size)
    ref_codes = np.flatnonzero(reference)
    candidates = np.flatnonzero(~reference)
    if ref_codes.size == 0 or candidates.size == 0:
        return {}
    k = min(int(ref_codes.size), int(candidates.size))
    token_counts = np.bincount(np.asarray(tokens, dtype=np.int64).reshape(-1), minlength=vocab_size)
    out: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(int(seed))

    for control_idx in range(max(0, int(random_controls))):
        chosen = rng.choice(candidates, size=k, replace=False)
        mask = np.zeros(vocab_size, dtype=bool)
        mask[chosen] = True
        out[f"random_{source}_{control_idx:02d}"] = mask

    ref_counts = token_counts[ref_codes]
    count_values, needs = np.unique(ref_counts, return_counts=True)
    for control_idx in range(max(0, int(frequency_controls))):
        available = np.ones(vocab_size, dtype=bool)
        available[reference] = False
        chosen_parts: list[np.ndarray] = []
        for count_value, need in zip(count_values, needs):
            available_codes = np.flatnonzero(available)
            if available_codes.size == 0:
                break
            exact = available_codes[token_counts[available_codes] == int(count_value)]
            take = min(int(need), int(exact.size))
            if take > 0:
                selected = rng.choice(exact, size=take, replace=False)
                chosen_parts.append(selected)
                available[selected] = False
            remaining_need = int(need) - take
            if remaining_need <= 0:
                continue
            available_codes = np.flatnonzero(available)
            if available_codes.size == 0:
                break
            distances = np.abs(token_counts[available_codes] - int(count_value))
            jitter = rng.random(available_codes.size) * 1e-6
            order = np.argsort(distances + jitter)
            selected = available_codes[order[: min(remaining_need, available_codes.size)]]
            chosen_parts.append(selected)
            available[selected] = False
        if chosen_parts:
            chosen = np.concatenate(chosen_parts)
            if chosen.size > k:
                chosen = rng.choice(chosen, size=k, replace=False)
            mask = np.zeros(vocab_size, dtype=bool)
            mask[chosen] = True
            out[f"freqmatched_{source}_{control_idx:02d}"] = mask
    return out


def summarize_paper_protocol(protocol: dict[str, Any], *, alpha: float) -> dict[str, Any]:
    out: dict[str, Any] = {
        "alpha": float(alpha),
        "any_rejected_count": int(np.sum(protocol["any_rejected"])),
        "manifold_any_rejected_count": int(np.sum(protocol["manifold_any_rejected"])),
        "fiber_any_rejected_count": int(np.sum(protocol["fiber_any_rejected"])),
        "min_raw_pvalue": min_or_nan(protocol["best_raw_pvalue"]),
        "min_adjusted_pvalue": min_or_nan(protocol["best_adjusted_pvalue"]),
        "radii": {},
    }
    for label, radius in protocol["radii"].items():
        out["radii"][label] = {
            "vol_min": int(radius["vol_min"]),
            "vol_max": int(radius["vol_max"]),
            "manifold_rejected_count": int(np.sum(radius["manifold_rejected"])),
            "fiber_rejected_count": int(np.sum(radius["fiber_rejected"])),
            "min_manifold_pvalue": min_or_nan(radius["manifold_pvalues"]),
            "min_fiber_pvalue": min_or_nan(radius["fiber_pvalues"]),
            "min_manifold_adjusted_pvalue": min_or_nan(radius["manifold_adjusted_pvalues"]),
            "min_fiber_adjusted_pvalue": min_or_nan(radius["fiber_adjusted_pvalues"]),
            "stratification_summary": radius["summary"],
        }
    return out


def positive_dimension_jump(result: dict[str, Any]) -> float:
    dims = [float(v) for v in result.get("dimensions", []) if v is not None]
    if len(dims) < 2:
        return 0.0
    jumps = [dims[idx + 1] - dims[idx] for idx in range(len(dims) - 1)]
    return float(max(0.0, max(jumps)))


def top_fraction_mask(
    scores: np.ndarray,
    *,
    fraction: float,
    min_count: int,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    mask = np.zeros(arr.shape, dtype=bool)
    if groups is None:
        finite = np.flatnonzero(np.isfinite(arr))
        if finite.size == 0:
            return mask
        count = min(finite.size, max(int(min_count), int(math.ceil(float(fraction) * finite.size))))
        order = finite[np.argsort(arr[finite])]
        mask[order[-count:]] = True
        return mask

    group_arr = np.asarray(groups)
    for group in np.unique(group_arr):
        idx = np.flatnonzero((group_arr == group) & np.isfinite(arr))
        if idx.size == 0:
            continue
        count = max(1, int(math.ceil(float(fraction) * idx.size)))
        order = idx[np.argsort(arr[idx])]
        mask[order[-count:]] = True
    if int(mask.sum()) < int(min_count):
        mask |= top_fraction_mask(arr, fraction=fraction, min_count=min_count)
    return mask


def knn_target_entropy(features: np.ndarray, labels: np.ndarray, *, k: int = 16) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("features must be rank-2 and labels must match rows")
    sq_norms = np.sum(x * x, axis=1, keepdims=True)
    d2 = np.maximum(sq_norms + sq_norms.T - 2.0 * (x @ x.T), 0.0)
    np.fill_diagonal(d2, np.inf)
    take = min(max(2, int(k)), max(1, x.shape[0] - 1))
    nn = np.argpartition(d2, take - 1, axis=1)[:, :take]
    out = np.zeros(x.shape[0], dtype=np.float64)
    denom = math.log(take)
    for idx, neighbors in enumerate(nn):
        _codes, counts = np.unique(y[neighbors], return_counts=True)
        probs = counts.astype(np.float64) / max(1, int(counts.sum()))
        entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
        out[idx] = entropy / max(denom, 1e-12)
    return out


def evaluate_singular_mask(
    *,
    name: str,
    mask: np.ndarray,
    branch_ks: np.ndarray,
    branch_entropy: np.ndarray,
    ranked_ks: np.ndarray,
    permuted_ks: np.ndarray,
    local_ball_ks: np.ndarray,
    local_ball_entropy: np.ndarray,
    local_ball_mass: np.ndarray,
    embedding_ball_ks: np.ndarray,
    embedding_ball_dimension: np.ndarray,
    embedding_ball_radius: np.ndarray,
    flat_mask: np.ndarray,
    local_ball_flat_mask: np.ndarray,
    embedding_ball_flat_mask: np.ndarray,
    permutation_reps: int,
    seed: int,
) -> dict[str, Any]:
    singular = np.asarray(mask, dtype=bool)
    regular = ~singular
    if int(singular.sum()) == 0 or int(regular.sum()) == 0:
        return {"name": name, "count": int(singular.sum()), "valid": False}
    branch_diff, branch_p = permutation_mean_diff_pvalue(
        branch_ks[singular], branch_ks[regular], reps=permutation_reps, seed=seed
    )
    entropy_diff, entropy_p = permutation_mean_diff_pvalue(
        branch_entropy[singular], branch_entropy[regular], reps=permutation_reps, seed=seed + 1
    )
    ranked_diff, ranked_p = permutation_mean_diff_pvalue(
        ranked_ks[singular], ranked_ks[regular], reps=permutation_reps, seed=seed + 2
    )
    perm_diff, perm_p = permutation_mean_diff_pvalue(
        permuted_ks[singular], permuted_ks[regular], reps=permutation_reps, seed=seed + 3
    )
    local_ks_diff, local_ks_p = permutation_mean_diff_pvalue(
        local_ball_ks[singular], local_ball_ks[regular], reps=permutation_reps, seed=seed + 4
    )
    local_entropy_diff, local_entropy_p = permutation_mean_diff_pvalue(
        local_ball_entropy[singular], local_ball_entropy[regular], reps=permutation_reps, seed=seed + 5
    )
    local_mass_diff, local_mass_p = permutation_mean_diff_pvalue(
        local_ball_mass[singular], local_ball_mass[regular], reps=permutation_reps, seed=seed + 6
    )
    embedding_ks_diff, embedding_ks_p = permutation_mean_diff_pvalue(
        embedding_ball_ks[singular], embedding_ball_ks[regular], reps=permutation_reps, seed=seed + 7
    )
    embedding_dim_diff, embedding_dim_p = permutation_mean_diff_pvalue(
        embedding_ball_dimension[singular],
        embedding_ball_dimension[regular],
        reps=permutation_reps,
        seed=seed + 8,
    )
    embedding_radius_diff, embedding_radius_p = permutation_mean_diff_pvalue(
        embedding_ball_radius[singular],
        embedding_ball_radius[regular],
        reps=permutation_reps,
        seed=seed + 9,
    )
    flat_rate_diff, flat_rate_p = permutation_mean_diff_pvalue(
        singular[flat_mask].astype(np.float64),
        singular[~flat_mask].astype(np.float64),
        reps=permutation_reps,
        seed=seed + 10,
    )
    local_flat_rate_diff, local_flat_rate_p = permutation_mean_diff_pvalue(
        singular[local_ball_flat_mask].astype(np.float64),
        singular[~local_ball_flat_mask].astype(np.float64),
        reps=permutation_reps,
        seed=seed + 11,
    )
    embedding_flat_rate_diff, embedding_flat_rate_p = permutation_mean_diff_pvalue(
        singular[embedding_ball_flat_mask].astype(np.float64),
        singular[~embedding_ball_flat_mask].astype(np.float64),
        reps=permutation_reps,
        seed=seed + 12,
    )
    return {
        "name": name,
        "valid": True,
        "count": int(singular.sum()),
        "fraction": float(np.mean(singular)),
        "mean_branch_ks_singular": mean_or_nan(branch_ks[singular]),
        "mean_branch_ks_regular": mean_or_nan(branch_ks[regular]),
        "branch_ks_singular_minus_regular": branch_diff,
        "branch_ks_permutation_p": branch_p,
        "branch_ks_cohen_d": cohen_d(branch_ks[singular], branch_ks[regular]),
        "mean_branch_entropy_singular": mean_or_nan(branch_entropy[singular]),
        "mean_branch_entropy_regular": mean_or_nan(branch_entropy[regular]),
        "branch_entropy_singular_minus_regular": entropy_diff,
        "branch_entropy_permutation_p": entropy_p,
        "mean_ranked_ks_singular": mean_or_nan(ranked_ks[singular]),
        "mean_ranked_ks_regular": mean_or_nan(ranked_ks[regular]),
        "ranked_ks_singular_minus_regular": ranked_diff,
        "ranked_ks_permutation_p": ranked_p,
        "mean_permuted_ks_singular": mean_or_nan(permuted_ks[singular]),
        "mean_permuted_ks_regular": mean_or_nan(permuted_ks[regular]),
        "permuted_ks_singular_minus_regular": perm_diff,
        "permuted_ks_permutation_p": perm_p,
        "mean_local_ball_ks_singular": mean_or_nan(local_ball_ks[singular]),
        "mean_local_ball_ks_regular": mean_or_nan(local_ball_ks[regular]),
        "local_ball_ks_singular_minus_regular": local_ks_diff,
        "local_ball_ks_permutation_p": local_ks_p,
        "mean_local_ball_entropy_singular": mean_or_nan(local_ball_entropy[singular]),
        "mean_local_ball_entropy_regular": mean_or_nan(local_ball_entropy[regular]),
        "local_ball_entropy_singular_minus_regular": local_entropy_diff,
        "local_ball_entropy_permutation_p": local_entropy_p,
        "mean_local_ball_mass_singular": mean_or_nan(local_ball_mass[singular]),
        "mean_local_ball_mass_regular": mean_or_nan(local_ball_mass[regular]),
        "local_ball_mass_singular_minus_regular": local_mass_diff,
        "local_ball_mass_permutation_p": local_mass_p,
        "mean_embedding_ball_ks_singular": mean_or_nan(embedding_ball_ks[singular]),
        "mean_embedding_ball_ks_regular": mean_or_nan(embedding_ball_ks[regular]),
        "embedding_ball_ks_singular_minus_regular": embedding_ks_diff,
        "embedding_ball_ks_permutation_p": embedding_ks_p,
        "mean_embedding_ball_dimension_singular": mean_or_nan(embedding_ball_dimension[singular]),
        "mean_embedding_ball_dimension_regular": mean_or_nan(embedding_ball_dimension[regular]),
        "embedding_ball_dimension_singular_minus_regular": embedding_dim_diff,
        "embedding_ball_dimension_permutation_p": embedding_dim_p,
        "mean_embedding_ball_radius_singular": mean_or_nan(embedding_ball_radius[singular]),
        "mean_embedding_ball_radius_regular": mean_or_nan(embedding_ball_radius[regular]),
        "embedding_ball_radius_singular_minus_regular": embedding_radius_diff,
        "embedding_ball_radius_permutation_p": embedding_radius_p,
        "flat_singular_rate": mean_or_nan(singular[flat_mask].astype(np.float64)),
        "rest_singular_rate": mean_or_nan(singular[~flat_mask].astype(np.float64)),
        "flat_singular_rate_minus_rest": flat_rate_diff,
        "flat_singular_rate_p": flat_rate_p,
        "local_ball_flat_singular_rate": mean_or_nan(singular[local_ball_flat_mask].astype(np.float64)),
        "local_ball_rest_singular_rate": mean_or_nan(singular[~local_ball_flat_mask].astype(np.float64)),
        "local_ball_flat_singular_rate_minus_rest": local_flat_rate_diff,
        "local_ball_flat_singular_rate_p": local_flat_rate_p,
        "embedding_ball_flat_singular_rate": mean_or_nan(
            singular[embedding_ball_flat_mask].astype(np.float64)
        ),
        "embedding_ball_rest_singular_rate": mean_or_nan(
            singular[~embedding_ball_flat_mask].astype(np.float64)
        ),
        "embedding_ball_flat_singular_rate_minus_rest": embedding_flat_rate_diff,
        "embedding_ball_flat_singular_rate_p": embedding_flat_rate_p,
    }


def plot_histogram(
    *,
    singular: np.ndarray,
    regular: np.ndarray,
    out_path: Path,
    title: str,
    xlabel: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    singular = np.asarray(singular, dtype=np.float64)
    regular = np.asarray(regular, dtype=np.float64)
    combined = np.concatenate([singular[np.isfinite(singular)], regular[np.isfinite(regular)]])
    if combined.size:
        lo = float(np.nanmin(combined))
        hi = float(np.nanmax(combined))
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = 0.0, 1.0
        if abs(hi - lo) < 1e-12:
            lo -= 0.5
            hi += 0.5
        bins = np.linspace(lo, hi, 36)
        regular_finite = regular[np.isfinite(regular)]
        singular_finite = singular[np.isfinite(singular)]
        regular_weights = np.full(regular_finite.shape, 1.0 / max(1, regular_finite.size), dtype=np.float64)
        singular_weights = np.full(singular_finite.shape, 1.0 / max(1, singular_finite.size), dtype=np.float64)
        ax.hist(
            regular_finite,
            bins=bins,
            weights=regular_weights,
            alpha=0.62,
            label=f"regular n={regular_finite.size}",
            color="#4c78a8",
        )
        ax.hist(
            singular_finite,
            bins=bins,
            weights=singular_weights,
            alpha=0.70,
            label=f"singular n={singular_finite.size}",
            color="#f58518",
        )
        regular_mean = mean_or_nan(regular)
        singular_mean = mean_or_nan(singular)
        if math.isfinite(regular_mean):
            ax.axvline(regular_mean, color="#4c78a8", linestyle="--", linewidth=2)
        if math.isfinite(singular_mean):
            ax.axvline(singular_mean, color="#f58518", linestyle="--", linewidth=2)
    else:
        ax.text(0.5, 0.5, "no finite values", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction within group")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = save_figure(fig, out_path, dpi=180)
    plt.close(fig)
    return path


def plot_radius_uniformity_curve(
    *,
    volumes: np.ndarray,
    ks_by_volume: np.ndarray,
    singular: np.ndarray,
    out_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    volumes = np.asarray(volumes, dtype=np.float64)
    ks = np.asarray(ks_by_volume, dtype=np.float64)
    mask = np.asarray(singular, dtype=bool)
    if ks.ndim != 2 or ks.shape[1] != volumes.size or mask.size != ks.shape[0]:
        ax.text(0.5, 0.5, "invalid curve inputs", ha="center", va="center", transform=ax.transAxes)
    elif int(mask.sum()) == 0 or int((~mask).sum()) == 0:
        ax.text(0.5, 0.5, "missing singular or regular group", ha="center", va="center", transform=ax.transAxes)
    else:
        singular_mean = np.nanmean(ks[mask], axis=0)
        regular_mean = np.nanmean(ks[~mask], axis=0)
        ax.plot(volumes, regular_mean, marker="o", linewidth=2, color="#4c78a8", label="regular")
        ax.plot(volumes, singular_mean, marker="o", linewidth=2, color="#f58518", label="singular")
        ax.fill_between(
            volumes,
            np.nanpercentile(ks[~mask], 25, axis=0),
            np.nanpercentile(ks[~mask], 75, axis=0),
            color="#4c78a8",
            alpha=0.12,
            linewidth=0,
        )
        ax.fill_between(
            volumes,
            np.nanpercentile(ks[mask], 25, axis=0),
            np.nanpercentile(ks[mask], 75, axis=0),
            color="#f58518",
            alpha=0.14,
            linewidth=0,
        )
        ax.legend(frameon=False)
    ax.set_title("Radius-threshold sweep for embedding-ball uniformity")
    ax.set_xlabel("neighbor volume defining radius")
    ax.set_ylabel("radial KS to Uniform(0,1), lower is flatter")
    fig.tight_layout()
    path = save_figure(fig, out_path, dpi=180)
    plt.close(fig)
    return path


def plot_scatter(
    *,
    irregularity: np.ndarray,
    branch_ks: np.ndarray,
    singular: np.ndarray,
    out_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    finite = irregularity[np.isfinite(irregularity)]
    cap = float(np.quantile(finite, 0.99)) if finite.size else 1.0
    x = np.minimum(irregularity, cap)
    ax.scatter(x[~singular], branch_ks[~singular], s=16, alpha=0.42, color="#4c78a8", label="regular")
    ax.scatter(x[singular], branch_ks[singular], s=24, alpha=0.78, color="#f58518", label="singular")
    ax.set_xlabel(f"fiber irregularity (clipped at {cap:.1f})")
    ax.set_ylabel("top-k branch KS to uniform (lower is flatter)")
    ax.set_title("Singularity vs next patch-token branch flatness")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = save_figure(fig, out_path, dpi=180)
    plt.close(fig)
    return path


def plot_heatmaps(
    *,
    branch_ks: np.ndarray,
    irregularity: np.ndarray,
    singular: np.ndarray,
    batch: int,
    grid: int,
    out_path: Path,
) -> Path:
    cols = min(4, int(batch))
    rows = int(math.ceil(batch / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols + 0.7, 4.0 * rows), squeeze=False)
    finite = branch_ks[np.isfinite(branch_ks)]
    vmin = float(np.quantile(finite, 0.02)) if finite.size else 0.0
    vmax = float(np.quantile(finite, 0.98)) if finite.size else 1.0
    im = None
    for sample_id in range(batch):
        ax = axes.ravel()[sample_id]
        start = sample_id * grid * grid
        stop = start + grid * grid
        values = branch_ks[start:stop].reshape(grid, grid)
        irr = irregularity[start:stop].reshape(grid, grid)
        mask = singular[start:stop].reshape(grid, grid)
        im = ax.imshow(values, cmap="viridis_r", vmin=vmin, vmax=vmax)
        if mask.any():
            ax.contour(mask.astype(float), levels=[0.5], colors=["#ff3b30"], linewidths=1.5)
        if np.isfinite(irr).any():
            high = irr >= np.nanquantile(irr[np.isfinite(irr)], 0.90)
            ax.contour(high.astype(float), levels=[0.5], colors=["white"], linewidths=0.7, alpha=0.7)
        ax.set_title(f"sample {sample_id}")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[batch:]:
        ax.axis("off")
    if im is not None:
        fig.subplots_adjust(right=0.90, top=0.86, bottom=0.08, wspace=0.28, hspace=0.35)
        cbar_ax = fig.add_axes([0.925, 0.17, 0.014, 0.62])
        fig.colorbar(im, cax=cbar_ax, label="top-k branch KS (lower = flatter)")
    fig.suptitle("Next patch-token branch flatness with singular contours", y=0.99)
    path = save_figure(fig, out_path, dpi=180)
    plt.close(fig)
    return path


def plot_sensitivity(
    *,
    rows: list[dict[str, Any]],
    out_path: Path,
) -> Path:
    valid = [row for row in rows if row.get("valid")]
    if not valid:
        raise ValueError("no valid sensitivity rows to plot")
    names = [str(row["name"]) for row in valid]
    branch_diff = np.asarray([float(row["branch_ks_singular_minus_regular"]) for row in valid], dtype=np.float64)
    entropy_diff = np.asarray([float(row["branch_entropy_singular_minus_regular"]) for row in valid], dtype=np.float64)
    counts = np.asarray([int(row["count"]) for row in valid], dtype=np.int64)
    x = np.arange(len(valid))
    fig, axes = plt.subplots(1, 2, figsize=(max(9.0, 1.15 * len(valid) + 5.0), 4.8))
    colors = np.where(branch_diff < 0.0, "#54a24b", "#e45756")
    axes[0].bar(x, branch_diff, color=colors, alpha=0.82)
    axes[0].axhline(0.0, color="#333333", linewidth=1)
    axes[0].set_ylabel("singular - regular top-k KS")
    axes[0].set_title("Negative supports uniform-polysemy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=35, ha="right")
    for idx, count in enumerate(counts):
        axes[0].text(idx, branch_diff[idx], f"n={count}", ha="center", va="bottom" if branch_diff[idx] >= 0 else "top", fontsize=8)

    colors = np.where(entropy_diff > 0.0, "#54a24b", "#e45756")
    axes[1].bar(x, entropy_diff, color=colors, alpha=0.82)
    axes[1].axhline(0.0, color="#333333", linewidth=1)
    axes[1].set_ylabel("singular - regular branch entropy")
    axes[1].set_title("Positive supports uniform-polysemy")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=35, ha="right")
    fig.suptitle("Sensitivity to singular-token definition", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = save_figure(fig, out_path, dpi=180)
    plt.close(fig)
    return path


def decode_sample_grid_if_requested(
    *,
    profile: dict[str, Any],
    repo_path: Path,
    tokens: torch.Tensor,
    class_labels: list[int],
    device: torch.device,
    out_path: Path,
) -> Path | None:
    try:
        vq_path = hf_hub_download(repo_id=profile["repo_id"], filename=profile["vq_file"])
        latent_size = int(profile["image_size"]) // int(profile["downsample_size"])
        with llamagen_import_context(repo_path):
            from tokenizer.tokenizer_image.vq_model import VQ_models

            vq_model = VQ_models[profile["vq_model"]](
                codebook_size=int(profile["codebook_size"]),
                codebook_embed_dim=int(profile["codebook_embed_dim"]),
            ).to(device)
            vq_model.load_state_dict(load_weight_payload(vq_path), strict=True)
            vq_model.eval()
            qzshape = [tokens.shape[0], int(profile["codebook_embed_dim"]), latent_size, latent_size]
            samples = vq_model.decode_code(tokens.to(device=device, dtype=torch.long), qzshape)
        return save_grid(samples, out_path, labels=class_labels, title="Teacher-forced LlamaGen token samples")
    except Exception as exc:
        print(f"[warn] could not decode sample grid: {exc}", flush=True)
        return None


def load_class_labels_file(path: Path, *, samples: int) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("class_labels"), list):
            labels = [int(value) for value in payload["class_labels"]]
        elif isinstance(payload, dict) and payload.get("records"):
            labels = load_class_labels_file(Path(payload["records"]), samples=samples)
        elif isinstance(payload, list):
            labels = []
            for row in payload:
                if not isinstance(row, dict):
                    raise ValueError(f"bad label record in {path}: {row!r}")
                if "class_label" in row:
                    labels.append(int(row["class_label"]))
                elif "label" in row:
                    labels.append(int(row["label"]))
                else:
                    raise ValueError(f"missing class_label/label in {path}: {row!r}")
        else:
            raise ValueError(f"could not read class labels from JSON file {path}")
    else:
        labels = []
        with path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                if "class_label" in row:
                    labels.append(int(row["class_label"]))
                elif "label" in row:
                    labels.append(int(row["label"]))
                else:
                    raise ValueError(f"missing class_label/label column in {path}")
    if len(labels) < int(samples):
        raise ValueError(f"{path} contains {len(labels)} labels but {samples} samples are required")
    return labels[: int(samples)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_path = resolve_llamagen_repo(args.llamagen_repo or None)
    device = resolve_device(args.device)
    dtype = resolve_precision(args.precision, device)
    tokens = torch.load(args.tokens_path, map_location="cpu")
    if isinstance(tokens, dict):
        tokens = tokens.get("tokens", tokens.get("index_sample"))
    if not isinstance(tokens, torch.Tensor):
        raise ValueError("--tokens-path must point to a tensor or dict containing tokens")
    tokens = tokens.long().cpu()
    if int(args.max_samples) > 0:
        tokens = tokens[: int(args.max_samples)]
    if args.class_labels_file:
        class_labels = load_class_labels_file(Path(args.class_labels_file), samples=int(tokens.shape[0]))
    else:
        class_labels = parse_class_labels(args.class_labels, samples=int(tokens.shape[0]))

    model, profile, missing, unexpected = load_llamagen_gpt(
        profile_name=args.profile,
        repo_path=repo_path,
        device=device,
        dtype=dtype,
    )
    logits, hidden = teacher_force_logits_and_hidden(model, tokens, class_labels, device=device)
    probs = F.softmax(logits, dim=-1).numpy().reshape(-1, logits.shape[-1])
    hidden_flat = hidden.reshape(-1, hidden.shape[-1])
    tokens_flat = tokens.reshape(-1).numpy()
    codebook_code_masks = load_codebook_singular_code_masks(
        args.codebook_singular_codes_path,
        vocab_size=int(profile["codebook_size"]),
    )
    codebook_control_source = args.codebook_control_source or args.codebook_singular_source
    codebook_control_code_masks = build_codebook_control_code_masks(
        codebook_code_masks,
        tokens.numpy(),
        source=codebook_control_source,
        random_controls=args.codebook_random_controls,
        frequency_controls=args.codebook_frequency_controls,
        seed=args.seed + 1000,
    )
    codebook_position_masks = codebook_position_masks_from_tokens(codebook_code_masks, tokens.numpy())
    codebook_control_position_masks = codebook_position_masks_from_tokens(codebook_control_code_masks, tokens.numpy())
    grid = int(round(math.sqrt(int(tokens.shape[1]))))
    if grid * grid != int(tokens.shape[1]):
        raise ValueError("token sequence length must be a square patch grid")

    geom_features = pca_project(hidden_flat, dims=args.geometry_pca_dim)
    paper_protocol = run_paper_protocol(
        geom_features,
        small_vol_min=args.paper_small_vol_min,
        small_vol_max=args.paper_small_vol_max,
        large_vol_min=args.paper_large_vol_min,
        large_vol_max=args.paper_large_vol_max,
        window_size=args.window_size,
        alpha=args.paper_alpha,
        nstrat=args.nstrat,
        geometry_mode=args.paper_geometry,
    )
    paper_summary = summarize_paper_protocol(paper_protocol, alpha=args.paper_alpha)
    small_radius = paper_protocol["radii"]["small"]
    large_radius = paper_protocol["radii"]["large"]
    fiber_results = small_radius["results"]
    fiber_summary = small_radius["summary"]
    pvalues = small_radius["fiber_pvalues"]
    fiber_adjusted_pvalues = small_radius["fiber_adjusted_pvalues"]
    manifold_pvalues = small_radius["manifold_pvalues"]
    manifold_adjusted_pvalues = small_radius["manifold_adjusted_pvalues"]
    legacy_irregularity = np.where(np.isfinite(pvalues), -np.log10(np.maximum(pvalues, 1e-300)), 0.0)
    legacy_raw_rejected = np.isfinite(pvalues) & (pvalues < float(args.alpha))
    legacy_singular_mask, legacy_singular_selection = select_singular_mask(
        pvalues=pvalues,
        irregularity=legacy_irregularity,
        alpha=args.alpha,
        fraction=args.singular_fraction,
        min_count=args.min_singular,
    )
    singular_mask, singular_selection = choose_paper_singular_mask(paper_protocol, args.singular_source)
    if args.use_codebook_singular_as_active:
        active_key = f"codebook_{args.codebook_active_position}_{args.codebook_singular_source}"
        if active_key not in codebook_position_masks:
            raise ValueError(
                f"requested active codebook mask {active_key!r}, but available masks are "
                f"{sorted(codebook_position_masks)}"
            )
        singular_mask = np.asarray(codebook_position_masks[active_key], dtype=bool)
        singular_selection = active_key
    irregularity = np.asarray(paper_protocol["irregularity"], dtype=np.float64)
    rejected = singular_mask

    ranked_ks = ranked_probability_uniform_ks(probs)
    branch_ks, branch_entropy, branch_top_mass, branch_top_indices = topk_branch_uniform_ks(probs, top_k=args.branch_top_k)
    permuted = permuted_categorical_uniform_ks(
        probs,
        permutations=args.permuted_ks,
        seed=args.seed,
    )
    geom_np = geom_features.numpy()
    local_ball = local_ball_uniformity_ks(
        features=geom_np,
        probs=probs,
        target_codes=tokens_flat,
        volume=args.local_ball_volume,
        exclude_self=not args.local_ball_include_self,
    )
    local_ball_ks = local_ball["ks"]
    local_ball_entropy = local_ball["entropy"]
    local_ball_mass = local_ball["mass"]
    embedding_ball = embedding_ball_radial_uniformity_ks(
        features=geom_np,
        volume=args.embedding_ball_volume,
        exclude_self=not args.embedding_ball_include_self,
        dimension_trim=args.embedding_ball_trim,
        min_inner=args.embedding_ball_min_inner,
    )
    embedding_ball_ks = embedding_ball["ks"]
    embedding_ball_pvalue = embedding_ball["pvalue"]
    embedding_ball_dimension = embedding_ball["dimension"]
    embedding_ball_radius = embedding_ball["radius"]
    embedding_radius = embedding_radius_uniformity_threshold(
        features=geom_np,
        volume_min=args.embedding_radius_volume_min,
        volume_max=args.embedding_radius_volume_max,
        volume_step=args.embedding_radius_volume_step,
        uniform_pvalue=args.embedding_radius_uniform_pvalue,
        max_ks=args.embedding_radius_max_ks,
        consecutive=args.embedding_radius_consecutive,
        exclude_self=not args.embedding_ball_include_self,
        dimension_trim=args.embedding_ball_trim,
        min_inner=args.embedding_ball_min_inner,
    )
    embedding_radius_found = embedding_radius["threshold_found"].astype(bool)
    embedding_radius_threshold_radius = embedding_radius["threshold_radius"]
    embedding_radius_threshold_volume = embedding_radius["threshold_volume"]
    embedding_radius_threshold_ks = embedding_radius["threshold_ks"]
    embedding_radius_best_radius = embedding_radius["best_radius"]
    embedding_radius_best_volume = embedding_radius["best_volume"]
    embedding_radius_best_ks = embedding_radius["best_ks"]

    regular_mask = ~singular_mask
    branch_diff, branch_p = permutation_mean_diff_pvalue(
        branch_ks[singular_mask],
        branch_ks[regular_mask],
        reps=args.permutation_reps,
        seed=args.seed + 1,
    )
    entropy_diff, entropy_p = permutation_mean_diff_pvalue(
        branch_entropy[singular_mask],
        branch_entropy[regular_mask],
        reps=args.permutation_reps,
        seed=args.seed + 2,
    )
    ranked_diff, ranked_p = permutation_mean_diff_pvalue(
        ranked_ks[singular_mask],
        ranked_ks[regular_mask],
        reps=args.permutation_reps,
        seed=args.seed + 3,
    )
    perm_diff, perm_p = permutation_mean_diff_pvalue(
        permuted["trimmed_mean"][singular_mask],
        permuted["trimmed_mean"][regular_mask],
        reps=args.permutation_reps,
        seed=args.seed + 4,
    )
    local_ball_ks_diff, local_ball_ks_p = permutation_mean_diff_pvalue(
        local_ball_ks[singular_mask],
        local_ball_ks[regular_mask],
        reps=args.permutation_reps,
        seed=args.seed + 5,
    )
    local_ball_entropy_diff, local_ball_entropy_p = permutation_mean_diff_pvalue(
        local_ball_entropy[singular_mask],
        local_ball_entropy[regular_mask],
        reps=args.permutation_reps,
        seed=args.seed + 6,
    )
    local_ball_mass_diff, local_ball_mass_p = permutation_mean_diff_pvalue(
        local_ball_mass[singular_mask],
        local_ball_mass[regular_mask],
        reps=args.permutation_reps,
        seed=args.seed + 7,
    )
    embedding_ball_ks_diff, embedding_ball_ks_p = permutation_mean_diff_pvalue(
        embedding_ball_ks[singular_mask],
        embedding_ball_ks[regular_mask],
        reps=args.permutation_reps,
        seed=args.seed + 8,
    )
    embedding_ball_dimension_diff, embedding_ball_dimension_p = permutation_mean_diff_pvalue(
        embedding_ball_dimension[singular_mask],
        embedding_ball_dimension[regular_mask],
        reps=args.permutation_reps,
        seed=args.seed + 9,
    )
    embedding_ball_radius_diff, embedding_ball_radius_p = permutation_mean_diff_pvalue(
        embedding_ball_radius[singular_mask],
        embedding_ball_radius[regular_mask],
        reps=args.permutation_reps,
        seed=args.seed + 10,
    )
    embedding_radius_found_diff, embedding_radius_found_p = permutation_mean_diff_pvalue(
        embedding_radius_found[singular_mask].astype(np.float64),
        embedding_radius_found[regular_mask].astype(np.float64),
        reps=args.permutation_reps,
        seed=args.seed + 11,
    )
    embedding_radius_threshold_diff, embedding_radius_threshold_p = permutation_mean_diff_pvalue(
        embedding_radius_threshold_radius[singular_mask],
        embedding_radius_threshold_radius[regular_mask],
        reps=args.permutation_reps,
        seed=args.seed + 12,
    )
    embedding_radius_volume_diff, embedding_radius_volume_p = permutation_mean_diff_pvalue(
        embedding_radius_threshold_volume[singular_mask],
        embedding_radius_threshold_volume[regular_mask],
        reps=args.permutation_reps,
        seed=args.seed + 13,
    )
    embedding_radius_best_ks_diff, embedding_radius_best_ks_p = permutation_mean_diff_pvalue(
        embedding_radius_best_ks[singular_mask],
        embedding_radius_best_ks[regular_mask],
        reps=args.permutation_reps,
        seed=args.seed + 14,
    )

    flat_mask = branch_ks <= np.nanquantile(branch_ks, float(args.flat_quantile))
    local_ball_flat_mask = local_ball_ks <= np.nanquantile(local_ball_ks, float(args.flat_quantile))
    embedding_ball_flat_mask = embedding_ball_ks <= np.nanquantile(embedding_ball_ks, float(args.flat_quantile))
    flat_singular_rate = mean_or_nan(singular_mask[flat_mask].astype(np.float64))
    rest_singular_rate = mean_or_nan(singular_mask[~flat_mask].astype(np.float64))
    flat_rate_diff, flat_rate_p = permutation_mean_diff_pvalue(
        singular_mask[flat_mask].astype(np.float64),
        singular_mask[~flat_mask].astype(np.float64),
        reps=args.permutation_reps,
        seed=args.seed + 15,
    )
    embedding_ball_flat_singular_rate = mean_or_nan(singular_mask[embedding_ball_flat_mask].astype(np.float64))
    embedding_ball_rest_singular_rate = mean_or_nan(singular_mask[~embedding_ball_flat_mask].astype(np.float64))
    embedding_ball_flat_rate_diff, embedding_ball_flat_rate_p = permutation_mean_diff_pvalue(
        singular_mask[embedding_ball_flat_mask].astype(np.float64),
        singular_mask[~embedding_ball_flat_mask].astype(np.float64),
        reps=args.permutation_reps,
        seed=args.seed + 16,
    )
    sample_ids = np.repeat(np.arange(int(tokens.shape[0])), int(tokens.shape[1]))
    dim_jump = np.asarray([positive_dimension_jump(result) for result in fiber_results], dtype=np.float64)
    knn_entropy = knn_target_entropy(geom_np, tokens_flat, k=args.knn_entropy_k)
    sensitivity_masks = {
        "active_selector": singular_mask,
        "paper_holm_any": paper_protocol["any_rejected"],
        "paper_holm_stratified_any": paper_protocol["any_rejected"],
        "paper_holm_manifold_any": paper_protocol["manifold_any_rejected"],
        "paper_holm_fiber_any": paper_protocol["fiber_any_rejected"],
        "paper_holm_stratified_small": small_radius["manifold_rejected"] | small_radius["fiber_rejected"],
        "paper_holm_stratified_large": large_radius["manifold_rejected"] | large_radius["fiber_rejected"],
        "paper_holm_manifold_small": small_radius["manifold_rejected"],
        "paper_holm_fiber_small": small_radius["fiber_rejected"],
        "paper_holm_manifold_large": large_radius["manifold_rejected"],
        "paper_holm_fiber_large": large_radius["fiber_rejected"],
        f"legacy_{legacy_singular_selection}": legacy_singular_mask,
        "legacy_raw_fiber_pvalue_small": legacy_raw_rejected,
        "top_paper_irregularity_global": top_fraction_mask(
            irregularity,
            fraction=args.singular_fraction,
            min_count=args.min_singular,
        ),
        "top_paper_irregularity_per_sample": top_fraction_mask(
            irregularity,
            fraction=args.singular_fraction,
            min_count=args.min_singular,
            groups=sample_ids,
        ),
        "top_dimension_jump_small": top_fraction_mask(
            dim_jump,
            fraction=args.singular_fraction,
            min_count=args.min_singular,
        ),
        "top_knn_target_entropy": top_fraction_mask(
            knn_entropy,
            fraction=args.singular_fraction,
            min_count=args.min_singular,
        ),
    }
    sensitivity_masks.update(codebook_position_masks)
    sensitivity_masks.update(codebook_control_position_masks)
    sensitivity_rows = [
        evaluate_singular_mask(
            name=name,
            mask=mask,
            branch_ks=branch_ks,
            branch_entropy=branch_entropy,
            ranked_ks=ranked_ks,
            permuted_ks=permuted["trimmed_mean"],
            local_ball_ks=local_ball_ks,
            local_ball_entropy=local_ball_entropy,
            local_ball_mass=local_ball_mass,
            embedding_ball_ks=embedding_ball_ks,
            embedding_ball_dimension=embedding_ball_dimension,
            embedding_ball_radius=embedding_ball_radius,
            flat_mask=flat_mask,
            local_ball_flat_mask=local_ball_flat_mask,
            embedding_ball_flat_mask=embedding_ball_flat_mask,
            permutation_reps=args.permutation_reps,
            seed=args.seed + 100 + idx * 17,
        )
        for idx, (name, mask) in enumerate(sensitivity_masks.items())
    ]

    figures = {
        "branch_ks_hist": str(plot_histogram(
            singular=branch_ks[singular_mask],
            regular=branch_ks[regular_mask],
            out_path=out_dir / "vq_ar_branch_ks_singular_vs_regular.png",
            title="Robust top-k KS: singular vs regular next patch predictions",
            xlabel=f"top-{args.branch_top_k} branch KS to uniform (lower = flatter)",
        )),
        "ranked_ks_hist": str(plot_histogram(
            singular=ranked_ks[singular_mask],
            regular=ranked_ks[regular_mask],
            out_path=out_dir / "vq_ar_ranked_ks_singular_vs_regular.png",
            title="Full-vocabulary ranked KS: singular vs regular",
            xlabel="ranked-probability KS to uniform (lower = flatter)",
        )),
        "local_ball_ks_hist": str(plot_histogram(
            singular=local_ball_ks[singular_mask],
            regular=local_ball_ks[regular_mask],
            out_path=out_dir / "vq_ar_local_ball_ks_singular_vs_regular.png",
            title="Fixed-volume local-ball KS: singular vs regular",
            xlabel=f"local ball volume {args.local_ball_volume} KS to uniform (lower = flatter)",
        )),
        "embedding_ball_ks_hist": str(plot_histogram(
            singular=embedding_ball_ks[singular_mask],
            regular=embedding_ball_ks[regular_mask],
            out_path=out_dir / "vq_ar_embedding_ball_ks_singular_vs_regular.png",
            title="Embedding-ball radial KS: singular vs regular",
            xlabel=f"embedding ball volume {args.embedding_ball_volume} radial KS to uniform (lower = flatter)",
        )),
        "embedding_radius_threshold_hist": str(plot_histogram(
            singular=embedding_radius_threshold_radius[singular_mask],
            regular=embedding_radius_threshold_radius[regular_mask],
            out_path=out_dir / "vq_ar_embedding_radius_threshold_singular_vs_regular.png",
            title="First near-uniform embedding radius: singular vs regular",
            xlabel="first near-uniform radius (lower = earlier threshold)",
        )),
        "embedding_radius_best_ks_hist": str(plot_histogram(
            singular=embedding_radius_best_ks[singular_mask],
            regular=embedding_radius_best_ks[regular_mask],
            out_path=out_dir / "vq_ar_embedding_radius_best_ks_singular_vs_regular.png",
            title="Best embedding-ball radial KS over radius sweep",
            xlabel="best radial KS across swept radii (lower = flatter)",
        )),
        "embedding_radius_uniformity_curve": str(plot_radius_uniformity_curve(
            volumes=embedding_radius["candidate_volumes"],
            ks_by_volume=embedding_radius["ks_by_volume"],
            singular=singular_mask,
            out_path=out_dir / "vq_ar_embedding_radius_uniformity_curve.png",
        )),
        "scatter": str(plot_scatter(
            irregularity=irregularity,
            branch_ks=branch_ks,
            singular=singular_mask,
            out_path=out_dir / "vq_ar_irregularity_vs_branch_ks.png",
        )),
        "heatmaps": str(plot_heatmaps(
            branch_ks=branch_ks,
            irregularity=irregularity,
            singular=singular_mask,
            batch=int(tokens.shape[0]),
            grid=grid,
            out_path=out_dir / "vq_ar_branch_ks_heatmaps.png",
        )),
        "singular_sensitivity": str(plot_sensitivity(
            rows=sensitivity_rows,
            out_path=out_dir / "vq_ar_singular_definition_sensitivity.png",
        )),
    }
    decoded_grid = None
    if args.decode_grid:
        decoded_grid = decode_sample_grid_if_requested(
            profile=profile,
            repo_path=repo_path,
            tokens=tokens,
            class_labels=class_labels,
            device=device,
            out_path=out_dir / "vq_ar_teacher_forced_samples.png",
        )
        if decoded_grid is not None:
            figures["decoded_samples"] = str(decoded_grid)

    records = []
    for idx in range(int(tokens_flat.shape[0])):
        sample_id = idx // (grid * grid)
        patch_id = idx % (grid * grid)
        records.append(
            {
                "token_index": idx,
                "sample_id": sample_id,
                "patch_id": patch_id,
                "row": patch_id // grid,
                "col": patch_id % grid,
                "target_code": int(tokens_flat[idx]),
                "singular": bool(singular_mask[idx]),
                "paper_best_raw_pvalue": float(paper_protocol["best_raw_pvalue"][idx])
                if math.isfinite(float(paper_protocol["best_raw_pvalue"][idx]))
                else None,
                "paper_best_adjusted_pvalue": float(paper_protocol["best_adjusted_pvalue"][idx])
                if math.isfinite(float(paper_protocol["best_adjusted_pvalue"][idx]))
                else None,
                "manifold_rejected": bool(small_radius["manifold_rejected"][idx]),
                "manifold_pvalue": float(manifold_pvalues[idx]) if math.isfinite(float(manifold_pvalues[idx])) else None,
                "manifold_adjusted_pvalue": (
                    float(manifold_adjusted_pvalues[idx])
                    if math.isfinite(float(manifold_adjusted_pvalues[idx]))
                    else None
                ),
                "fiber_rejected": bool(small_radius["fiber_rejected"][idx]),
                "fiber_pvalue": float(pvalues[idx]) if math.isfinite(float(pvalues[idx])) else None,
                "fiber_adjusted_pvalue": (
                    float(fiber_adjusted_pvalues[idx])
                    if math.isfinite(float(fiber_adjusted_pvalues[idx]))
                    else None
                ),
                "paper_small_manifold_delta": float(small_radius["manifold_delta"][idx]),
                "paper_small_fiber_delta": float(small_radius["fiber_delta"][idx]),
                "paper_large_manifold_delta": float(large_radius["manifold_delta"][idx]),
                "paper_large_fiber_delta": float(large_radius["fiber_delta"][idx]),
                "legacy_raw_fiber_rejected": bool(legacy_raw_rejected[idx]),
                "legacy_singular": bool(legacy_singular_mask[idx]),
                "irregularity": float(irregularity[idx]),
                "legacy_fiber_irregularity": float(legacy_irregularity[idx]),
                "ranked_ks": float(ranked_ks[idx]),
                "branch_ks": float(branch_ks[idx]),
                "branch_entropy": float(branch_entropy[idx]),
                "branch_top_mass": float(branch_top_mass[idx]),
                "permuted_ks_trimmed_mean": float(permuted["trimmed_mean"][idx]),
                "local_ball_ks": float(local_ball_ks[idx]) if math.isfinite(float(local_ball_ks[idx])) else None,
                "local_ball_entropy": (
                    float(local_ball_entropy[idx]) if math.isfinite(float(local_ball_entropy[idx])) else None
                ),
                "local_ball_mass": float(local_ball_mass[idx]) if math.isfinite(float(local_ball_mass[idx])) else None,
                "local_ball_unique_count": int(local_ball["unique_count"][idx]),
                "local_ball_radius": float(local_ball["radius"][idx]) if math.isfinite(float(local_ball["radius"][idx])) else None,
                "embedding_ball_ks": float(embedding_ball_ks[idx])
                if math.isfinite(float(embedding_ball_ks[idx]))
                else None,
                "embedding_ball_ks_pvalue": float(embedding_ball_pvalue[idx])
                if math.isfinite(float(embedding_ball_pvalue[idx]))
                else None,
                "embedding_ball_dimension": float(embedding_ball_dimension[idx])
                if math.isfinite(float(embedding_ball_dimension[idx]))
                else None,
                "embedding_ball_radius": float(embedding_ball_radius[idx])
                if math.isfinite(float(embedding_ball_radius[idx]))
                else None,
                "embedding_ball_inner_count": int(embedding_ball["inner_count"][idx]),
                "embedding_radius_threshold_found": bool(embedding_radius_found[idx]),
                "embedding_radius_threshold_radius": float(embedding_radius_threshold_radius[idx])
                if math.isfinite(float(embedding_radius_threshold_radius[idx]))
                else None,
                "embedding_radius_threshold_volume": float(embedding_radius_threshold_volume[idx])
                if math.isfinite(float(embedding_radius_threshold_volume[idx]))
                else None,
                "embedding_radius_threshold_ks": float(embedding_radius_threshold_ks[idx])
                if math.isfinite(float(embedding_radius_threshold_ks[idx]))
                else None,
                "embedding_radius_best_radius": float(embedding_radius_best_radius[idx])
                if math.isfinite(float(embedding_radius_best_radius[idx]))
                else None,
                "embedding_radius_best_volume": float(embedding_radius_best_volume[idx])
                if math.isfinite(float(embedding_radius_best_volume[idx]))
                else None,
                "embedding_radius_best_ks": float(embedding_radius_best_ks[idx])
                if math.isfinite(float(embedding_radius_best_ks[idx]))
                else None,
                "dimension_jump": float(dim_jump[idx]),
                "knn_target_entropy": float(knn_entropy[idx]),
                "top_branch_codes": [int(v) for v in branch_top_indices[idx, : min(8, branch_top_indices.shape[1])].tolist()],
                **{name: bool(mask[idx]) for name, mask in codebook_position_masks.items()},
                **{name: bool(mask[idx]) for name, mask in codebook_control_position_masks.items()},
            }
        )

    summary = {
        "mode": "llamagen-c2i-ks",
        "profile": args.profile,
        "tokens_path": str(Path(args.tokens_path).resolve()),
        "out_dir": str(out_dir),
        "device": str(device),
        "dtype": str(dtype),
        "num_samples": int(tokens.shape[0]),
        "sequence_length": int(tokens.shape[1]),
        "num_tokens": int(tokens_flat.shape[0]),
        "grid": grid,
        "class_labels": class_labels,
        "branch_top_k": int(args.branch_top_k),
        "singular_selection": singular_selection,
        "singular_count": int(singular_mask.sum()),
        "codebook_singular_codes_path": str(Path(args.codebook_singular_codes_path).resolve())
        if args.codebook_singular_codes_path
        else "",
        "codebook_singular_source": str(args.codebook_singular_source),
        "codebook_control_source": str(codebook_control_source),
        "codebook_active_position": str(args.codebook_active_position),
        "use_codebook_singular_as_active": bool(args.use_codebook_singular_as_active),
        "codebook_random_controls": int(args.codebook_random_controls),
        "codebook_frequency_controls": int(args.codebook_frequency_controls),
        "codebook_position_mask_counts": {
            name: int(mask.sum()) for name, mask in sorted(codebook_position_masks.items())
        },
        "codebook_control_position_mask_counts": {
            name: int(mask.sum()) for name, mask in sorted(codebook_control_position_masks.items())
        },
        "paper_geometry": str(args.paper_geometry),
        "paper_alpha": float(args.paper_alpha),
        "paper_protocol": paper_summary,
        "paper_any_rejected_count": int(paper_protocol["any_rejected"].sum()),
        "paper_full_any_rejected_count": int(paper_protocol["any_rejected"].sum()),
        "paper_manifold_any_rejected_count": int(paper_protocol["manifold_any_rejected"].sum()),
        "paper_fiber_any_rejected_count": int(paper_protocol["fiber_any_rejected"].sum()),
        "paper_manifold_small_rejected_count": int(small_radius["manifold_rejected"].sum()),
        "paper_fiber_small_rejected_count": int(small_radius["fiber_rejected"].sum()),
        "paper_manifold_large_rejected_count": int(large_radius["manifold_rejected"].sum()),
        "paper_fiber_large_rejected_count": int(large_radius["fiber_rejected"].sum()),
        "fiber_rejected_count": int(small_radius["fiber_rejected"].sum()),
        "legacy_singular_selection": legacy_singular_selection,
        "legacy_singular_count": int(legacy_singular_mask.sum()),
        "legacy_raw_fiber_rejected_count": int(legacy_raw_rejected.sum()),
        "fiber_summary": fiber_summary,
        "mean_branch_ks_singular": mean_or_nan(branch_ks[singular_mask]),
        "mean_branch_ks_regular": mean_or_nan(branch_ks[regular_mask]),
        "branch_ks_singular_minus_regular": branch_diff,
        "branch_ks_permutation_p": branch_p,
        "branch_ks_cohen_d": cohen_d(branch_ks[singular_mask], branch_ks[regular_mask]),
        "mean_branch_entropy_singular": mean_or_nan(branch_entropy[singular_mask]),
        "mean_branch_entropy_regular": mean_or_nan(branch_entropy[regular_mask]),
        "branch_entropy_singular_minus_regular": entropy_diff,
        "branch_entropy_permutation_p": entropy_p,
        "mean_ranked_ks_singular": mean_or_nan(ranked_ks[singular_mask]),
        "mean_ranked_ks_regular": mean_or_nan(ranked_ks[regular_mask]),
        "ranked_ks_singular_minus_regular": ranked_diff,
        "ranked_ks_permutation_p": ranked_p,
        "mean_permuted_ks_singular": mean_or_nan(permuted["trimmed_mean"][singular_mask]),
        "mean_permuted_ks_regular": mean_or_nan(permuted["trimmed_mean"][regular_mask]),
        "permuted_ks_singular_minus_regular": perm_diff,
        "permuted_ks_permutation_p": perm_p,
        "local_ball_volume": int(args.local_ball_volume),
        "local_ball_include_self": bool(args.local_ball_include_self),
        "mean_local_ball_ks_singular": mean_or_nan(local_ball_ks[singular_mask]),
        "mean_local_ball_ks_regular": mean_or_nan(local_ball_ks[regular_mask]),
        "local_ball_ks_singular_minus_regular": local_ball_ks_diff,
        "local_ball_ks_permutation_p": local_ball_ks_p,
        "mean_local_ball_entropy_singular": mean_or_nan(local_ball_entropy[singular_mask]),
        "mean_local_ball_entropy_regular": mean_or_nan(local_ball_entropy[regular_mask]),
        "local_ball_entropy_singular_minus_regular": local_ball_entropy_diff,
        "local_ball_entropy_permutation_p": local_ball_entropy_p,
        "mean_local_ball_mass_singular": mean_or_nan(local_ball_mass[singular_mask]),
        "mean_local_ball_mass_regular": mean_or_nan(local_ball_mass[regular_mask]),
        "local_ball_mass_singular_minus_regular": local_ball_mass_diff,
        "local_ball_mass_permutation_p": local_ball_mass_p,
        "mean_local_ball_unique_count": mean_or_nan(local_ball["unique_count"]),
        "mean_local_ball_radius": mean_or_nan(local_ball["radius"]),
        "embedding_ball_volume": int(args.embedding_ball_volume),
        "embedding_ball_include_self": bool(args.embedding_ball_include_self),
        "embedding_ball_dimension_trim": float(args.embedding_ball_trim),
        "embedding_ball_min_inner": int(args.embedding_ball_min_inner),
        "mean_embedding_ball_ks_singular": mean_or_nan(embedding_ball_ks[singular_mask]),
        "mean_embedding_ball_ks_regular": mean_or_nan(embedding_ball_ks[regular_mask]),
        "embedding_ball_ks_singular_minus_regular": embedding_ball_ks_diff,
        "embedding_ball_ks_permutation_p": embedding_ball_ks_p,
        "mean_embedding_ball_ks_pvalue_singular": mean_or_nan(embedding_ball_pvalue[singular_mask]),
        "mean_embedding_ball_ks_pvalue_regular": mean_or_nan(embedding_ball_pvalue[regular_mask]),
        "mean_embedding_ball_dimension_singular": mean_or_nan(embedding_ball_dimension[singular_mask]),
        "mean_embedding_ball_dimension_regular": mean_or_nan(embedding_ball_dimension[regular_mask]),
        "embedding_ball_dimension_singular_minus_regular": embedding_ball_dimension_diff,
        "embedding_ball_dimension_permutation_p": embedding_ball_dimension_p,
        "mean_embedding_ball_radius_singular": mean_or_nan(embedding_ball_radius[singular_mask]),
        "mean_embedding_ball_radius_regular": mean_or_nan(embedding_ball_radius[regular_mask]),
        "embedding_ball_radius_singular_minus_regular": embedding_ball_radius_diff,
        "embedding_ball_radius_permutation_p": embedding_ball_radius_p,
        "embedding_radius_volume_min": int(args.embedding_radius_volume_min),
        "embedding_radius_volume_max": int(args.embedding_radius_volume_max),
        "embedding_radius_volume_step": int(args.embedding_radius_volume_step),
        "embedding_radius_uniform_pvalue": float(args.embedding_radius_uniform_pvalue),
        "embedding_radius_max_ks": float(args.embedding_radius_max_ks),
        "embedding_radius_consecutive": int(args.embedding_radius_consecutive),
        "embedding_radius_candidate_volumes": [
            int(v) for v in embedding_radius["candidate_volumes"].astype(np.int64).tolist()
        ],
        "embedding_radius_threshold_rate_singular": mean_or_nan(
            embedding_radius_found[singular_mask].astype(np.float64)
        ),
        "embedding_radius_threshold_rate_regular": mean_or_nan(
            embedding_radius_found[regular_mask].astype(np.float64)
        ),
        "embedding_radius_threshold_rate_singular_minus_regular": embedding_radius_found_diff,
        "embedding_radius_threshold_rate_permutation_p": embedding_radius_found_p,
        "mean_embedding_radius_threshold_radius_singular": mean_or_nan(
            embedding_radius_threshold_radius[singular_mask]
        ),
        "mean_embedding_radius_threshold_radius_regular": mean_or_nan(
            embedding_radius_threshold_radius[regular_mask]
        ),
        "embedding_radius_threshold_radius_singular_minus_regular": embedding_radius_threshold_diff,
        "embedding_radius_threshold_radius_permutation_p": embedding_radius_threshold_p,
        "mean_embedding_radius_threshold_volume_singular": mean_or_nan(
            embedding_radius_threshold_volume[singular_mask]
        ),
        "mean_embedding_radius_threshold_volume_regular": mean_or_nan(
            embedding_radius_threshold_volume[regular_mask]
        ),
        "embedding_radius_threshold_volume_singular_minus_regular": embedding_radius_volume_diff,
        "embedding_radius_threshold_volume_permutation_p": embedding_radius_volume_p,
        "mean_embedding_radius_threshold_ks_singular": mean_or_nan(
            embedding_radius_threshold_ks[singular_mask]
        ),
        "mean_embedding_radius_threshold_ks_regular": mean_or_nan(
            embedding_radius_threshold_ks[regular_mask]
        ),
        "mean_embedding_radius_best_radius_singular": mean_or_nan(embedding_radius_best_radius[singular_mask]),
        "mean_embedding_radius_best_radius_regular": mean_or_nan(embedding_radius_best_radius[regular_mask]),
        "mean_embedding_radius_best_volume_singular": mean_or_nan(embedding_radius_best_volume[singular_mask]),
        "mean_embedding_radius_best_volume_regular": mean_or_nan(embedding_radius_best_volume[regular_mask]),
        "mean_embedding_radius_best_ks_singular": mean_or_nan(embedding_radius_best_ks[singular_mask]),
        "mean_embedding_radius_best_ks_regular": mean_or_nan(embedding_radius_best_ks[regular_mask]),
        "embedding_radius_best_ks_singular_minus_regular": embedding_radius_best_ks_diff,
        "embedding_radius_best_ks_permutation_p": embedding_radius_best_ks_p,
        "embedding_radius_mean_ks_by_volume_singular": [
            float(v) if math.isfinite(float(v)) else None
            for v in np.nanmean(embedding_radius["ks_by_volume"][singular_mask], axis=0).tolist()
        ],
        "embedding_radius_mean_ks_by_volume_regular": [
            float(v) if math.isfinite(float(v)) else None
            for v in np.nanmean(embedding_radius["ks_by_volume"][regular_mask], axis=0).tolist()
        ],
        "flat_quantile": float(args.flat_quantile),
        "flat_singular_rate": flat_singular_rate,
        "rest_singular_rate": rest_singular_rate,
        "flat_singular_rate_minus_rest": flat_rate_diff,
        "flat_singular_rate_p": flat_rate_p,
        "embedding_ball_flat_singular_rate": embedding_ball_flat_singular_rate,
        "embedding_ball_rest_singular_rate": embedding_ball_rest_singular_rate,
        "embedding_ball_flat_singular_rate_minus_rest": embedding_ball_flat_rate_diff,
        "embedding_ball_flat_singular_rate_p": embedding_ball_flat_rate_p,
        "singular_sensitivity": sensitivity_rows,
        "missing_weight_keys": missing,
        "unexpected_weight_keys": unexpected,
        "figures": figures,
    }

    summary_path = out_dir / "vq_ar_ks_summary.json"
    records_path = out_dir / "vq_ar_ks_tokens.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    records_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=[tag.strip() for tag in str(args.wandb_tags).split(",") if tag.strip()],
            config={k: v for k, v in summary.items() if isinstance(v, (str, int, float, bool))},
        )
        payload = {
            "vq_ar_ks/mean_branch_ks_singular": summary["mean_branch_ks_singular"],
            "vq_ar_ks/mean_branch_ks_regular": summary["mean_branch_ks_regular"],
            "vq_ar_ks/branch_ks_singular_minus_regular": summary["branch_ks_singular_minus_regular"],
            "vq_ar_ks/branch_ks_permutation_p": summary["branch_ks_permutation_p"],
            "vq_ar_ks/mean_branch_entropy_singular": summary["mean_branch_entropy_singular"],
            "vq_ar_ks/mean_branch_entropy_regular": summary["mean_branch_entropy_regular"],
            "vq_ar_ks/mean_local_ball_ks_singular": summary["mean_local_ball_ks_singular"],
            "vq_ar_ks/mean_local_ball_ks_regular": summary["mean_local_ball_ks_regular"],
            "vq_ar_ks/local_ball_ks_singular_minus_regular": summary["local_ball_ks_singular_minus_regular"],
            "vq_ar_ks/local_ball_ks_permutation_p": summary["local_ball_ks_permutation_p"],
            "vq_ar_ks/mean_local_ball_entropy_singular": summary["mean_local_ball_entropy_singular"],
            "vq_ar_ks/mean_local_ball_entropy_regular": summary["mean_local_ball_entropy_regular"],
            "vq_ar_ks/local_ball_entropy_singular_minus_regular": summary["local_ball_entropy_singular_minus_regular"],
            "vq_ar_ks/local_ball_entropy_permutation_p": summary["local_ball_entropy_permutation_p"],
            "vq_ar_ks/mean_embedding_ball_ks_singular": summary["mean_embedding_ball_ks_singular"],
            "vq_ar_ks/mean_embedding_ball_ks_regular": summary["mean_embedding_ball_ks_regular"],
            "vq_ar_ks/embedding_ball_ks_singular_minus_regular": summary[
                "embedding_ball_ks_singular_minus_regular"
            ],
            "vq_ar_ks/embedding_ball_ks_permutation_p": summary["embedding_ball_ks_permutation_p"],
            "vq_ar_ks/mean_embedding_ball_dimension_singular": summary[
                "mean_embedding_ball_dimension_singular"
            ],
            "vq_ar_ks/mean_embedding_ball_dimension_regular": summary[
                "mean_embedding_ball_dimension_regular"
            ],
            "vq_ar_ks/embedding_radius_threshold_rate_singular": summary[
                "embedding_radius_threshold_rate_singular"
            ],
            "vq_ar_ks/embedding_radius_threshold_rate_regular": summary[
                "embedding_radius_threshold_rate_regular"
            ],
            "vq_ar_ks/embedding_radius_threshold_rate_singular_minus_regular": summary[
                "embedding_radius_threshold_rate_singular_minus_regular"
            ],
            "vq_ar_ks/embedding_radius_threshold_rate_permutation_p": summary[
                "embedding_radius_threshold_rate_permutation_p"
            ],
            "vq_ar_ks/mean_embedding_radius_threshold_radius_singular": summary[
                "mean_embedding_radius_threshold_radius_singular"
            ],
            "vq_ar_ks/mean_embedding_radius_threshold_radius_regular": summary[
                "mean_embedding_radius_threshold_radius_regular"
            ],
            "vq_ar_ks/embedding_radius_threshold_radius_permutation_p": summary[
                "embedding_radius_threshold_radius_permutation_p"
            ],
            "vq_ar_ks/mean_embedding_radius_best_ks_singular": summary[
                "mean_embedding_radius_best_ks_singular"
            ],
            "vq_ar_ks/mean_embedding_radius_best_ks_regular": summary[
                "mean_embedding_radius_best_ks_regular"
            ],
            "vq_ar_ks/embedding_radius_best_ks_permutation_p": summary[
                "embedding_radius_best_ks_permutation_p"
            ],
            "vq_ar_ks/singular_count": summary["singular_count"],
            "vq_ar_ks/fiber_rejected_count": summary["fiber_rejected_count"],
        }
        for name, count in summary["codebook_position_mask_counts"].items():
            payload[f"vq_ar_ks/{name}_count"] = count
        for name, count in summary["codebook_control_position_mask_counts"].items():
            payload[f"vq_ar_ks/{name}_count"] = count
        for key, path in figures.items():
            payload[f"vq_ar_ks/{key}"] = wandb.Image(path)
        wandb.log(payload)
        artifact = wandb.Artifact(f"{args.wandb_name}_outputs", type="analysis")
        artifact.add_file(str(summary_path))
        artifact.add_file(str(records_path))
        for path in figures.values():
            artifact.add_file(path)
        run.log_artifact(artifact)
        run.finish()

    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(LLAMAGEN_PROFILES), default="c2i-B-256")
    parser.add_argument("--tokens-path", required=True)
    parser.add_argument("--class-labels", default="207,360,387,974")
    parser.add_argument("--class-labels-file", default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--llamagen-repo", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--geometry-pca-dim", type=int, default=64)
    parser.add_argument("--vol-min", type=int, default=8)
    parser.add_argument("--vol-max", type=int, default=64)
    parser.add_argument("--paper-small-vol-min", type=int, default=10)
    parser.add_argument("--paper-small-vol-max", type=int, default=50)
    parser.add_argument("--paper-large-vol-min", type=int, default=50)
    parser.add_argument("--paper-large-vol-max", type=int, default=200)
    parser.add_argument("--paper-alpha", type=float, default=1e-3)
    parser.add_argument("--paper-geometry", choices=["original", "robust"], default="original")
    parser.add_argument(
        "--singular-source",
        choices=[
            "paper_any",
            "paper_stratified_any",
            "paper_manifold_any",
            "paper_fiber_any",
            "paper_stratified_small",
            "paper_stratified_large",
            "paper_fiber_small",
            "paper_fiber_large",
            "paper_manifold_small",
            "paper_manifold_large",
        ],
        default="paper_fiber_any",
    )
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--nstrat", type=int, default=3)
    parser.add_argument("--singular-fraction", type=float, default=0.10)
    parser.add_argument("--min-singular", type=int, default=24)
    parser.add_argument("--codebook-singular-codes-path", default="")
    parser.add_argument(
        "--codebook-singular-source",
        choices=[
            "singular_any",
            "singular_manifold_any",
            "singular_fiber_any",
            "small_manifold",
            "small_fiber",
            "large_manifold",
            "large_fiber",
        ],
        default="singular_any",
    )
    parser.add_argument("--codebook-active-position", choices=["target", "prev"], default="target")
    parser.add_argument("--use-codebook-singular-as-active", action="store_true")
    parser.add_argument("--codebook-control-source", default="")
    parser.add_argument("--codebook-random-controls", type=int, default=0)
    parser.add_argument("--codebook-frequency-controls", type=int, default=0)
    parser.add_argument("--knn-entropy-k", type=int, default=16)
    parser.add_argument("--branch-top-k", type=int, default=32)
    parser.add_argument("--local-ball-volume", type=int, default=32)
    parser.add_argument("--local-ball-include-self", action="store_true")
    parser.add_argument("--embedding-ball-volume", type=int, default=50)
    parser.add_argument("--embedding-ball-include-self", action="store_true")
    parser.add_argument("--embedding-ball-trim", type=float, default=0.10)
    parser.add_argument("--embedding-ball-min-inner", type=int, default=8)
    parser.add_argument("--embedding-radius-volume-min", type=int, default=20)
    parser.add_argument("--embedding-radius-volume-max", type=int, default=200)
    parser.add_argument("--embedding-radius-volume-step", type=int, default=5)
    parser.add_argument("--embedding-radius-uniform-pvalue", type=float, default=0.05)
    parser.add_argument("--embedding-radius-max-ks", type=float, default=0.15)
    parser.add_argument("--embedding-radius-consecutive", type=int, default=2)
    parser.add_argument("--permuted-ks", type=int, default=16)
    parser.add_argument("--permutation-reps", type=int, default=5000)
    parser.add_argument("--flat-quantile", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decode-grid", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="stratified-manifold-learning")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default="llamagen-c2i-B-256-robust-ks")
    parser.add_argument("--wandb-tags", default="vq-ar,llamagen,robust-ks,singularity")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = run(args)
    printable = {k: v for k, v in summary.items() if k not in {"fiber_summary", "figures"}}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
