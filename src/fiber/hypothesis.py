"""Stratified-manifold hypothesis metrics and reporting."""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import torch

from fiber.geometry import min_change_pvalue, min_fiber_violation_pvalue


def _pairwise_distance_matrix(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    squared_norms = np.sum(pts * pts, axis=1, keepdims=True)
    distances_sq = np.maximum(squared_norms + squared_norms.T - 2.0 * (pts @ pts.T), 0.0)
    return np.sqrt(distances_sq, out=distances_sq)


def _first_dimension(res: Dict[str, Any] | None) -> float:
    if res and res.get("dimensions"):
        return float(res["dimensions"][0])
    return float("nan")


def _min_pvalue(res: Dict[str, Any] | None) -> float:
    return min_change_pvalue(res)


def _finite_mean(values: np.ndarray | List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _finite_std(values: np.ndarray | List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if arr.size else float("nan")


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def summarize_hypothesis_metrics(
    *,
    embeddings: torch.Tensor,
    fiber_results: List[Dict[str, Any]],
    image_ids: torch.Tensor,
    bboxes: torch.Tensor,
    neighborhood_dims: List[float] | None,
    neighborhood_size: int,
    alpha: float,
    precomputed_dists: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Compute hypothesis metrics.

    Parameters
    ----------
    precomputed_dists : optional
        Full (unsorted) pairwise distance matrix (N, N).  If provided, avoids
        recomputing all-pairs distances from ``embeddings``.
    """
    n = min(
        int(embeddings.shape[0]),
        len(fiber_results),
        int(image_ids.shape[0]) if isinstance(image_ids, torch.Tensor) else len(image_ids),
        int(bboxes.shape[0]) if isinstance(bboxes, torch.Tensor) else len(bboxes),
    )
    _nan_result = {
        "neighbor_k": 0,
        "same_image_neighbor_ratio_mean": float("nan"),
        "same_image_neighbor_top1": float("nan"),
        "same_image_neighbor_chance": float("nan"),
        "same_image_neighbor_lift": float("nan"),
        "local_same_image_neighbor_ratio_mean": float("nan"),
        "local_same_image_given_same_image_mean": float("nan"),
        "mean_strata_count": float("nan"),
        "multi_strata_ratio": float("nan"),
        "mean_second_strata_gap": float("nan"),
        "neighborhood_dim_gap_mean": float("nan"),
        "image_mean_dim_std": float("nan"),
        "image_mean_dim_cv": float("nan"),
        "image_internal_dim_std_mean": float("nan"),
        "change_point_ratio": float("nan"),
        "fiber_violation_ratio": float("nan"),
        "regular_token_ratio": float("nan"),
        "same_image_fiber_score": float("nan"),
        "local_chart_score": float("nan"),
        "smoothness_score": float("nan"),
        "regularity_score": float("nan"),
        "hypothesis_score": float("nan"),
        "hypothesis_stage": "insufficient_tokens",
        "hypothesis_narrative": "too few tokens for hypothesis logging",
    }
    if n <= 1:
        return _nan_result

    dims = np.asarray([_first_dimension(res) for res in fiber_results[:n]], dtype=np.float64)
    min_pvals = np.asarray([_min_pvalue(res) for res in fiber_results[:n]], dtype=np.float64)
    violation_pvals = np.asarray(
        [min_fiber_violation_pvalue(res) for res in fiber_results[:n]],
        dtype=np.float64,
    )
    strata_counts = np.asarray(
        [len(res.get("dimensions", [])) if res else 0 for res in fiber_results[:n]],
        dtype=np.int64,
    )
    second_strata_gaps = np.asarray(
        [
            abs(float(res["dimensions"][1]) - float(res["dimensions"][0]))
            for res in fiber_results[:n]
            if res and len(res.get("dimensions", [])) > 1
        ],
        dtype=np.float64,
    )
    img_ids = image_ids[:n].detach().cpu().numpy().astype(np.int64)
    bbox_np = bboxes[:n].detach().cpu().numpy().astype(np.float64)
    centers = np.column_stack(
        ((bbox_np[:, 0] + bbox_np[:, 2]) * 0.5, (bbox_np[:, 1] + bbox_np[:, 3]) * 0.5)
    )

    # Reuse precomputed distances when available
    if precomputed_dists is not None:
        dists = precomputed_dists[:n, :n].copy()
    else:
        emb_np = embeddings[:n].detach().cpu().numpy().astype(np.float64)
        dists = _pairwise_distance_matrix(emb_np)
    np.fill_diagonal(dists, np.inf)

    neighbor_k = min(16, max(1, n - 1), max(2, int(round(math.sqrt(n)))))
    nn_idx = np.argpartition(dists, kth=neighbor_k - 1, axis=1)[:, :neighbor_k]
    nn_dists = np.take_along_axis(dists, nn_idx, axis=1)
    nn_order = np.argsort(nn_dists, axis=1)
    nn_idx = np.take_along_axis(nn_idx, nn_order, axis=1)

    same_image_mask = img_ids[nn_idx] == img_ids[:, None]
    center_dists = _pairwise_distance_matrix(centers)
    local_same_mask = same_image_mask & (
        np.take_along_axis(center_dists, nn_idx, axis=1) <= float(neighborhood_size)
    )

    unique_ids, counts = np.unique(img_ids, return_counts=True)
    count_map = {int(img_id): int(count) for img_id, count in zip(unique_ids.tolist(), counts.tolist())}
    same_image_chance = _finite_mean(
        [(count_map.get(int(img_id), 1) - 1) / max(1, n - 1) for img_id in img_ids.tolist()]
    )
    same_image_neighbor_ratio_mean = float(np.mean(same_image_mask.mean(axis=1)))
    same_image_neighbor_top1 = (
        float(np.mean(same_image_mask[:, 0])) if same_image_mask.shape[1] else float("nan")
    )
    same_image_neighbor_lift = (
        same_image_neighbor_ratio_mean / same_image_chance
        if math.isfinite(same_image_chance) and same_image_chance > 0
        else float("nan")
    )
    local_same_image_neighbor_ratio_mean = float(np.mean(local_same_mask.mean(axis=1)))
    local_same_den = np.maximum(1, same_image_mask.sum(axis=1))
    local_same_image_given_same_image_mean = float(
        np.mean(local_same_mask.sum(axis=1) / local_same_den)
    )

    neigh_dims = np.asarray(neighborhood_dims[:n] if neighborhood_dims else [], dtype=np.float64)
    if neigh_dims.size:
        neighborhood_dim_gap_mean = _finite_mean(np.abs(dims[: neigh_dims.size] - neigh_dims))
    else:
        neighborhood_dim_gap_mean = float("nan")

    image_mean_dims, image_internal_dim_stds = [], []
    for img_id in unique_ids.tolist():
        vals = dims[img_ids == int(img_id)]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            image_mean_dims.append(float(np.mean(vals)))
            if vals.size > 1:
                image_internal_dim_stds.append(float(np.std(vals)))
    image_mean_dim_std = _finite_std(image_mean_dims)
    image_mean_dim_mean = _finite_mean(image_mean_dims)
    image_mean_dim_cv = (
        image_mean_dim_std / max(abs(image_mean_dim_mean), 1e-8)
        if math.isfinite(image_mean_dim_std) and math.isfinite(image_mean_dim_mean)
        else float("nan")
    )
    image_internal_dim_std_mean = _finite_mean(image_internal_dim_stds)
    valid_change = np.isfinite(min_pvals)
    valid_violation = np.isfinite(violation_pvals)
    change_point_ratio = (
        float(np.mean(valid_change & (min_pvals < alpha))) if min_pvals.size else float("nan")
    )
    irregular_token_ratio = (
        float(np.mean(valid_violation & (violation_pvals < alpha)))
        if violation_pvals.size
        else float("nan")
    )
    regular_token_ratio = 1.0 - irregular_token_ratio if math.isfinite(irregular_token_ratio) else float("nan")

    same_image_fiber_score = _clip01(
        ((same_image_neighbor_lift if math.isfinite(same_image_neighbor_lift) else 0.0) - 1.0)
        / 9.0
    )
    local_chart_score = _clip01(local_same_image_given_same_image_mean)
    smoothness_score = (
        _clip01(1.0 / (1.0 + max(0.0, neighborhood_dim_gap_mean)))
        if math.isfinite(neighborhood_dim_gap_mean)
        else 0.0
    )
    regularity_score = _clip01(regular_token_ratio)
    hypothesis_score = float(
        np.mean([same_image_fiber_score, local_chart_score, smoothness_score, regularity_score])
    )

    stage = "weak"
    if hypothesis_score >= 0.8:
        stage = "strong"
    elif hypothesis_score >= 0.6:
        stage = "moderate"
    elif hypothesis_score >= 0.4:
        stage = "mixed"

    narrative_parts: list[str] = []
    if math.isfinite(same_image_neighbor_lift):
        if same_image_neighbor_lift >= 5.0:
            narrative_parts.append("strong same-image fiber concentration")
        elif same_image_neighbor_lift >= 2.0:
            narrative_parts.append("moderate same-image fiber concentration")
        else:
            narrative_parts.append("weak same-image fiber concentration")
    if math.isfinite(local_same_image_given_same_image_mean):
        if local_same_image_given_same_image_mean >= 0.6:
            narrative_parts.append("neighbors remain spatially local within images")
        elif local_same_image_given_same_image_mean >= 0.3:
            narrative_parts.append("some within-image chart locality")
        else:
            narrative_parts.append("weak within-image chart locality")
    if math.isfinite(regular_token_ratio):
        if regular_token_ratio >= 0.8:
            narrative_parts.append("singular set appears concentrated")
        elif regular_token_ratio >= 0.5:
            narrative_parts.append("singular set is present but not dominant")
        else:
            narrative_parts.append("irregular set is too diffuse")
    if math.isfinite(neighborhood_dim_gap_mean):
        if neighborhood_dim_gap_mean <= 0.25:
            narrative_parts.append("dimension field is locally smooth")
        elif neighborhood_dim_gap_mean <= 0.75:
            narrative_parts.append("dimension field has moderate local drift")
        else:
            narrative_parts.append("dimension field varies sharply")

    return {
        "neighbor_k": int(neighbor_k),
        "same_image_neighbor_ratio_mean": same_image_neighbor_ratio_mean,
        "same_image_neighbor_top1": same_image_neighbor_top1,
        "same_image_neighbor_chance": same_image_chance,
        "same_image_neighbor_lift": same_image_neighbor_lift,
        "local_same_image_neighbor_ratio_mean": local_same_image_neighbor_ratio_mean,
        "local_same_image_given_same_image_mean": local_same_image_given_same_image_mean,
        "mean_strata_count": float(np.mean(strata_counts)) if strata_counts.size else float("nan"),
        "multi_strata_ratio": float(np.mean(strata_counts > 1)) if strata_counts.size else float("nan"),
        "mean_second_strata_gap": _finite_mean(second_strata_gaps),
        "neighborhood_dim_gap_mean": neighborhood_dim_gap_mean,
        "image_mean_dim_std": image_mean_dim_std,
        "image_mean_dim_cv": float(image_mean_dim_cv) if math.isfinite(image_mean_dim_cv) else float("nan"),
        "image_internal_dim_std_mean": image_internal_dim_std_mean,
        "change_point_ratio": change_point_ratio,
        "fiber_violation_ratio": irregular_token_ratio,
        "regular_token_ratio": regular_token_ratio,
        "same_image_fiber_score": same_image_fiber_score,
        "local_chart_score": local_chart_score,
        "smoothness_score": smoothness_score,
        "regularity_score": regularity_score,
        "hypothesis_score": hypothesis_score,
        "hypothesis_stage": stage,
        "hypothesis_narrative": "; ".join(narrative_parts) if narrative_parts else "no stable hypothesis signal",
    }


def format_hypothesis_summary_line(*, epoch: int, metrics: Dict[str, Any]) -> str:
    def _fmt(value: Any, fmt: str) -> str:
        try:
            value_f = float(value)
        except Exception:
            return "nan"
        return format(value_f, fmt) if math.isfinite(value_f) else "nan"

    return (
        f"[hypothesis] Epoch {epoch:03d} | score {_fmt(metrics.get('hypothesis_score'), '.2f')} "
        f"({metrics.get('hypothesis_stage', 'n/a')}) | same-image {_fmt(metrics.get('same_image_neighbor_ratio_mean'), '.2f')} "
        f"(chance {_fmt(metrics.get('same_image_neighbor_chance'), '.2f')}, "
        f"lift x{_fmt(metrics.get('same_image_neighbor_lift'), '.1f')}) | "
        f"local {_fmt(metrics.get('local_same_image_given_same_image_mean'), '.2f')} | "
        f"dim-gap {_fmt(metrics.get('neighborhood_dim_gap_mean'), '.2f')} | "
        f"multi-strata {_fmt(metrics.get('multi_strata_ratio'), '.2f')} | "
        f"regular {_fmt(metrics.get('regular_token_ratio'), '.2f')}"
    )


def summarize_class_dimensions(
    fiber_results: List[Dict], labels: torch.Tensor, num_classes: int
) -> tuple[List[float], List[int]]:
    dims = np.array(
        [
            res["dimensions"][0] if res and res.get("dimensions") else np.nan
            for res in fiber_results
        ],
        dtype=np.float64,
    )
    if dims.size == 0:
        return [float("nan")] * num_classes, [0] * num_classes
    lbls = labels[: dims.shape[0]]
    if isinstance(lbls, torch.Tensor) and lbls.dim() == 2 and lbls.shape[1] == num_classes:
        present = (lbls > 0).cpu().numpy().astype(np.bool_)
        means, counts = [], []
        for j in range(num_classes):
            vals = dims[present[:, j]]
            vals = vals[np.isfinite(vals)]
            means.append(float(np.mean(vals)) if vals.size else float("nan"))
            counts.append(int(present[:, j].sum()))
        return means, counts
    lbl_np = lbls.cpu().numpy().astype(np.int64).reshape(-1)
    buckets: list[list[float]] = [[] for _ in range(num_classes)]
    for d, l in zip(dims.tolist(), lbl_np.tolist()):
        if math.isfinite(d) and 0 <= l < num_classes:
            buckets[l].append(d)
    return (
        [float(np.mean(b)) if b else float("nan") for b in buckets],
        [len(b) for b in buckets],
    )


def estimate_neighborhood_dimensions(
    fiber_results: List[Dict], bboxes: torch.Tensor, neighborhood_size: int
) -> List[float]:
    if not fiber_results or bboxes is None or bboxes.numel() == 0:
        return []
    dims = np.array(
        [
            res["dimensions"][0] if res and res.get("dimensions") else np.nan
            for res in fiber_results
        ],
        dtype=np.float64,
    )
    b_np = bboxes[: len(dims)].cpu().numpy()
    centers = np.column_stack(
        ((b_np[:, 0] + b_np[:, 2]) * 0.5, (b_np[:, 1] + b_np[:, 3]) * 0.5)
    )
    dist = _pairwise_distance_matrix(centers)
    radius = neighborhood_size * 0.5
    result: list[float] = []
    for i in range(len(dims)):
        mask = dist[i] <= radius
        masked_dims = dims[mask]
        finite = masked_dims[np.isfinite(masked_dims)]
        result.append(float(np.mean(finite)) if finite.size else float("nan"))
    return result


def compute_stratified_manifold_hypothesis_metrics(
    *,
    embeddings: torch.Tensor,
    fiber_results: List[Dict[str, Any]],
    image_ids: torch.Tensor,
    bboxes: torch.Tensor,
    neighborhood_dims: List[float] | None,
    neighborhood_size: int,
    alpha: float,
    precomputed_dists: np.ndarray | None = None,
) -> Dict[str, Any]:
    return summarize_hypothesis_metrics(
        embeddings=embeddings,
        fiber_results=fiber_results,
        image_ids=image_ids,
        bboxes=bboxes,
        neighborhood_dims=neighborhood_dims,
        neighborhood_size=neighborhood_size,
        alpha=alpha,
        precomputed_dists=precomputed_dists,
    )


def format_hypothesis_log_line(*, epoch: int, metrics: Dict[str, Any]) -> str:
    return format_hypothesis_summary_line(epoch=epoch, metrics=metrics)


def compute_class_dim_means(
    fiber_results: List[Dict], labels: torch.Tensor, num_classes: int
) -> tuple[List[float], List[int]]:
    return summarize_class_dimensions(fiber_results, labels, num_classes)


def compute_neighborhood_dimensions(
    fiber_results: List[Dict], bboxes: torch.Tensor, neighborhood_size: int
) -> List[float]:
    return estimate_neighborhood_dimensions(fiber_results, bboxes, neighborhood_size)
