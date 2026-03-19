#!/usr/bin/env python3
"""
Fiber bundle / stratified estimator analysis utilities for TinyViT embeddings.
Includes patch-token collection, stratification tests, and visualization helpers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.spatial
import scipy.stats
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image, ImageDraw, ImageFilter

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    TSNE = None
    HAS_TSNE = False

from data import get_norm_stats
from utils import denormalize_images, to_serializable

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    matplotlib = None
    plt = None
    HAS_MATPLOTLIB = False


def _require_matplotlib():
    if plt is None:
        raise ImportError("matplotlib is required for fiber-bundle plotting outputs")
    return plt


# ---------------------------------------------------------------------------
# Patch Token Collection
# ---------------------------------------------------------------------------
def _boxes_to_patch_multihot(boxes: List[tuple], *, grid: int, patch_px: int, num_classes: int) -> torch.Tensor:
    P = grid * grid
    y = torch.zeros((P, num_classes), dtype=torch.float32)
    if grid <= 0 or patch_px <= 0 or num_classes <= 0 or not boxes:
        return y
    for x0, y0, x1, y1, cat in boxes:
        cat_i = int(cat)
        if cat_i < 0 or cat_i >= num_classes:
            continue
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = max(x0, x1), max(y0, y1)
        c0 = int(max(0, min(grid - 1, math.floor(x0 / patch_px))))
        c1 = int(max(0, min(grid - 1, math.floor((x1 - 1e-6) / patch_px))))
        r0 = int(max(0, min(grid - 1, math.floor(y0 / patch_px))))
        r1 = int(max(0, min(grid - 1, math.floor((y1 - 1e-6) / patch_px))))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                y[r * grid + c, cat_i] = 1.0
    return y


def collect_patch_tokens(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, patch_size: int,
    max_tokens: int | None = 256, show_progress: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    base_ds = getattr(loader, "dataset", None)
    while hasattr(base_ds, "dataset"):
        base_ds = base_ds.dataset
    has_instance_boxes = hasattr(base_ds, "instances_after_eval_transform")

    embeddings, labels, images, bboxes, patch_indices, image_ids, pred_labels = [], [], [], [], [], [], []
    collected, image_offset = 0, 0
    iterator = tqdm(loader, desc="Collect val tokens", leave=False) if show_progress else loader

    with torch.no_grad():
        for batch in iterator:
            if max_tokens is not None and collected >= max_tokens:
                break
            imgs, lbls = batch[0].to(device), batch[1]
            idxs = batch[2] if len(batch) > 2 else None
            feats = model.forward_features(imgs)
            logits = model.tokens_to_logits(feats) if hasattr(model, "tokens_to_logits") else model.head(feats[:, 0])
            preds = logits.argmax(dim=-1).cpu()
            start_idx = 2 if getattr(model, "has_dist_token", False) else 1
            patch_tokens = feats[:, start_idx:, :].cpu()
            B, P, E = patch_tokens.shape
            grid = int(math.sqrt(P))
            img_size_px = int(imgs.shape[-1])

            patch_labels_per_image = None
            if has_instance_boxes and idxs is not None and isinstance(lbls, torch.Tensor) and lbls.dim() == 2:
                try:
                    idx_list = idxs.detach().cpu().tolist() if isinstance(idxs, torch.Tensor) else list(idxs)
                    num_classes = int(lbls.shape[1])
                    patch_labels_per_image = [
                        _boxes_to_patch_multihot(
                            base_ds.instances_after_eval_transform(int(idx_list[i]), img_size_px),
                            grid=grid, patch_px=patch_size, num_classes=num_classes
                        ) for i in range(B)
                    ]
                except Exception:
                    patch_labels_per_image = None

            for i in range(B):
                if max_tokens is not None and collected >= max_tokens:
                    break
                for p in range(P):
                    if max_tokens is not None and collected >= max_tokens:
                        break
                    embeddings.append(patch_tokens[i, p])
                    labels.append(patch_labels_per_image[i][p] if patch_labels_per_image else lbls[i].cpu())
                    images.append(imgs[i].cpu())
                    row, col = divmod(p, grid)
                    bboxes.append(torch.tensor([col * patch_size, row * patch_size,
                                                (col + 1) * patch_size, (row + 1) * patch_size], dtype=torch.int32))
                    patch_indices.append(torch.tensor(p, dtype=torch.int32))
                    image_ids.append(torch.tensor(image_offset + i, dtype=torch.int32))
                    pred_labels.append(preds[i])
                    collected += 1
            image_offset += B

    if not embeddings:
        return (torch.empty(0, getattr(model, 'embed_dim', 192)), torch.empty(0, dtype=torch.long),
                torch.empty(0), torch.empty(0, 4, dtype=torch.int32), torch.empty(0, dtype=torch.int32),
                torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.int64))
    return (torch.stack(embeddings)[:max_tokens], torch.stack(labels)[:max_tokens], torch.stack(images)[:max_tokens],
            torch.stack(bboxes)[:max_tokens], torch.stack(patch_indices)[:max_tokens],
            torch.stack(image_ids)[:max_tokens], torch.stack(pred_labels)[:max_tokens])


# ---------------------------------------------------------------------------
# Geometric Estimator & Stratification
# ---------------------------------------------------------------------------
def geo_estimator(radii: np.ndarray, volumes: np.ndarray, npts: int, args: SimpleNamespace) -> Tuple[float, float, float]:
    rstack = np.column_stack((np.ones_like(radii), np.log(radii)))
    pointwise_lfit_data = np.linalg.lstsq(rstack, np.log(volumes), rcond=None)
    pointwise_lfit = pointwise_lfit_data[0]
    scaling_coeff = np.exp(pointwise_lfit[0]) / npts
    dimension = pointwise_lfit[1]
    if args.miller:
        scaling_coeff *= np.exp(0.5 * pointwise_lfit_data[1][0] ** 2)
    ricci = np.mean(-pointwise_lfit_data[1] * 6 * (dimension + 2) / radii**2) if args.ricci else 0.0
    return scaling_coeff, dimension, ricci


def stratification_test(radii: np.ndarray, volumes: np.ndarray, ws: int = 10, alpha: float = 1e-3) -> Tuple[Optional[int], float]:
    radii_safe = np.clip(radii, 1e-12, None)
    volumes_safe = np.maximum(volumes, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        grad_v = np.gradient(np.log(volumes_safe))
        grad_r = np.gradient(np.log(radii_safe))
        grad_r = np.where(np.abs(grad_r) < 1e-12, np.nan, grad_r)
        dimvec = np.divide(grad_v, grad_r, out=np.full_like(grad_v, np.nan), where=~np.isnan(grad_r))
    for w in range(2 * ws, dimvec.shape[0] - 2 * ws):
        t1 = dimvec[w - 2 * ws:w - ws]
        t1 = t1[np.logical_and(np.abs(t1) > 1e-5, np.isfinite(t1))]
        t2 = dimvec[w + ws:w + 2 * ws]
        t2 = t2[np.logical_and(np.abs(t2) > 1e-5, np.isfinite(t2))]
        pvalue = scipy.stats.ttest_ind(t1, t2, equal_var=False).pvalue
        if pvalue < alpha:
            return w, pvalue
    return None, 1.0


def estimate_stratifications(dists_sorted: np.ndarray, vol_min: int, vol_max: int, npts: int,
                              args: SimpleNamespace, ws: int = 10, alpha: float = 1e-3) -> Dict[str, List[float]]:
    radii = dists_sorted[vol_min:vol_max]
    volumes = np.arange(vol_min, vol_max)
    output = {"scaling_coeffs": [], "dimensions": [], "riccis": [], "strat_radii": [], "strat_volumes": [], "pvalues": []}
    vol_min_current = np.argmax(radii > 1e-10)
    for _ in range(args.nstrat):
        vol_max_current = radii.shape[0]
        strat_idx, pvalue = stratification_test(radii[vol_min_current:vol_max_current],
                                                  volumes[vol_min_current:vol_max_current], ws, alpha / args.nstrat)
        if strat_idx is not None:
            vol_max_current = strat_idx + vol_min_current
        scaling_coeff, dimension, ricci = geo_estimator(radii[vol_min_current:vol_max_current],
                                                         volumes[vol_min_current:vol_max_current], npts, args)
        output["scaling_coeffs"].append(scaling_coeff)
        output["dimensions"].append(dimension)
        output["riccis"].append(ricci)
        output["strat_volumes"].append(vol_min + vol_min_current)
        output["strat_radii"].append(radii[vol_min_current])
        output["pvalues"].append(pvalue)
        if strat_idx is None:
            break
        vol_min_current = strat_idx + vol_min_current
    return output


def normalize_volume_range(npts: int, vol_min: int, vol_max: int) -> tuple[int, int]:
    """Clamp (vol_min, vol_max) to a valid kNN range given npts.

    The estimator expects k in [vol_min, vol_max) with vol_min >= 1 and vol_max <= npts - 1.
    """
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
    dists = scipy.spatial.distance_matrix(coords, coords)
    return np.sort(dists, axis=0)


def run_fiber_bundle_test_from_sorted_dists(
    dists_sorted: np.ndarray,
    *,
    vol_min: int = 8,
    vol_max: int = 64,
    ws: int = 8,
    alpha: float = 1e-2,
    nstrat: int = 3,
) -> List[Dict[str, List[float]]]:
    """Fiber bundle test using a precomputed sorted distance matrix.

    `dists_sorted` must be shape (npts, npts) and sorted along axis=0.
    """
    npts = int(dists_sorted.shape[0])
    if npts < 2:
        return []
    vol_min, vol_max = normalize_volume_range(npts, vol_min, vol_max)
    args = SimpleNamespace(nstrat=nstrat, miller=True, ricci=False)
    return [
        estimate_stratifications(dists_sorted[:, i], vol_min, vol_max, npts, args, ws=ws, alpha=alpha)
        for i in range(npts)
    ]


def run_fiber_bundle_test(embeddings: torch.Tensor, vol_min: int = 8, vol_max: int = 64,
                           ws: int = 8, alpha: float = 1e-2, nstrat: int = 3) -> List[Dict[str, List[float]]]:
    coords = embeddings.cpu().numpy().astype(np.float64)
    dists_sorted = sorted_distance_matrix(coords)
    return run_fiber_bundle_test_from_sorted_dists(
        dists_sorted,
        vol_min=vol_min,
        vol_max=vol_max,
        ws=ws,
        alpha=alpha,
        nstrat=nstrat,
    )


def summarize_stratifications(results: List[Dict[str, List[float]]], alpha: float = 1e-2) -> Dict[str, float]:
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
        "num_tokens": len(results), "tokens_with_strata": len(first_dims),
        "mean_dim": float(np.mean(first_dims)) if first_dims else float("nan"),
        "median_dim": float(np.median(first_dims)) if first_dims else float("nan"),
        "min_pvalue": float(np.min(min_pvals)) if min_pvals else float("nan"),
        "max_pvalue": float(np.max(min_pvals)) if min_pvals else float("nan"),
        "mean_irregularity": float(np.mean(irr_scores)) if irr_scores else float("nan"),
        "max_irregularity": float(np.max(irr_scores)) if irr_scores else float("nan"),
        "irregular_ratio": irregular_tokens / len(results) if results else float("nan"),
    }


def compute_class_dim_means(fiber_results: List[Dict], labels: torch.Tensor, num_classes: int) -> tuple[List[float], List[int]]:
    dims = np.array([res["dimensions"][0] if res and res.get("dimensions") else np.nan for res in fiber_results], dtype=np.float64)
    if dims.size == 0:
        return [float("nan")] * num_classes, [0] * num_classes
    lbls = labels[:dims.shape[0]]
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
    buckets = [[] for _ in range(num_classes)]
    for d, l in zip(dims.tolist(), lbl_np.tolist()):
        if math.isfinite(d) and 0 <= l < num_classes:
            buckets[l].append(d)
    return [float(np.mean(b)) if b else float("nan") for b in buckets], [len(b) for b in buckets]


def compute_neighborhood_dimensions(fiber_results: List[Dict], bboxes: torch.Tensor, neighborhood_size: int) -> List[float]:
    if not fiber_results or bboxes is None or bboxes.numel() == 0:
        return []
    dims = np.array([res["dimensions"][0] if res and res.get("dimensions") else np.nan for res in fiber_results], dtype=np.float64)
    b_np = bboxes[:len(dims)].cpu().numpy()
    centers = np.column_stack(((b_np[:, 0] + b_np[:, 2]) * 0.5, (b_np[:, 1] + b_np[:, 3]) * 0.5))
    dist = scipy.spatial.distance_matrix(centers, centers)
    radius = neighborhood_size * 0.5
    return [float(np.mean(dims[mask][np.isfinite(dims[mask])])) if np.any(mask := (dist[i] <= radius)) else float("nan") for i in range(len(dims))]


# ---------------------------------------------------------------------------
# Projection & Visualization Helpers
# ---------------------------------------------------------------------------
def project_embeddings_3d(embeddings: torch.Tensor) -> np.ndarray:
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=3)
    return (centered @ v[:, :3]).cpu().numpy()


def tsne_embeddings_3d(embeddings: torch.Tensor, perplexity: float = 30.0, seed: int = 42, max_points: int = 2048) -> tuple[np.ndarray, np.ndarray] | None:
    if not HAS_TSNE:
        return None
    emb_np = embeddings.cpu().numpy()
    n = emb_np.shape[0]
    if n > max_points:
        idx = np.random.default_rng(seed).choice(n, size=max_points, replace=False)
        emb_np = emb_np[idx]
    else:
        idx = np.arange(n)
    tsne = TSNE(n_components=3, perplexity=min(perplexity, max(5, len(emb_np) - 1)), init="pca", learning_rate="auto", random_state=seed)
    return tsne.fit_transform(emb_np), idx


def extract_patch_image(img_tensor: torch.Tensor, bbox: torch.Tensor, upscale: int = 128) -> Image.Image:
    np_img = img_tensor.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    h, w = np_img.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox.tolist()]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    patch = np_img[y0:y1, x0:x1, :] if x1 > x0 and y1 > y0 else np_img
    pil = Image.fromarray((patch * 255).astype("uint8"))
    if upscale and (pil.width < upscale or pil.height < upscale):
        pil = pil.resize((upscale, upscale), resample=getattr(Image, "Resampling", Image).BILINEAR)
    return pil


def add_heatmap_patch(img_tensor: torch.Tensor, bbox: torch.Tensor, value: float, max_value: float = 5.0,
                      neigh_value: float | None = None, neigh_max: float = 10.0, neighborhood_size: float | None = None) -> Image.Image:
    np_img = img_tensor.permute(1, 2, 0).clamp(0, 1).numpy()
    h, w = np_img.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox.tolist()]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return Image.fromarray((np_img * 255).astype("uint8"))

    if neigh_value is not None and math.isfinite(neigh_value) and neighborhood_size:
        half = neighborhood_size * 0.5
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        nbx0, nby0 = int(max(0, cx - half)), int(max(0, cy - half))
        nbx1, nby1 = int(min(w, cx + half)), int(min(h, cy + half))
        if nbx1 > nbx0 and nby1 > nby0:
            blue_alpha = 0.15 + 0.45 * max(0, min(1, neigh_value / neigh_max))
            np_img[nby0:nby1, nbx0:nbx1, :] = (1 - blue_alpha) * np_img[nby0:nby1, nbx0:nbx1, :] + blue_alpha * np.array([0.2, 0.4, 1.0])

    norm = max(0, min(1, value / max_value))
    color, alpha = np.array([1.0, norm, 0.0]), 0.25 + 0.45 * norm
    np_img[y0:y1, x0:x1, :] = (1 - alpha) * np_img[y0:y1, x0:x1, :] + alpha * color
    return Image.fromarray((np_img * 255).astype("uint8"))


def _make_patch_grid(patches: List[Image.Image], *, cols: int = 8, pad: int = 2, bg: tuple = (10, 10, 10)) -> Image.Image:
    if not patches:
        return Image.new("RGB", (64, 64), bg)
    w, h = patches[0].size
    rows = math.ceil(len(patches) / cols)
    grid = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), bg)
    for i, p in enumerate(patches):
        grid.paste(p, (pad + (i % cols) * (w + pad), pad + (i // cols) * (h + pad)))
    return grid


# ---------------------------------------------------------------------------
# Clustering & Polysemy Utilities
# ---------------------------------------------------------------------------
def _shannon_entropy_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    ps = counts.astype(np.float64) / total
    ps = ps[ps > 0]
    return float(-np.sum(ps * np.log(ps)))


def _torch_kmeans(x: torch.Tensor, k: int, *, iters: int = 15, seed: int = 0, device: torch.device | None = None) -> torch.Tensor:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device=device, dtype=torch.float32)
    n, d = x.shape
    k = max(2, min(k, n))
    g = torch.Generator(device=device).manual_seed(seed)
    c = x[torch.randperm(n, generator=g, device=device)[:k]].clone()
    for _ in range(iters):
        d2 = (x * x).sum(1, keepdim=True) + (c * c).sum(1).unsqueeze(0) - 2 * (x @ c.T)
        a = d2.argmin(dim=1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
    return c.cpu()


def _assign_centroids(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    x, c = x.float(), centroids.float()
    d2 = (x * x).sum(1, keepdim=True) + (c * c).sum(1).unsqueeze(0) - 2 * (x @ c.T)
    return d2.argmin(dim=1)


def _cluster_label_entropy(ids: np.ndarray, labels: np.ndarray, num_classes: int) -> Dict[int, Dict[str, float]]:
    out = {}
    for cid in np.unique(ids):
        m = ids == cid
        if not m.any():
            continue
        labs = labels[m]
        if labs.ndim == 2:
            counts_u = np.sum(labs > 0, axis=0).astype(np.float64)
        else:
            counts_u = np.bincount(labs.astype(np.int64), minlength=num_classes).astype(np.float64)
        ent = _shannon_entropy_from_counts(counts_u + 1.0)
        out[int(cid)] = {"count": float(counts_u.sum()), "label_entropy": ent,
                         "unique_labels": float((counts_u > 0).sum()), "top_label": float(np.argmax(counts_u))}
    return out


@torch.no_grad()
def _sample_patch_embeddings_from_loader(*, model: torch.nn.Module, loader: DataLoader,
                                          device: torch.device, max_tokens: int = 50000) -> torch.Tensor:
    model.eval()
    chunks, seen = [], 0
    for batch in loader:
        feats = model.forward_features(batch[0].to(device))
        start = 2 if getattr(model, "has_dist_token", False) else 1
        flat = feats[:, start:, :].detach().cpu().reshape(-1, feats.shape[-1])
        chunks.append(flat)
        seen += flat.shape[0]
        if seen >= max_tokens:
            break
    return torch.cat(chunks, 0)[:max_tokens] if chunks else torch.empty(0, 1)


@torch.no_grad()
def _cluster_label_counts_from_loader(*, model: torch.nn.Module, loader: DataLoader, centroids: torch.Tensor,
                                       device: torch.device, num_classes: int, max_batches: int | None = None) -> np.ndarray:
    model.eval()
    K = centroids.shape[0]
    counts = np.zeros((K, num_classes), dtype=np.int64)
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        imgs, labels = batch[0].to(device), batch[1].to(device)
        feats = model.forward_features(imgs)
        start = 2 if getattr(model, "has_dist_token", False) else 1
        tokens = feats[:, start:, :]
        B, P, E = tokens.shape
        ids = _assign_centroids(tokens.reshape(B * P, E).cpu(), centroids).view(B, P).numpy()
        if labels.dim() == 2:
            y = (labels > 0).float().cpu().unsqueeze(1).expand(B, P, num_classes).reshape(B * P, num_classes).numpy()
            for i, cid in enumerate(ids.flat):
                counts[cid] += y[i].astype(np.int64)
        else:
            flat_lbls = labels.cpu().numpy().astype(np.int64).repeat(P)
            np.add.at(counts, (ids.flat, flat_lbls), 1)
    return counts


def _stats_from_count_matrix(counts: np.ndarray, *, smooth: float = 1.0) -> Dict[int, Dict[str, float]]:
    out = {}
    for cid in range(counts.shape[0]):
        cts = counts[cid].astype(np.float64)
        total = cts.sum()
        if total <= 0:
            continue
        out[cid] = {"count": total, "label_entropy": _shannon_entropy_from_counts(cts + smooth),
                    "unique_labels": float((cts > 0).sum()), "top_label": float(np.argmax(cts))}
    return out


def compute_token_polysemy_entropy_scores(
    embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if embeddings.numel() == 0:
        return np.array([]), np.array([]), np.array([])
    n = embeddings.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64), np.full(n, -1, dtype=np.int64)
    k = max(1, min(int(k), n - 1))

    emb = embeddings.float()
    emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
    sim = (emb @ emb.T).cpu()
    sim.fill_diagonal_(-1e9)
    knn = sim.topk(k, dim=1).indices.cpu().numpy()

    ent = np.zeros(n, dtype=np.float64)
    top_share = np.zeros(n, dtype=np.float64)
    top_label = np.full(n, -1, dtype=np.int64)
    if isinstance(labels, torch.Tensor) and labels.dim() == 2:
        lbl = labels.cpu().numpy()
        for i in range(n):
            counts = np.sum(lbl[knn[i]] > 0, axis=0).astype(np.float64)
            total = float(np.sum(counts))
            ent[i] = _shannon_entropy_from_counts(counts + 1.0)
            if total > 0:
                top_label[i] = int(np.argmax(counts))
                top_share[i] = float(np.max(counts) / max(1.0, total))
            else:
                top_share[i] = 0.0
    else:
        lbl = labels.cpu().numpy().astype(np.int64).reshape(-1)
        for i in range(n):
            counts = np.bincount(lbl[knn[i]], minlength=num_classes).astype(np.float64)
            total = float(np.sum(counts))
            ent[i] = _shannon_entropy_from_counts(counts + 1.0)
            if total > 0:
                top_label[i] = int(np.argmax(counts))
                top_share[i] = float(np.max(counts) / max(1.0, total))
            else:
                top_share[i] = 0.0
    return ent, top_share, top_label


def _normalize_token_mask(token_mask: np.ndarray | List[int] | None, n: int) -> np.ndarray:
    if token_mask is None:
        return np.ones(n, dtype=np.bool_)
    if isinstance(token_mask, np.ndarray) and token_mask.dtype == np.bool_:
        mask = np.zeros(n, dtype=np.bool_)
        if token_mask.size:
            mask[:min(n, token_mask.size)] = token_mask[:n]
        return mask
    mask = np.zeros(n, dtype=np.bool_)
    for idx in token_mask:
        if 0 <= int(idx) < n:
            mask[int(idx)] = True
    return mask


def select_polysemy_entropy_images(
    *, image_ids: torch.Tensor, entropies: np.ndarray, bboxes: torch.Tensor, top_k_images: int = 9,
    token_mask: np.ndarray | List[int] | None = None
) -> List[Dict[str, Any]]:
    if entropies.size == 0:
        return []
    img_ids = image_ids.cpu().numpy().astype(np.int64).reshape(-1)
    ent = entropies.astype(np.float64)
    mask = _normalize_token_mask(token_mask, ent.shape[0])
    sums, counts, first_idx = {}, {}, {}
    for i, img_id in enumerate(img_ids):
        if not mask[i]:
            continue
        sums[img_id] = sums.get(img_id, 0.0) + float(ent[i])
        counts[img_id] = counts.get(img_id, 0) + 1
        first_idx.setdefault(img_id, i)
    scored = [(img_id, sums[img_id] / max(1, counts[img_id])) for img_id in sums]
    scored.sort(key=lambda x: x[1], reverse=True)
    picked = scored[: max(1, top_k_images)]
    selected: List[Dict[str, Any]] = []
    for img_id, mean_entropy in picked:
        idxs = np.where((img_ids == img_id) & mask)[0]
        if idxs.size == 0:
            continue
        base_idx = first_idx.get(img_id, int(idxs[0]))
        ent_sel = ent[idxs]
        top_local = int(np.nanargmax(ent_sel))
        top_idx = int(idxs[top_local])
        top_bbox = bboxes[top_idx].cpu().numpy()
        selected.append({
            "image_id": int(img_id),
            "mean_entropy": float(mean_entropy),
            "max_entropy": float(ent_sel[top_local]) if ent_sel.size else float("nan"),
            "base_idx": int(base_idx),
            "token_id": int(top_idx),
            "bbox": top_bbox,
        })
    return selected


# ---------------------------------------------------------------------------
# Polysemy & Ablation Helpers
# ---------------------------------------------------------------------------
def compute_token_polysemy_for_anchors(*, embeddings: torch.Tensor, labels: torch.Tensor, pred_labels: torch.Tensor | None,
                                        images: torch.Tensor, bboxes: torch.Tensor, dataset: str, anchor_ids: List[int],
                                        k: int, grid_cols: int, out_dir: Path, prefix: str,
                                        class_names: List[str] | None = None) -> Dict[str, Any]:
    if embeddings.numel() == 0 or not anchor_ids:
        return {"anchors": [], "paths": {}}
    denorm = denormalize_images(images, dataset).cpu()
    emb = embeddings.float()
    emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
    lbl_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else None
    pred_np = pred_labels.cpu().numpy() if isinstance(pred_labels, torch.Tensor) else None
    paths, anchors_out = {}, []

    for anchor in anchor_ids:
        if anchor < 0 or anchor >= emb.shape[0]:
            continue
        sims = torch.mv(emb, emb[anchor])
        sims[anchor] = -1e9
        nn = torch.topk(sims, k=min(k, emb.shape[0] - 1), largest=True).indices.cpu().tolist()
        patch_imgs = [extract_patch_image(denorm[anchor], bboxes[anchor])] if anchor < len(bboxes) else []
        patch_imgs += [extract_patch_image(denorm[j], bboxes[j]) for j in nn if j < len(bboxes)]
        grid = _make_patch_grid(patch_imgs, cols=grid_cols)
        out_path = out_dir / f"{prefix}_polysemy_token_{anchor:05d}.png"
        grid.save(out_path)
        paths[f"polysemy/token_{anchor:05d}/neighbors_grid"] = out_path

        metrics = {"token_id": anchor, "k": len(nn)}
        if lbl_np is not None and nn:
            neigh_lbls = lbl_np[nn]
            counts = np.sum(neigh_lbls > 0, axis=0) if neigh_lbls.ndim == 2 else np.bincount(neigh_lbls.astype(np.int64))
            total = float(np.sum(counts))
            top_idx = np.argsort(counts)[::-1][:3] if counts.size else np.array([], dtype=np.int64)
            top_labels = []
            for idx in top_idx:
                if counts[idx] <= 0:
                    continue
                name = class_names[idx] if class_names and 0 <= idx < len(class_names) else str(int(idx))
                top_labels.append({
                    "id": int(idx),
                    "name": name,
                    "count": int(counts[idx]),
                    "fraction": float(counts[idx] / max(1.0, total)),
                })
            metrics.update({
                "label_entropy": _shannon_entropy_from_counts(counts),
                "unique_labels": int((counts > 0).sum()),
                "top_label": int(np.argmax(counts)) if counts.size else -1,
                "top_label_share": float(counts[top_idx[0]] / max(1.0, total)) if top_idx.size else 0.0,
                "top_labels": top_labels,
            })
            if class_names and 0 <= metrics["top_label"] < len(class_names):
                metrics["top_label_name"] = class_names[metrics["top_label"]]
        anchors_out.append(metrics)
    return {"anchors": anchors_out, "paths": paths}


def make_polysemy_gallery(polysemy_result: Dict[str, Any], *, out_dir: Path, prefix: str,
                          top_k: int = 8, cols: int = 2) -> List[Dict[str, Any]]:
    anchors = [a for a in polysemy_result.get("anchors", []) if a and "label_entropy" in a]
    if not anchors:
        return []
    anchors.sort(key=lambda a: a.get("label_entropy", 0.0), reverse=True)
    paths = polysemy_result.get("paths", {})
    results: List[Dict[str, Any]] = []
    for a in anchors[:max(1, top_k)]:
        key = f"polysemy/token_{a['token_id']:05d}/neighbors_grid"
        path = paths.get(key)
        if not path:
            continue
        results.append({
            "path": path,
            "token_id": int(a.get("token_id", -1)),
            "label_entropy": float(a.get("label_entropy", 0.0)),
            "unique_labels": int(a.get("unique_labels", 0)),
            "top_label_share": float(a.get("top_label_share", 0.0)),
            "top_labels": a.get("top_labels", []),
            "k": int(a.get("k", 0)),
        })
    polysemy_result.setdefault("paths", {})["polysemy/top_entropy_sets"] = [
        r["path"] for r in results
    ]
    return results


def _format_label_text(label: torch.Tensor, class_names: List[str] | None, max_items: int = 3) -> str:
    if label is None:
        return "label n/a"
    if isinstance(label, torch.Tensor) and label.dim() == 0:
        idx = int(label.item())
        name = class_names[idx] if class_names and 0 <= idx < len(class_names) else str(idx)
        return f"label {name}"
    if isinstance(label, torch.Tensor) and label.dim() == 1:
        pos = (label > 0).nonzero().view(-1).tolist()
        if not pos:
            return "labels none"
        names = [class_names[i] if class_names and 0 <= i < len(class_names) else str(i) for i in pos[:max_items]]
        extra = f" +{len(pos) - max_items}" if len(pos) > max_items else ""
        return f"labels {', '.join(names)}{extra}"
    return "label n/a"


def _format_top_label(top_label: int, top_share: float, class_names: List[str] | None) -> str:
    if top_label < 0:
        return "top label n/a"
    name = class_names[top_label] if class_names and 0 <= top_label < len(class_names) else str(top_label)
    return f"top label {name} ({top_share:.0%})"


def _label_name(idx: int, class_names: List[str] | None) -> str:
    if class_names and 0 <= idx < len(class_names):
        return class_names[idx]
    return str(idx)


def _tensor01_to_pil(img01: torch.Tensor) -> Image.Image:
    np_img = (img01.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(np_img)


def _pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    np_img = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(np_img).permute(2, 0, 1)


def _draw_patch_box(img: Image.Image, bbox: np.ndarray, color: tuple = (255, 0, 0), width: int = 3) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    x0, y0, x1, y1 = [int(v) for v in bbox]
    for w in range(width):
        draw.rectangle((x0 - w, y0 - w, x1 + w, y1 + w), outline=color)
    return out


def _mask_patch(img: Image.Image, bbox: np.ndarray, mode: str = "gray") -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(out.width, x1), min(out.height, y1)
    if x1 <= x0 or y1 <= y0:
        return out
    mode = mode.lower()
    if mode == "blur":
        blurred = out.filter(ImageFilter.GaussianBlur(radius=2.0))
        patch = blurred.crop((x0, y0, x1, y1))
        out.paste(patch, (x0, y0))
        return out
    fill = (0, 0, 0) if mode == "black" else (160, 160, 160)
    draw.rectangle((x0, y0, x1, y1), fill=fill)
    return out


def make_polysemy_entropy_triptychs(
    *, images: torch.Tensor, bboxes: torch.Tensor, image_ids: torch.Tensor, entropies: np.ndarray,
    labels: torch.Tensor, class_names: List[str] | None, top_shares: np.ndarray, top_labels: np.ndarray,
    dataset: str, out_dir: Path, prefix: str, top_k_images: int = 9, min_width: int = 320,
    selection: List[Dict[str, Any]] | None = None, mask_mode: str = "gray",
    token_mask: np.ndarray | List[int] | None = None
) -> List[Dict[str, Any]]:
    if entropies.size == 0 or images.numel() == 0 or bboxes.numel() == 0:
        return []
    selection = selection or select_polysemy_entropy_images(
        image_ids=image_ids, entropies=entropies, bboxes=bboxes, top_k_images=top_k_images, token_mask=token_mask
    )
    denorm = denormalize_images(images, dataset).cpu()
    results: List[Dict[str, Any]] = []
    for item in selection:
        img_id = int(item["image_id"])
        base_idx = int(item["base_idx"])
        top_idx = int(item["token_id"])
        top_bbox = np.array(item["bbox"])
        label_text = _format_label_text(labels[base_idx], class_names)
        top_label_text = _format_top_label(int(top_labels[top_idx]), float(top_shares[top_idx]), class_names)

        base_img = _tensor01_to_pil(denorm[base_idx])
        img_box = _draw_patch_box(base_img, top_bbox)
        img_mask = _mask_patch(base_img, top_bbox, mode=mask_mode)

        trip_w = base_img.width * 3 + 2 * 8
        trip_h = base_img.height
        trip = Image.new("RGB", (trip_w, trip_h), (0, 0, 0))
        trip.paste(base_img, (0, 0))
        trip.paste(img_box, (base_img.width + 8, 0))
        trip.paste(img_mask, (2 * (base_img.width + 8), 0))

        if trip.width < min_width:
            scale = max(2, int(math.ceil(min_width / max(1, trip.width))))
            trip = trip.resize((trip.width * scale, trip.height * scale), resample=getattr(Image, "Resampling", Image).BILINEAR)

        out_path = out_dir / f"{prefix}_polysemy_entropy_triptych_{mask_mode}_{int(img_id):05d}.png"
        trip.save(out_path)
        results.append({
            "path": out_path,
            "image_id": int(img_id),
            "mean_entropy": float(item.get("mean_entropy", float("nan"))),
            "max_entropy": float(item.get("max_entropy", float("nan"))),
            "token_id": int(top_idx),
            "label_text": label_text,
            "top_label_text": top_label_text,
            "mask_mode": mask_mode,
        })

    return results


def compute_masked_classification_effects(
    *, model: torch.nn.Module, device: torch.device, images: torch.Tensor, labels: torch.Tensor,
    selection: List[Dict[str, Any]], dataset: str, mask_mode: str, class_names: List[str] | None,
    num_classes: int
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    if not selection:
        return [], {}
    model.eval()
    task_multilabel = isinstance(labels, torch.Tensor) and labels.dim() == 2
    mean_t, std_t = get_norm_stats(dataset, device=device, as_tensor=True)
    mean_t = mean_t.view(1, 3, 1, 1)
    std_t = std_t.view(1, 3, 1, 1)

    orig_batch = torch.stack([images[item["base_idx"]] for item in selection], dim=0).to(device)
    masked_list = []
    for item in selection:
        base_idx = int(item["base_idx"])
        bbox = np.array(item["bbox"])
        img01 = denormalize_images(images[base_idx].unsqueeze(0), dataset)[0]
        base_img = _tensor01_to_pil(img01)
        masked_img = _mask_patch(base_img, bbox, mode=mask_mode)
        masked01 = _pil_to_tensor01(masked_img).to(device)
        masked_norm = (masked01.unsqueeze(0) - mean_t) / std_t
        masked_list.append(masked_norm.squeeze(0))
    masked_batch = torch.stack(masked_list, dim=0)

    with torch.no_grad():
        logits_orig = model(orig_batch)
        logits_mask = model(masked_batch)

    if task_multilabel:
        probs_orig = torch.sigmoid(logits_orig)
        probs_mask = torch.sigmoid(logits_mask)
    else:
        probs_orig = torch.softmax(logits_orig, dim=-1)
        probs_mask = torch.softmax(logits_mask, dim=-1)

    per_image: List[Dict[str, Any]] = []
    pred_changed = []
    top1_drops = []
    true_drops = []

    for i, item in enumerate(selection):
        image_id = int(item["image_id"])
        token_id = int(item["token_id"])
        orig_probs = probs_orig[i]
        mask_probs = probs_mask[i]
        orig_top = int(torch.argmax(orig_probs).item())
        mask_top = int(torch.argmax(mask_probs).item())
        orig_top_prob = float(orig_probs[orig_top].item())
        mask_top_prob = float(mask_probs[mask_top].item())
        top1_drop = float(orig_top_prob - mask_probs[orig_top].item())
        pred_changed.append(float(orig_top != mask_top))
        top1_drops.append(top1_drop)

        true_drop = float("nan")
        true_label_name = "n/a"
        if task_multilabel:
            lbl = labels[item["base_idx"]]
            pos = (lbl > 0).to(orig_probs.device)
            if pos.any():
                true_prob = float(orig_probs[pos].mean().item())
                true_prob_mask = float(mask_probs[pos].mean().item())
                true_drop = true_prob - true_prob_mask
                true_label_name = "multi"
        else:
            true_label = int(labels[item["base_idx"]].item())
            true_prob = float(orig_probs[true_label].item())
            true_prob_mask = float(mask_probs[true_label].item())
            true_drop = true_prob - true_prob_mask
            true_label_name = _label_name(true_label, class_names)
        if math.isfinite(true_drop):
            true_drops.append(true_drop)

        per_image.append({
            "image_id": image_id,
            "token_id": token_id,
            "orig_pred": orig_top,
            "orig_pred_name": _label_name(orig_top, class_names),
            "orig_pred_prob": orig_top_prob,
            "mask_pred": mask_top,
            "mask_pred_name": _label_name(mask_top, class_names),
            "mask_pred_prob": mask_top_prob,
            "pred_changed": bool(orig_top != mask_top),
            "top1_drop": top1_drop,
            "true_drop": true_drop,
            "true_label_name": true_label_name,
        })

    agg = {
        "pred_change_rate": float(np.mean(pred_changed)) if pred_changed else float("nan"),
        "mean_top1_drop": float(np.mean(top1_drops)) if top1_drops else float("nan"),
        "mean_true_drop": float(np.mean(true_drops)) if true_drops else float("nan"),
        "mask_mode": mask_mode,
        "num_images": float(len(selection)),
    }
    return per_image, agg


def _compute_irregularity_scores(
    fiber_results: List[Dict[str, Any]], alpha: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(fiber_results)
    min_p = np.full(n, np.nan, dtype=np.float64)
    irr = np.full(n, np.nan, dtype=np.float64)
    rejected = np.zeros(n, dtype=np.bool_)
    for i, res in enumerate(fiber_results):
        if not res or not res.get("pvalues"):
            continue
        p = float(np.min(res["pvalues"]))
        min_p[i] = p
        irr[i] = -math.log10(p + 1e-12)
        rejected[i] = p < alpha
    return min_p, irr, rejected


def _singular_token_mask(fiber_results: List[Dict[str, Any]], alpha: float) -> np.ndarray:
    _, _, rejected = _compute_irregularity_scores(fiber_results, alpha)
    return rejected


def select_singular_token_indices(
    *, fiber_results: List[Dict[str, Any]], alpha: float, top_k: int
) -> List[int]:
    min_p, irr, rejected = _compute_irregularity_scores(fiber_results, alpha)
    if irr.size == 0 or not np.any(rejected):
        return []
    idxs = np.where(rejected)[0]
    order = np.argsort(-irr[idxs])
    picks = idxs[order][: max(1, top_k)]
    return [int(i) for i in picks if math.isfinite(min_p[int(i)])]


def make_polysemy_irregularity_plot(
    *, entropies: np.ndarray, fiber_results: List[Dict[str, Any]], out_dir: Path, prefix: str,
    alpha: float = 1e-2
) -> Tuple[Path | None, Dict[str, float]]:
    plt_mod = _require_matplotlib()
    if entropies.size == 0 or not fiber_results:
        return None, {}
    _, irr, rejected = _compute_irregularity_scores(fiber_results, alpha)
    n = min(entropies.shape[0], irr.shape[0])
    if n == 0:
        return None, {}
    ent = entropies[:n].astype(np.float64)
    irr = irr[:n]
    rejected = rejected[:n]
    mask = np.isfinite(ent) & np.isfinite(irr)
    if not np.any(mask):
        return None, {}
    ent = ent[mask]
    irr = irr[mask]
    rejected = rejected[mask]

    pearson_r, pearson_p = (float("nan"), float("nan"))
    spearman_r, spearman_p = (float("nan"), float("nan"))
    if ent.size > 2:
        pearson_r, pearson_p = scipy.stats.pearsonr(ent, irr)
        spearman_r, spearman_p = scipy.stats.spearmanr(ent, irr)

    ent_rej = ent[rejected]
    ent_ok = ent[~rejected]
    stats = {
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        "mean_entropy_reject": float(np.mean(ent_rej)) if ent_rej.size else float("nan"),
        "mean_entropy_non_reject": float(np.mean(ent_ok)) if ent_ok.size else float("nan"),
        "n_reject": float(ent_rej.size), "n_total": float(ent.size),
        "alpha": float(alpha),
    }

    fig, axes = plt_mod.subplots(1, 2, figsize=(9, 4))
    ax0, ax1 = axes
    ax0.scatter(ent[~rejected], irr[~rejected], s=14, alpha=0.7, label="non-reject", color="#4c78a8")
    if ent_rej.size:
        ax0.scatter(ent_rej, irr[rejected], s=16, alpha=0.85, label="reject", color="#e45756")
    ax0.set_xlabel("Polysemy entropy (kNN labels)")
    ax0.set_ylabel("Irregularity (-log10 p)")
    ax0.set_title(f"Entropy vs irregularity (r={pearson_r:.2f}, rho={spearman_r:.2f})")
    ax0.legend(fontsize=8, frameon=False)

    box_data = [ent_ok, ent_rej] if ent_rej.size else [ent_ok]
    ax1.boxplot(box_data, labels=["non-reject", "reject"] if ent_rej.size else ["non-reject"], showfliers=False)
    ax1.set_ylabel("Polysemy entropy")
    ax1.set_title("Entropy by fiber-bundle rejection")

    out_path = out_dir / f"{prefix}_polysemy_entropy_vs_irregularity.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt_mod.close(fig)
    return out_path, stats


def make_polysemy_entropy_scatter(polysemy_result: Dict[str, Any], *, out_dir: Path, prefix: str,
                                  annotate_top: int = 6) -> Path | None:
    plt_mod = _require_matplotlib()
    anchors = [a for a in polysemy_result.get("anchors", []) if a and "label_entropy" in a]
    if not anchors:
        return None
    ent = np.array([a.get("label_entropy", np.nan) for a in anchors], dtype=np.float64)
    share = np.array([a.get("top_label_share", np.nan) for a in anchors], dtype=np.float64)
    uniq = np.array([a.get("unique_labels", np.nan) for a in anchors], dtype=np.float64)
    ids = [a.get("token_id", -1) for a in anchors]

    mask = np.isfinite(ent) & np.isfinite(share)
    if not np.any(mask):
        return None
    ent, share, uniq = ent[mask], share[mask], uniq[mask]
    ids = [i for i, m in zip(ids, mask.tolist()) if m]

    fig, ax = plt_mod.subplots(figsize=(6, 4))
    sc = ax.scatter(share, ent, c=uniq, cmap="viridis", s=45, alpha=0.85)
    ax.set_xlabel("Top-label share")
    ax.set_ylabel("Label entropy")
    ax.set_title("Polysemy anchors: entropy vs top-label share")
    ax.set_xlim(0.0, 1.0)
    fig.colorbar(sc, ax=ax, shrink=0.8, label="unique labels")

    top_idx = np.argsort(-ent)[: max(1, annotate_top)]
    for i in top_idx:
        ax.text(share[i], ent[i], str(ids[i]), fontsize=7)

    out_path = out_dir / f"{prefix}_polysemy_entropy_scatter.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt_mod.close(fig)
    polysemy_result.setdefault("paths", {})["polysemy/entropy_scatter"] = out_path
    return out_path


@torch.no_grad()
def _eval_ablation_controls(*, model: torch.nn.Module, loader: DataLoader, centroids: torch.Tensor,
                             poly_cluster_ids: List[int], patch_size: int, img_size: int, device: torch.device,
                             batches: int = 10, seed: int = 0) -> Dict[str, float]:
    model.eval()
    rng = np.random.default_rng(seed)
    K = centroids.shape[0]
    poly_set = set(poly_cluster_ids)
    pool = [i for i in range(K) if i not in poly_set]
    rand_ids = rng.choice(pool if len(pool) >= len(poly_cluster_ids) else list(range(K)),
                          size=max(1, len(poly_cluster_ids)), replace=False).tolist()
    rand_set = set(rand_ids)

    stats = {"correct_base": 0, "correct_poly": 0, "correct_rc": 0, "correct_rp": 0,
             "total": 0, "n_images": 0, "masked_poly": 0, "masked_rc": 0, "masked_rp": 0,
             "flip_poly": 0, "flip_rc": 0, "flip_rp": 0, "base_correct": 0}

    for bi, batch in enumerate(loader):
        if bi >= batches:
            break
        imgs, labels = batch[0].to(device), batch[1].to(device)
        B = imgs.size(0)
        stats["n_images"] += B
        feats = model.forward_features(imgs)
        start = 2 if getattr(model, "has_dist_token", False) else 1
        tokens = feats[:, start:, :]
        P = tokens.shape[1]
        grid = int(math.sqrt(P))
        ids = _assign_centroids(tokens.reshape(B * P, -1).cpu(), centroids).view(B, P)

        logits = model(imgs)
        multilabel = labels.dim() == 2
        if multilabel:
            preds = (torch.sigmoid(logits) > 0.5)
            stats["correct_base"] += ((preds == (labels > 0)).sum().item())
            stats["total"] += labels.numel()
        else:
            preds = logits.argmax(-1)
            stats["correct_base"] += (preds == labels).sum().item()
            stats["total"] += labels.numel()

        masked_poly, masked_rc, masked_rp = imgs.clone(), imgs.clone(), imgs.clone()
        for b in range(B):
            poly_p = [p for p in range(P) if int(ids[b, p]) in poly_set]
            rc_p = [p for p in range(P) if int(ids[b, p]) in rand_set]
            rp_p = rng.choice(P, size=min(len(poly_p), P), replace=False).tolist() if poly_p else []
            stats["masked_poly"] += len(poly_p)
            stats["masked_rc"] += len(rc_p)
            stats["masked_rp"] += len(rp_p)
            for m, patches in [(masked_poly, poly_p), (masked_rc, rc_p), (masked_rp, rp_p)]:
                for p in patches:
                    r, c = divmod(p, grid)
                    m[b, :, r * patch_size:min(img_size, (r + 1) * patch_size),
                      c * patch_size:min(img_size, (c + 1) * patch_size)] = 0

        for m, key in [(masked_poly, "poly"), (masked_rc, "rc"), (masked_rp, "rp")]:
            logits_m = model(m)
            if multilabel:
                stats[f"correct_{key}"] += ((torch.sigmoid(logits_m) > 0.5) == (labels > 0)).sum().item()
            else:
                preds_m = logits_m.argmax(-1)
                stats[f"correct_{key}"] += (preds_m == labels).sum().item()
                base_ok = preds == labels
                stats["base_correct"] += base_ok.sum().item()
                stats[f"flip_{key}"] += ((preds_m != labels) & base_ok).sum().item()

    t, n = max(1, stats["total"]), max(1, stats["n_images"])
    bc = max(1, stats["base_correct"])
    return {
        "acc_base": stats["correct_base"] / t, "acc_drop_poly": (stats["correct_base"] - stats["correct_poly"]) / t,
        "acc_drop_random_clusters": (stats["correct_base"] - stats["correct_rc"]) / t,
        "acc_drop_random_patches": (stats["correct_base"] - stats["correct_rp"]) / t,
        "avg_masked_patches_poly": stats["masked_poly"] / n, "avg_masked_patches_random_clusters": stats["masked_rc"] / n,
        "avg_masked_patches_random_patches": stats["masked_rp"] / n, "base_correct_count": bc,
        "flip_rate_poly_on_correct": stats["flip_poly"] / bc, "flip_rate_random_clusters_on_correct": stats["flip_rc"] / bc,
        "flip_rate_random_patches_on_correct": stats["flip_rp"] / bc,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_progress(train_history: List[Dict], fiber_history: List[Dict], final_coords_3d: np.ndarray,
                  final_colors: np.ndarray, out_path: Path) -> None:
    plt_mod = _require_matplotlib()
    fig = plt_mod.figure(figsize=(18, 5))
    ax1, ax2 = fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    epochs = [m["epoch"] for m in train_history]
    ax1.plot(epochs, [m["train_acc"] for m in train_history], label="train acc")
    ax1.plot(epochs, [m["eval_acc"] for m in train_history], label="val acc")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("accuracy"); ax1.set_title("Training"); ax1.legend()
    fiber_epochs = [m["epoch"] for m in fiber_history]
    ax2.plot(fiber_epochs, [m["mean_dim"] for m in fiber_history], "o-", label="mean dim")
    ax2.plot(fiber_epochs, [m.get("mean_irregularity", np.nan) for m in fiber_history], "x-", label="irregularity")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("value"); ax2.set_title("Fiber Summary"); ax2.legend()
    sc = ax3.scatter(final_coords_3d[:, 0], final_coords_3d[:, 1], final_coords_3d[:, 2], c=final_colors, cmap="viridis", s=12, alpha=0.85)
    ax3.set_title("Embeddings (PCA 3D)"); ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
    fig.colorbar(sc, ax=ax3, shrink=0.6, label="dim")
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt_mod.close(fig)


def make_embedding_figure_3d(coords3d: np.ndarray, dims: np.ndarray, title: str = "Embeddings (PCA 3D)") -> plt.Figure:
    plt_mod = _require_matplotlib()
    fig = plt_mod.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(coords3d[:, 0], coords3d[:, 1], coords3d[:, 2], c=dims, cmap="viridis", s=10, alpha=0.85)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.colorbar(sc, ax=ax, shrink=0.6, label="dim"); fig.tight_layout()
    return fig


def make_embedding_figure_tsne(coords3d: np.ndarray, dims: np.ndarray) -> plt.Figure:
    return make_embedding_figure_3d(coords3d, dims, "Embeddings (t-SNE 3D)")


# ---------------------------------------------------------------------------
# Irregular Image Selection
# ---------------------------------------------------------------------------
def select_irregular_images(images: torch.Tensor, labels: torch.Tensor, fiber_results: List[Dict], dataset: str,
                            bboxes: torch.Tensor, neighborhood_dims: List[float] | None = None,
                            image_ids: torch.Tensor | None = None, class_names: List[str] | None = None,
                            image_mean_dims: Dict[int, float] | None = None, pred_labels: torch.Tensor | None = None,
                            top_k: int = 12) -> List[Dict[str, Any]]:
    irregs = []
    for idx, res in enumerate(fiber_results):
        if not res or not res.get("pvalues"):
            continue
        pval = res["pvalues"][0]
        irregs.append((
            -np.log10(pval + 1e-12),
            res["dimensions"][0] if res.get("dimensions") else np.nan,
            idx,
            neighborhood_dims[idx] if neighborhood_dims and idx < len(neighborhood_dims) else np.nan,
            int(image_ids[idx]) if image_ids is not None and idx < len(image_ids) else idx,
            int(pred_labels[idx]) if pred_labels is not None and idx < len(pred_labels) else -1
        ))
    irregs.sort(reverse=True, key=lambda x: x[0])
    picks = irregs[:top_k]
    if not picks:
        return []
    imgs = denormalize_images(images, dataset).cpu()
    outputs = []
    for irr, dim, idx, neigh_dim, img_id, pred_lbl in picks:
        lbl = labels[idx].cpu() if idx < len(labels) else None
        if isinstance(lbl, torch.Tensor) and lbl.dim() == 0:
            lbl_val = int(lbl.item())
            cls_name = class_names[lbl_val] if class_names and 0 <= lbl_val < len(class_names) else str(lbl_val)
        elif isinstance(lbl, torch.Tensor) and lbl.dim() == 1:
            pos = (lbl > 0).nonzero().view(-1).tolist()
            cls_name = ", ".join([class_names[i] if class_names and 0 <= i < len(class_names) else str(i) for i in pos[:6]])
            lbl_val = pos
        else:
            lbl_val, cls_name = -1, None
        pred_name = class_names[pred_lbl] if class_names and 0 <= pred_lbl < len(class_names) else None
        outputs.append({
            "img": imgs[idx], "irregularity": irr, "dim": dim, "neigh_dim": neigh_dim,
            "label": lbl_val, "label_name": cls_name, "pred_label": pred_lbl, "pred_label_name": pred_name,
            "token_id": idx, "image_id": img_id, "bbox": bboxes[idx],
            "image_mean_dim": image_mean_dims.get(img_id, np.nan) if image_mean_dims else np.nan
        })
    return outputs


# ---------------------------------------------------------------------------
# Main Orchestration: run_fiber_analysis_epoch
# ---------------------------------------------------------------------------
def run_fiber_analysis_epoch(
    *, epoch: int, embeddings: torch.Tensor, labels: torch.Tensor, images: torch.Tensor, bboxes: torch.Tensor,
    patch_indices: torch.Tensor | None = None, image_ids: torch.Tensor, pred_labels: torch.Tensor,
    num_classes: int, class_names: List[str] | None, dataset: str, base_dir: Path, analysis_dir: Path,
    embed_dir: Path, vol_min: int, vol_max: int, ws: int, alpha: float, nstrat: int, neighborhood_size: int,
    polysemy: bool = False, polysemy_k: int = 48, polysemy_anchors: int = 12, polysemy_grid_cols: int = 8,
    polysemy_invert: bool = False, polysemy_invert_steps: int = 200, polysemy_invert_restarts: int = 6,
    polysemy_invert_lr: float = 0.08, polysemy_invert_tv: float = 1e-3, polysemy_invert_l2: float = 1e-4,
    polysemy_invert_patch_only: bool = True, polysemy_invert_blur_every: int = 10, polysemy_invert_blur_sigma: float = 0.8,
    vit_token_polysemy: bool = False, vit_token_polysemy_k: int = 256, vit_token_polysemy_topk: int = 16,
    vit_token_polysemy_ablate: bool = True, vit_token_polysemy_ablate_batches: int = 10,
    vit_token_polysemy_min_count: int = 50, vit_token_polysemy_ablate_reps: int = 5,
    wandb_module=None, model: torch.nn.Module | None = None, device: torch.device | None = None,
    img_size: int | None = None, patch_size: int | None = None, val_loader: DataLoader | None = None
) -> Dict[str, Any]:
    """Run fiber bundle diagnostics for one epoch worth of embeddings."""

    # Save embeddings
    torch.save({"embeddings": embeddings, "labels": labels, "images": images, "bboxes": bboxes,
                "image_ids": image_ids, "pred_labels": pred_labels}, embed_dir / f"epoch_{epoch:03d}.pt")

    # Core fiber analysis
    fiber_results = run_fiber_bundle_test(embeddings, vol_min=vol_min, vol_max=vol_max, ws=ws, alpha=alpha, nstrat=nstrat)
    neighborhood_dims = compute_neighborhood_dimensions(fiber_results, bboxes, neighborhood_size)

    # Per-image mean dimension
    mean_dim_by_image: Dict[int, float] = {}
    count_by_image: Dict[int, int] = {}
    for dim_val, img_id in zip([r["dimensions"][0] if r.get("dimensions") else np.nan for r in fiber_results], image_ids):
        if math.isfinite(dim_val):
            k = int(img_id.item())
            mean_dim_by_image[k] = mean_dim_by_image.get(k, 0) + dim_val
            count_by_image[k] = count_by_image.get(k, 0) + 1
    for k in mean_dim_by_image:
        mean_dim_by_image[k] /= max(1, count_by_image[k])

    # Summary stats
    fiber_summary = summarize_stratifications(fiber_results, alpha=alpha)
    class_dim_means, class_dim_counts = compute_class_dim_means(fiber_results, labels, num_classes)
    fiber_summary.update({
        "class_dim_means": class_dim_means, "class_dim_counts": class_dim_counts,
        "mean_neighborhood_dim": float(np.mean([d for d in neighborhood_dims if math.isfinite(d)])) if neighborhood_dims else np.nan,
        "neighborhood_size": neighborhood_size, "epoch": epoch
    })

    # Save fiber results
    with open(base_dir / f"fiber_epoch_{epoch:03d}.json", "w") as fp:
        json.dump(to_serializable(fiber_results), fp, indent=2)

    # Projections
    final_coords_3d = project_embeddings_3d(embeddings)
    final_tsne_3d, tsne_idx = (None, None)
    try:
        result = tsne_embeddings_3d(embeddings)
        if result:
            final_tsne_3d, tsne_idx = result
    except Exception as e:
        print(f"[tsne] failed: {e}")
    final_dims = np.array([r["dimensions"][0] if r.get("dimensions") else np.nan for r in fiber_results])

    # Polysemy analysis
    polysemy_result = None
    if polysemy:
        try:
            finite_idx = np.where(np.isfinite(final_dims) & (final_dims > 0))[0]
            sorted_idx = finite_idx[np.argsort(final_dims[finite_idx])] if finite_idx.size else np.array([], dtype=np.int64)
            singular_mask = _singular_token_mask(fiber_results, alpha)
            anchors = select_singular_token_indices(
                fiber_results=fiber_results, alpha=alpha, top_k=polysemy_anchors
            )
            if not anchors:
                anchors = [item["token_id"] for item in select_irregular_images(
                    images, labels, fiber_results, dataset, bboxes, neighborhood_dims, image_ids, class_names, pred_labels=pred_labels,
                    image_mean_dims=mean_dim_by_image, top_k=polysemy_anchors
                )]
            if sorted_idx.size and len(anchors) < polysemy_anchors:
                anchors.extend([int(sorted_idx[0]), int(sorted_idx[sorted_idx.size // 2]), int(sorted_idx[-1])])
            anchors = list(dict.fromkeys(anchors))[:polysemy_anchors]
            polysemy_result = compute_token_polysemy_for_anchors(
                embeddings=embeddings, labels=labels, pred_labels=pred_labels, images=images, bboxes=bboxes,
                dataset=dataset, anchor_ids=anchors, k=polysemy_k, grid_cols=polysemy_grid_cols, out_dir=analysis_dir,
                prefix=f"epoch_{epoch:03d}", class_names=class_names
            )
            if polysemy_result is not None:
                singular_count = int(np.sum(singular_mask)) if singular_mask is not None else 0
                total_tokens = len(fiber_results)
                polysemy_result["singular_token_count"] = singular_count
                polysemy_result["singular_token_ratio"] = singular_count / total_tokens if total_tokens else float("nan")
            top_entropy_sets = make_polysemy_gallery(
                polysemy_result,
                out_dir=analysis_dir,
                prefix=f"epoch_{epoch:03d}",
                top_k=min(polysemy_anchors, 8),
                cols=2,
            )
            if top_entropy_sets:
                polysemy_result["top_entropy_sets"] = top_entropy_sets
            make_polysemy_entropy_scatter(
                polysemy_result,
                out_dir=analysis_dir,
                prefix=f"epoch_{epoch:03d}",
                annotate_top=min(polysemy_anchors, 6),
            )
            ent_scores, top_shares, top_labels = compute_token_polysemy_entropy_scores(
                embeddings=embeddings, labels=labels, num_classes=num_classes, k=polysemy_k
            )
            irreg_path, irreg_stats = make_polysemy_irregularity_plot(
                entropies=ent_scores,
                fiber_results=fiber_results,
                out_dir=analysis_dir,
                prefix=f"epoch_{epoch:03d}",
                alpha=alpha,
            )
            if irreg_path:
                polysemy_result.setdefault("paths", {})["polysemy/entropy_irregularity"] = irreg_path
                polysemy_result["entropy_irregularity_stats"] = irreg_stats
            selection_mask = singular_mask if singular_mask is not None and np.any(singular_mask) else None
            selection = select_polysemy_entropy_images(
                image_ids=image_ids, entropies=ent_scores, bboxes=bboxes, top_k_images=9, token_mask=selection_mask
            )
            selection_source = "singular_tokens" if selection_mask is not None else "all_tokens"
            if not selection and selection_mask is not None:
                selection = select_polysemy_entropy_images(
                    image_ids=image_ids, entropies=ent_scores, bboxes=bboxes, top_k_images=9
                )
                selection_source = "all_tokens"
            if polysemy_result is not None:
                polysemy_result["entropy_triptych_source"] = selection_source
            entropy_triptych_sets: Dict[str, List[Dict[str, Any]]] = {}
            mask_effects: Dict[str, Dict[str, Any]] = {}
            for mask_mode in ("gray", "black", "blur"):
                triptych_sets = make_polysemy_entropy_triptychs(
                    images=images,
                    bboxes=bboxes,
                    image_ids=image_ids,
                    entropies=ent_scores,
                    labels=labels,
                    class_names=class_names,
                    top_shares=top_shares,
                    top_labels=top_labels,
                    dataset=dataset,
                    out_dir=analysis_dir,
                    prefix=f"epoch_{epoch:03d}",
                    selection=selection,
                    mask_mode=mask_mode,
                )
                if triptych_sets:
                    polysemy_result.setdefault("paths", {})[f"polysemy/entropy_triptychs_{mask_mode}"] = [
                        item["path"] for item in triptych_sets
                    ]
                    entropy_triptych_sets[mask_mode] = triptych_sets
                if model and device and selection:
                    per_image, agg = compute_masked_classification_effects(
                        model=model,
                        device=device,
                        images=images,
                        labels=labels,
                        selection=selection,
                        dataset=dataset,
                        mask_mode=mask_mode,
                        class_names=class_names,
                        num_classes=num_classes,
                    )
                    if per_image:
                        mask_effects[mask_mode] = {"per_image": per_image, "aggregate": agg}
                        if mask_mode in entropy_triptych_sets:
                            effect_map = {
                                (entry.get("image_id"), entry.get("token_id")): entry for entry in per_image
                            }
                            for item in entropy_triptych_sets[mask_mode]:
                                eff = effect_map.get((item.get("image_id"), item.get("token_id")))
                                if eff:
                                    item.update(eff)
            if entropy_triptych_sets:
                polysemy_result["entropy_triptych_sets"] = entropy_triptych_sets
            if mask_effects:
                polysemy_result["mask_effects"] = mask_effects
        except Exception as e:
            print(f"[polysemy] failed: {e}")

    # ViT token polysemy
    vit_poly_metrics = None
    if vit_token_polysemy and model and device and img_size:
        try:
            analysis_loader = val_loader
            if val_loader and isinstance(getattr(val_loader, "sampler", None), DistributedSampler):
                analysis_loader = DataLoader(val_loader.dataset, batch_size=val_loader.batch_size, shuffle=False,
                                             num_workers=getattr(val_loader, "num_workers", 0), drop_last=False)
            x = _sample_patch_embeddings_from_loader(model=model, loader=analysis_loader, device=device, max_tokens=50000) if analysis_loader else embeddings
            centroids = _torch_kmeans(x, vit_token_polysemy_k, iters=15, seed=0)

            if analysis_loader:
                count_mat = _cluster_label_counts_from_loader(model=model, loader=analysis_loader, centroids=centroids,
                                                               device=device, num_classes=num_classes, max_batches=None)
                stats = _stats_from_count_matrix(count_mat, smooth=1.0)
            else:
                assign_ids = _assign_centroids(embeddings, centroids).cpu().numpy()
                stats = _cluster_label_entropy(assign_ids, labels.cpu().numpy(), num_classes)

            min_ct = vit_token_polysemy_min_count
            filtered = [(cid, st) for cid, st in stats.items() if st.get("count", 0) >= min_ct]
            if not filtered:
                filtered = list(stats.items())
            ranked = sorted(filtered, key=lambda kv: (kv[1].get("label_entropy", 0), kv[1].get("count", 0)), reverse=True)
            topk = [int(cid) for cid, _ in ranked[:vit_token_polysemy_topk]]

            vit_poly_metrics = {"top_clusters": topk, "mean_top_entropy": float(np.mean([stats[c]["label_entropy"] for c in topk if c in stats]))}

            if vit_token_polysemy_ablate and analysis_loader and patch_size:
                reps = vit_token_polysemy_ablate_reps
                results = [_eval_ablation_controls(model=model, loader=analysis_loader, centroids=centroids,
                                                    poly_cluster_ids=topk, patch_size=patch_size, img_size=img_size,
                                                    device=device, batches=vit_token_polysemy_ablate_batches, seed=epoch * 10000 + r)
                           for r in range(reps)]
                for key in ["acc_drop_poly", "acc_drop_random_clusters", "acc_drop_random_patches",
                            "flip_rate_poly_on_correct", "flip_rate_random_clusters_on_correct", "flip_rate_random_patches_on_correct"]:
                    vals = [r[key] for r in results]
                    vit_poly_metrics[f"vit_polysemy/{key}_mean"] = float(np.mean(vals))
                    vit_poly_metrics[f"vit_polysemy/{key}_std"] = float(np.std(vals))
        except Exception as e:
            print(f"[vit_polysemy] failed: {e}")

    # Wandb logging
    if wandb_module:
        try:
            plt_mod = _require_matplotlib()
            fig3d = make_embedding_figure_3d(final_coords_3d, final_dims)
            log_dict = {
                "epoch": epoch, "fiber/mean_dim": fiber_summary["mean_dim"], "fiber/median_dim": fiber_summary.get("median_dim", np.nan),
                "fiber/mean_irregularity": fiber_summary.get("mean_irregularity", np.nan), "fiber/irregular_ratio": fiber_summary.get("irregular_ratio", np.nan),
                "embeddings/pca_3d": wandb_module.Image(fig3d, caption=f"Epoch {epoch}")
            }
            plt_mod.close(fig3d)
            if final_tsne_3d is not None:
                fig_tsne = make_embedding_figure_tsne(final_tsne_3d, final_dims[tsne_idx] if tsne_idx is not None else final_dims)
                log_dict["embeddings/tsne_3d"] = wandb_module.Image(fig_tsne, caption=f"Epoch {epoch}")
                plt_mod.close(fig_tsne)
            if polysemy_result and polysemy_result.get("anchors"):
                ents = [a.get("label_entropy", np.nan) for a in polysemy_result["anchors"] if a]
                if ents:
                    log_dict["polysemy/mean_label_entropy"] = float(np.nanmean(ents))
                if "singular_token_count" in polysemy_result:
                    log_dict["polysemy/singular_token_count"] = polysemy_result.get("singular_token_count")
                if "singular_token_ratio" in polysemy_result:
                    log_dict["polysemy/singular_token_ratio"] = polysemy_result.get("singular_token_ratio")
                if polysemy_result.get("entropy_triptych_source"):
                    log_dict["polysemy/entropy_triptych_source"] = polysemy_result.get("entropy_triptych_source")
                top_sets = polysemy_result.get("top_entropy_sets", [])
                if top_sets:
                    log_dict["polysemy/top_entropy_sets"] = []
                    for item in top_sets:
                        top_labels = item.get("top_labels", [])
                        labels_str = ", ".join(
                            [f"{t.get('name', t.get('id'))} {t.get('fraction', 0.0):.0%}" for t in top_labels]
                        ) if top_labels else "n/a"
                        poly_desc = "polysemy: diverse neighbor labels" if item.get("unique_labels", 0) > 1 else "polysemy: low"
                        caption = (
                            f"token {item.get('token_id', -1):05d} | "
                            f"H {item.get('label_entropy', 0.0):.2f} | "
                            f"uniq {item.get('unique_labels', 0)} | "
                            f"top {item.get('top_label_share', 0.0):.0%} | "
                            f"k {item.get('k', 0)} | "
                            f"labels {labels_str} | {poly_desc}"
                        )
                        log_dict["polysemy/top_entropy_sets"].append(
                            wandb_module.Image(str(item["path"]), caption=caption)
                        )
                scatter_path = polysemy_result.get("paths", {}).get("polysemy/entropy_scatter")
                if scatter_path:
                    log_dict["polysemy/entropy_scatter"] = wandb_module.Image(str(scatter_path))
                irreg_path = polysemy_result.get("paths", {}).get("polysemy/entropy_irregularity")
                irreg_stats = polysemy_result.get("entropy_irregularity_stats", {})
                if irreg_path:
                    caption = (
                        f"entropy vs irregularity | "
                        f"pearson r {float(irreg_stats.get('pearson_r', float('nan'))):.2f} | "
                        f"spearman rho {float(irreg_stats.get('spearman_r', float('nan'))):.2f}"
                    )
                    log_dict["polysemy/entropy_irregularity"] = wandb_module.Image(str(irreg_path), caption=caption)
                if isinstance(irreg_stats, dict):
                    for key in ("pearson_r", "pearson_p", "spearman_r", "spearman_p",
                                "mean_entropy_reject", "mean_entropy_non_reject", "n_reject", "n_total", "alpha"):
                        if key in irreg_stats:
                            log_dict[f"polysemy/entropy_irregularity/{key}"] = irreg_stats[key]
                triptych_sets = polysemy_result.get("entropy_triptych_sets", {})
                if isinstance(triptych_sets, dict):
                    for mask_mode, items in triptych_sets.items():
                        if not items:
                            continue
                        log_dict[f"polysemy/entropy_triptychs_{mask_mode}"] = [
                            wandb_module.Image(
                                str(item["path"]),
                                caption=(
                                    f"img {item.get('image_id', -1)} | "
                                    f"{item.get('label_text', 'label n/a')} | "
                                    f"token {item.get('token_id', -1)} (single-token patch, highest entropy) | "
                                    f"max H {item.get('max_entropy', 0.0):.2f} | "
                                    f"{item.get('top_label_text', 'top label n/a')} | "
                                    f"polysemy cue: mixed neighbor labels | "
                                    f"pred {item.get('orig_pred_name', 'n/a')} {float(item.get('orig_pred_prob', float('nan'))):.2f} -> "
                                    f"{item.get('mask_pred_name', 'n/a')} {float(item.get('mask_pred_prob', float('nan'))):.2f} | "
                                    f"top1 drop {float(item.get('top1_drop', float('nan'))):.2f} | "
                                    f"true ({item.get('true_label_name', 'n/a')}) drop {float(item.get('true_drop', float('nan'))):.2f} | "
                                    f"mask {mask_mode}"
                                ),
                            )
                            for item in items
                        ]
                elif triptych_sets:
                    log_dict["polysemy/entropy_triptychs"] = [
                        wandb_module.Image(
                            str(item["path"]),
                            caption=(
                                f"img {item.get('image_id', -1)} | "
                                f"{item.get('label_text', 'label n/a')} | "
                                f"max H {item.get('max_entropy', 0.0):.2f} | "
                                f"{item.get('top_label_text', 'top label n/a')} | "
                                f"highlighted patch = token {item.get('token_id', -1)}"
                            ),
                        )
                        for item in triptych_sets
                    ]
                mask_effects = polysemy_result.get("mask_effects", {})
                if isinstance(mask_effects, dict):
                    for mask_mode, data in mask_effects.items():
                        agg = data.get("aggregate", {})
                        if not isinstance(agg, dict):
                            continue
                        for key in ("pred_change_rate", "mean_top1_drop", "mean_true_drop", "num_images"):
                            if key in agg:
                                log_dict[f"polysemy/mask_{mask_mode}/{key}"] = agg[key]
            if vit_poly_metrics:
                log_dict.update({k: v for k, v in vit_poly_metrics.items() if isinstance(k, str) and k.startswith("vit_polysemy/")})

            irregular = select_irregular_images(images, labels, fiber_results, dataset, bboxes, neighborhood_dims,
                                                 image_ids, class_names, mean_dim_by_image, pred_labels, top_k=12)
            if irregular:
                neigh_max = max([item["neigh_dim"] for item in irregular if math.isfinite(item["neigh_dim"])], default=1)
                log_dict["embeddings/irregular_samples"] = [
                    wandb_module.Image(add_heatmap_patch(item["img"], item["bbox"], item["irregularity"],
                                                         neigh_value=item["neigh_dim"], neigh_max=neigh_max, neighborhood_size=neighborhood_size),
                                       caption=f"dim {item['dim']:.2f}, irr {item['irregularity']:.2f}")
                    for item in irregular
                ]
            wandb_module.log(log_dict)
        except Exception as e:
            print(f"[wandb] logging failed: {e}")

    return {"fiber_summary": fiber_summary, "final_dims": final_dims, "final_coords_3d": final_coords_3d, "final_tsne_3d": final_tsne_3d}
