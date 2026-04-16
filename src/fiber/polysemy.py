"""Polysemy analysis: kNN entropy scores, anchor analysis, and triptych generation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import denormalize_images
from fiber.visualization import (
    _draw_patch_box,
    _format_label_text,
    _format_top_label,
    _label_name,
    _make_patch_grid,
    _mask_patch,
    _tensor01_to_pil,
    extract_patch_image,
)

from PIL import Image


# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------

def _shannon_entropy_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    ps = counts.astype(np.float64) / total
    ps = ps[ps > 0]
    return float(-np.sum(ps * np.log(ps)))


# ---------------------------------------------------------------------------
# Clustering utilities
# ---------------------------------------------------------------------------

def _torch_kmeans(
    x: torch.Tensor, k: int, *, iters: int = 15, seed: int = 0, device: torch.device | None = None
) -> torch.Tensor:
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


def _cluster_label_entropy(
    ids: np.ndarray, labels: np.ndarray, num_classes: int
) -> Dict[int, Dict[str, float]]:
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
        out[int(cid)] = {
            "count": float(counts_u.sum()),
            "label_entropy": ent,
            "unique_labels": float((counts_u > 0).sum()),
            "top_label": float(np.argmax(counts_u)),
        }
    return out


@torch.no_grad()
def _sample_patch_embeddings_from_loader(
    *, model: torch.nn.Module, loader: DataLoader, device: torch.device, max_tokens: int = 50000
) -> torch.Tensor:
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
def _cluster_label_counts_from_loader(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    centroids: torch.Tensor,
    device: torch.device,
    num_classes: int,
    max_batches: int | None = None,
) -> np.ndarray:
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


def _stats_from_count_matrix(
    counts: np.ndarray, *, smooth: float = 1.0
) -> Dict[int, Dict[str, float]]:
    out = {}
    for cid in range(counts.shape[0]):
        cts = counts[cid].astype(np.float64)
        total = cts.sum()
        if total <= 0:
            continue
        out[cid] = {
            "count": total,
            "label_entropy": _shannon_entropy_from_counts(cts + smooth),
            "unique_labels": float((cts > 0).sum()),
            "top_label": float(np.argmax(cts)),
        }
    return out


# ---------------------------------------------------------------------------
# Polysemy entropy scores
# ---------------------------------------------------------------------------

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
        lbl = labels.cpu().numpy().astype(np.int64).reshape(-1)
        for i in range(n):
            counts = np.bincount(lbl[knn[i]], minlength=num_classes).astype(np.float64)
            total = float(np.sum(counts))
            ent[i] = _shannon_entropy_from_counts(counts + 1.0)
            if total > 0:
                top_label[i] = int(np.argmax(counts))
                top_share[i] = float(np.max(counts) / max(1.0, total))
    return ent, top_share, top_label


# ---------------------------------------------------------------------------
# Token mask / selection
# ---------------------------------------------------------------------------

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
    *,
    image_ids: torch.Tensor,
    entropies: np.ndarray,
    bboxes: torch.Tensor,
    top_k_images: int = 9,
    token_mask: np.ndarray | List[int] | None = None,
) -> List[Dict[str, Any]]:
    if entropies.size == 0:
        return []
    img_ids = image_ids.cpu().numpy().astype(np.int64).reshape(-1)
    ent = entropies.astype(np.float64)
    mask = _normalize_token_mask(token_mask, ent.shape[0])
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    first_idx: Dict[int, int] = {}
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
# Anchor-level polysemy analysis
# ---------------------------------------------------------------------------

def compute_token_polysemy_for_anchors(
    *,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    pred_labels: torch.Tensor | None,
    images: torch.Tensor,
    image_ids: torch.Tensor,
    bboxes: torch.Tensor,
    dataset: str,
    anchor_ids: List[int],
    k: int,
    grid_cols: int,
    out_dir: Path,
    prefix: str,
    class_names: List[str] | None = None,
) -> Dict[str, Any]:
    if embeddings.numel() == 0 or not anchor_ids:
        return {"anchors": [], "paths": {}}
    denorm = denormalize_images(images, dataset).cpu()
    emb = embeddings.float()
    emb = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
    lbl_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else None
    paths: Dict[str, Any] = {}
    anchors_out: list[Dict[str, Any]] = []

    for anchor in anchor_ids:
        if anchor < 0 or anchor >= emb.shape[0]:
            continue
        sims = torch.mv(emb, emb[anchor])
        sims[anchor] = -1e9
        nn = torch.topk(sims, k=min(k, emb.shape[0] - 1), largest=True).indices.cpu().tolist()
        # Look up source images via image_ids
        anchor_img_id = int(image_ids[anchor])
        anchor_denorm = denorm[anchor_img_id] if anchor_img_id < denorm.shape[0] else denorm[0]
        patch_imgs = [extract_patch_image(anchor_denorm, bboxes[anchor])] if anchor < len(bboxes) else []
        for j in nn:
            if j < len(bboxes):
                j_img_id = int(image_ids[j])
                j_denorm = denorm[j_img_id] if j_img_id < denorm.shape[0] else denorm[0]
                patch_imgs.append(extract_patch_image(j_denorm, bboxes[j]))
        grid = _make_patch_grid(patch_imgs, cols=grid_cols)
        out_path = out_dir / f"{prefix}_polysemy_token_{anchor:05d}.png"
        grid.save(out_path)
        paths[f"polysemy/token_{anchor:05d}/neighbors_grid"] = out_path

        metrics: Dict[str, Any] = {"token_id": anchor, "k": len(nn)}
        if lbl_np is not None and nn:
            neigh_lbls = lbl_np[nn]
            counts = np.sum(neigh_lbls > 0, axis=0) if neigh_lbls.ndim == 2 else np.bincount(neigh_lbls.astype(np.int64))
            total = float(np.sum(counts))
            top_idx_arr = np.argsort(counts)[::-1][:3] if counts.size else np.array([], dtype=np.int64)
            top_labels_list: list[Dict[str, Any]] = []
            for idx in top_idx_arr:
                if counts[idx] <= 0:
                    continue
                name = class_names[idx] if class_names and 0 <= idx < len(class_names) else str(int(idx))
                top_labels_list.append({
                    "id": int(idx), "name": name,
                    "count": int(counts[idx]),
                    "fraction": float(counts[idx] / max(1.0, total)),
                })
            metrics.update({
                "label_entropy": _shannon_entropy_from_counts(counts),
                "unique_labels": int((counts > 0).sum()),
                "top_label": int(np.argmax(counts)) if counts.size else -1,
                "top_label_share": float(counts[top_idx_arr[0]] / max(1.0, total)) if top_idx_arr.size else 0.0,
                "top_labels": top_labels_list,
            })
            if class_names and 0 <= metrics["top_label"] < len(class_names):
                metrics["top_label_name"] = class_names[metrics["top_label"]]
        anchors_out.append(metrics)
    return {"anchors": anchors_out, "paths": paths}


def make_polysemy_gallery(
    polysemy_result: Dict[str, Any],
    *,
    out_dir: Path,
    prefix: str,
    top_k: int = 8,
    cols: int = 2,
) -> List[Dict[str, Any]]:
    anchors = [a for a in polysemy_result.get("anchors", []) if a and "label_entropy" in a]
    if not anchors:
        return []
    anchors.sort(key=lambda a: a.get("label_entropy", 0.0), reverse=True)
    paths = polysemy_result.get("paths", {})
    results: List[Dict[str, Any]] = []
    for a in anchors[: max(1, top_k)]:
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


def make_polysemy_entropy_triptychs(
    *,
    images: torch.Tensor,
    image_ids: torch.Tensor,
    bboxes: torch.Tensor,
    entropies: np.ndarray,
    labels: torch.Tensor,
    class_names: List[str] | None,
    top_shares: np.ndarray,
    top_labels: np.ndarray,
    dataset: str,
    out_dir: Path,
    prefix: str,
    top_k_images: int = 9,
    min_width: int = 320,
    selection: List[Dict[str, Any]] | None = None,
    mask_mode: str = "gray",
    token_mask: np.ndarray | List[int] | None = None,
) -> List[Dict[str, Any]]:
    if entropies.size == 0 or images.numel() == 0 or bboxes.numel() == 0:
        return []
    selection = selection or select_polysemy_entropy_images(
        image_ids=image_ids, entropies=entropies, bboxes=bboxes,
        top_k_images=top_k_images, token_mask=token_mask,
    )
    denorm = denormalize_images(images, dataset).cpu()
    results: List[Dict[str, Any]] = []
    for item in selection:
        img_id = int(item["image_id"])
        base_idx = int(item["base_idx"])
        top_idx = int(item["token_id"])
        top_bbox = np.array(item["bbox"])
        label_text = _format_label_text(labels[base_idx], class_names)
        top_label_text = _format_top_label(
            int(top_labels[top_idx]), float(top_shares[top_idx]), class_names
        )

        # Use image_ids to look up the source image
        source_img_id = int(image_ids[base_idx]) if base_idx < len(image_ids) else img_id
        source_img = denorm[source_img_id] if source_img_id < denorm.shape[0] else denorm[0]
        base_img = _tensor01_to_pil(source_img)
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
            trip = trip.resize(
                (trip.width * scale, trip.height * scale),
                resample=getattr(Image, "Resampling", Image).BILINEAR,
            )

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
