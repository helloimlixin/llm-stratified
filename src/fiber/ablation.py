"""Masked classification effects and ViT token polysemy ablation."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import get_dataset_normalization
from utils import denormalize_images
from fiber.polysemy import _assign_centroids
from fiber.plots import (
    _label_name,
    _mask_patch,
    _pil_to_tensor01,
    _tensor01_to_pil,
)


def compute_masked_classification_effects(
    *,
    model: torch.nn.Module,
    device: torch.device,
    images: torch.Tensor,
    image_ids: torch.Tensor,
    labels: torch.Tensor,
    selection: List[Dict[str, Any]],
    dataset: str,
    mask_mode: str,
    class_names: List[str] | None,
    num_classes: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Compute the effect of masking the highest-entropy patch on classification.

    ``images`` is the **unique** image buffer; ``image_ids`` maps tokens to images.
    """
    if not selection:
        return [], {}
    model.eval()
    task_multilabel = isinstance(labels, torch.Tensor) and labels.dim() == 2
    mean_t, std_t = get_dataset_normalization(dataset, device=device, as_tensor=True)
    mean_t = mean_t.view(1, 3, 1, 1)
    std_t = std_t.view(1, 3, 1, 1)

    # Build orig and masked batches using image_ids
    orig_list = []
    masked_list = []
    for item in selection:
        base_idx = int(item["base_idx"])
        bbox = np.array(item["bbox"])
        # Look up the unique source image
        src_img_id = int(image_ids[base_idx]) if base_idx < len(image_ids) else 0
        src_img = images[src_img_id] if src_img_id < images.shape[0] else images[0]
        orig_list.append(src_img.to(device))
        img01 = denormalize_images(src_img.unsqueeze(0), dataset)[0]
        base_img = _tensor01_to_pil(img01)
        masked_img = _mask_patch(base_img, bbox, mode=mask_mode)
        masked01 = _pil_to_tensor01(masked_img).to(device)
        masked_norm = (masked01.unsqueeze(0) - mean_t) / std_t
        masked_list.append(masked_norm.squeeze(0))

    orig_batch = torch.stack(orig_list, dim=0)
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
    pred_changed: list[float] = []
    top1_drops: list[float] = []
    true_drops: list[float] = []

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


@torch.no_grad()
def _eval_ablation_controls(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    centroids: torch.Tensor,
    poly_cluster_ids: List[int],
    patch_size: int,
    img_size: int,
    device: torch.device,
    batches: int = 10,
    seed: int = 0,
) -> Dict[str, float]:
    model.eval()
    rng = np.random.default_rng(seed)
    K = centroids.shape[0]
    poly_set = set(poly_cluster_ids)
    pool = [i for i in range(K) if i not in poly_set]
    rand_ids = rng.choice(
        pool if len(pool) >= len(poly_cluster_ids) else list(range(K)),
        size=max(1, len(poly_cluster_ids)),
        replace=False,
    ).tolist()
    rand_set = set(rand_ids)

    stats: Dict[str, int] = {
        "correct_base": 0, "correct_poly": 0, "correct_rc": 0, "correct_rp": 0,
        "total": 0, "n_images": 0, "masked_poly": 0, "masked_rc": 0, "masked_rp": 0,
        "flip_poly": 0, "flip_rc": 0, "flip_rp": 0, "base_correct": 0,
    }

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
            preds = torch.sigmoid(logits) > 0.5
            stats["correct_base"] += (preds == (labels > 0)).sum().item()
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
        "acc_base": stats["correct_base"] / t,
        "acc_drop_poly": (stats["correct_base"] - stats["correct_poly"]) / t,
        "acc_drop_random_clusters": (stats["correct_base"] - stats["correct_rc"]) / t,
        "acc_drop_random_patches": (stats["correct_base"] - stats["correct_rp"]) / t,
        "avg_masked_patches_poly": stats["masked_poly"] / n,
        "avg_masked_patches_random_clusters": stats["masked_rc"] / n,
        "avg_masked_patches_random_patches": stats["masked_rp"] / n,
        "base_correct_count": bc,
        "flip_rate_poly_on_correct": stats["flip_poly"] / bc,
        "flip_rate_random_clusters_on_correct": stats["flip_rc"] / bc,
        "flip_rate_random_patches_on_correct": stats["flip_rp"] / bc,
    }
