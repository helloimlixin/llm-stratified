"""Patch token collection from a model and DataLoader.

Tokens are collected with *deduplicated* image storage: only one copy of each
source image is kept, and a per-token ``image_ids`` tensor maps tokens back to
their source image in the unique-images buffer.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader


def _boxes_to_patch_multihot(
    boxes: List[tuple],
    *,
    grid: int,
    patch_px: int,
    num_classes: int,
) -> torch.Tensor:
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
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    patch_size: int,
    max_tokens: int | None = 256,
    show_progress: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Collect patch-token embeddings from the validation set.

    Returns
    -------
    embeddings : (N, D)
    labels     : (N, ...) per-token labels
    images     : (U, C, H, W) **unique** source images (U <= N)
    bboxes     : (N, 4) per-token patch bounding boxes
    patch_indices : (N,) patch index within its source image
    image_ids  : (N,) index into ``images`` for each token
    pred_labels : (N,) predicted class per source image
    """
    model.eval()
    base_ds = getattr(loader, "dataset", None)
    while hasattr(base_ds, "dataset"):
        base_ds = base_ds.dataset
    has_instance_boxes = hasattr(base_ds, "instances_after_eval_transform")

    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    unique_images: list[torch.Tensor] = []
    bboxes: list[torch.Tensor] = []
    patch_indices: list[torch.Tensor] = []
    image_ids: list[torch.Tensor] = []
    pred_labels: list[torch.Tensor] = []
    collected = 0
    image_count = 0  # number of unique images stored

    iterator = (
        tqdm(loader, desc="Collect val tokens", leave=False)
        if show_progress
        else loader
    )

    with torch.no_grad():
        for batch in iterator:
            if max_tokens is not None and collected >= max_tokens:
                break
            imgs, lbls = batch[0].to(device), batch[1]
            idxs = batch[2] if len(batch) > 2 else None
            feats = model.forward_features(imgs)
            logits = (
                model.tokens_to_logits(feats)
                if hasattr(model, "tokens_to_logits")
                else model.head(feats[:, 0])
            )
            preds = logits.argmax(dim=-1).cpu()
            start_idx = 2 if getattr(model, "has_dist_token", False) else 1
            patch_tokens = feats[:, start_idx:, :].cpu()
            B, P, E = patch_tokens.shape
            grid = int(math.sqrt(P))
            img_size_px = int(imgs.shape[-1])

            patch_labels_per_image = None
            if (
                has_instance_boxes
                and idxs is not None
                and isinstance(lbls, torch.Tensor)
                and lbls.dim() == 2
            ):
                try:
                    idx_list = (
                        idxs.detach().cpu().tolist()
                        if isinstance(idxs, torch.Tensor)
                        else list(idxs)
                    )
                    num_classes = int(lbls.shape[1])
                    patch_labels_per_image = [
                        _boxes_to_patch_multihot(
                            base_ds.instances_after_eval_transform(
                                int(idx_list[i]), img_size_px
                            ),
                            grid=grid,
                            patch_px=patch_size,
                            num_classes=num_classes,
                        )
                        for i in range(B)
                    ]
                except Exception:
                    patch_labels_per_image = None

            # Pre-compute per-patch tensors once per grid size
            rows_idx = torch.arange(grid).repeat_interleave(grid)
            cols_idx = torch.arange(grid).repeat(grid)
            all_bboxes = torch.stack(
                [cols_idx * patch_size, rows_idx * patch_size,
                 (cols_idx + 1) * patch_size, (rows_idx + 1) * patch_size],
                dim=1,
            ).to(dtype=torch.int32)
            all_patch_idx = torch.arange(P, dtype=torch.int32)

            for i in range(B):
                if max_tokens is not None and collected >= max_tokens:
                    break
                # Store this image once
                img_idx = image_count
                unique_images.append(imgs[i].cpu())
                image_count += 1

                remain = (max_tokens - collected) if max_tokens is not None else P
                take = min(P, remain)
                embeddings.append(patch_tokens[i, :take].clone())
                if patch_labels_per_image:
                    labels.append(patch_labels_per_image[i][:take])
                else:
                    lbl_i = lbls[i].cpu()
                    labels.append(lbl_i.unsqueeze(0).expand(take, *lbl_i.shape))
                image_ids.append(torch.full((take,), img_idx, dtype=torch.int32))
                bboxes.append(all_bboxes[:take])
                patch_indices.append(all_patch_idx[:take])
                pred_labels.append(preds[i].expand(take))
                collected += take

    if not embeddings:
        return (
            torch.empty(0, getattr(model, "embed_dim", 192)),
            torch.empty(0, dtype=torch.long),
            torch.empty(0),
            torch.empty(0, 4, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int64),
        )
    return (
        torch.cat(embeddings, dim=0),
        torch.cat(labels, dim=0),
        torch.stack(unique_images),
        torch.cat(bboxes, dim=0),
        torch.cat(patch_indices, dim=0),
        torch.cat(image_ids, dim=0),
        torch.cat(pred_labels, dim=0),
    )
