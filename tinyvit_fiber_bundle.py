#!/usr/bin/env python3
"""
TinyViT training + fiber bundle test using the stratified_estimator routines.

Runs a 100-epoch CIFAR-10 training (subsampled for speed on CPU), saves CLS
token embeddings every 10 epochs, applies the stratified estimator to detect
stratifications, and produces a visualization. Supports multi-GPU training
via Hugging Face Accelerate (use `accelerate launch`).
"""

import argparse
import os
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.spatial  # noqa: E402
import scipy.stats  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
try:
    import wandb  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore[assignment]

try:
    from sklearn.manifold import TSNE  # noqa: E402

    HAS_TSNE = True
except ImportError:
    TSNE = None
    HAS_TSNE = False
try:
    import timm  # noqa: E402
except ImportError:
    timm = None

from tinyvit import TinyViT, build_dataset, get_criterion, multilabel_accuracy  # noqa: E402


def resolve_patch_size(model) -> int | None:
    """Best-effort patch size extraction for TinyViT/timm ViT."""
    pe = getattr(model, "patch_embed", None)
    if pe is None:
        return None
    ps = getattr(pe, "patch_size", None)
    if ps is None:
        return None
    if isinstance(ps, (tuple, list)):
        return int(ps[0])
    if hasattr(ps, "numel") and ps.numel() > 0:
        return int(ps[0])
    try:
        return int(ps)
    except Exception:
        return None


class TimmViTWrapper(torch.nn.Module):
    """Wrap timm ViT to expose patch tokens via forward_features."""

    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for --timm-model; pip install timm")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.has_dist_token = getattr(self.backbone, "dist_token", None) is not None
        self.embed_dim = getattr(self.backbone, "embed_dim", None) or getattr(self.backbone, "num_features", None)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.backbone.patch_embed(x)
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
        if self.has_dist_token:
            dist_token = self.backbone.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        return x

    def tokens_to_logits(self, tokens):
        if hasattr(self.backbone, "forward_head"):
            return self.backbone.forward_head(tokens, pre_logits=False)
        cls_tok = tokens[:, 0]
        head = getattr(self.backbone, "head", None)
        return head(cls_tok) if head is not None else cls_tok

    def forward(self, x):
        feats = self.forward_features(x)
        return self.tokens_to_logits(feats)


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = 1337) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_subset_loaders(
    dataset_name: str = "CIFAR10",
    root: str = "./data",
    img_size: int | None = None,
    batch_size_train: int = 128,
    batch_size_test: int = 256,
    num_workers: int = 4,
    subset_train: int | None = None,
    subset_test: int | None = None,
    device: torch.device | None = None,
) -> Tuple[
    DataLoader,
    DataLoader,
    int,
    int,
    int,
    str,
]:
    """Build dataloaders, optionally restricting to a subset for quick runs."""
    train_ds, test_ds, num_classes, in_chans, final_img_size, task = build_dataset(
        dataset_name, root, img_size
    )

    if subset_train is not None:
        subset_train = min(subset_train, len(train_ds))
        train_ds = Subset(train_ds, list(range(subset_train)))
    if subset_test is not None:
        subset_test = min(subset_test, len(test_ds))
        test_ds = Subset(test_ds, list(range(subset_test)))

    pin = device is not None and device.type == "cuda"
    pw = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=pw,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size_test,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=pw,
    )
    return (
        train_loader,
        test_loader,
        num_classes,
        in_chans,
        final_img_size,
        task,
    )


# --------------------------------------------------------------------------- #
# Embedding + stratification utilities (adapted from kb1dds/stratified_estimator)
# --------------------------------------------------------------------------- #
def collect_patch_tokens(
    model: TinyViT,
    loader: DataLoader,
    device: torch.device,
    patch_size: int,
    max_tokens: int | None = 256,
    show_progress: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grab patch token embeddings (and labels + images + patch bboxes + image ids + preds) from a few batches."""
    model.eval()
    embeddings: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    images: List[torch.Tensor] = []
    bboxes: List[torch.Tensor] = []
    image_ids: List[torch.Tensor] = []
    pred_labels: List[torch.Tensor] = []
    collected = 0
    iterator = loader
    if show_progress:
        iterator = tqdm(loader, desc="Collect val tokens", leave=False)

    with torch.no_grad():
        for imgs, lbls in iterator:
            if max_tokens is not None and collected >= max_tokens:
                break
            imgs = imgs.to(device)
            feats = model.forward_features(imgs)
            if hasattr(model, "tokens_to_logits"):
                logits = model.tokens_to_logits(feats)
            else:
                logits = model.head(feats[:, 0])
            preds = logits.argmax(dim=-1).cpu()
            start_idx = 2 if getattr(model, "has_dist_token", False) else 1
            patch_tokens = feats[:, start_idx:, :].cpu()  # remove CLS (and dist if present)
            B, P, E = patch_tokens.shape
            grid = int(math.sqrt(P))
            for i in range(B):
                for p in range(P):
                    if max_tokens is not None and collected >= max_tokens:
                        break
                    embeddings.append(patch_tokens[i, p])
                    labels.append(lbls[i].cpu())
                    images.append(imgs[i].cpu())
                    row, col = divmod(p, grid)
                    x0 = col * patch_size
                    y0 = row * patch_size
                    x1 = x0 + patch_size
                    y1 = y0 + patch_size
                    bboxes.append(torch.tensor([x0, y0, x1, y1], dtype=torch.int32))
                    image_ids.append(torch.tensor(i, dtype=torch.int32))
                    pred_labels.append(preds[i])
                    collected += 1
                if max_tokens is not None and collected >= max_tokens:
                    break
    if len(embeddings) == 0:
        return (
            torch.empty(0, model.embed_dim),
            torch.empty(0, dtype=torch.long),
            torch.empty(0),
            torch.empty(0, 4, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int64),
        )
    emb_tensor = torch.stack(embeddings, dim=0)
    label_tensor = torch.stack(labels, dim=0)
    img_tensor = torch.stack(images, dim=0)
    bbox_tensor = torch.stack(bboxes, dim=0)
    imgid_tensor = torch.stack(image_ids, dim=0)
    pred_tensor = torch.stack(pred_labels, dim=0)
    if max_tokens is not None:
        emb_tensor = emb_tensor[:max_tokens]
        label_tensor = label_tensor[:max_tokens]
        img_tensor = img_tensor[:max_tokens]
        bbox_tensor = bbox_tensor[:max_tokens]
        imgid_tensor = imgid_tensor[:max_tokens]
        pred_tensor = pred_tensor[:max_tokens]
    return emb_tensor, label_tensor, img_tensor, bbox_tensor, imgid_tensor, pred_tensor


def geo_estimator(
    radii: np.ndarray,
    volumes: np.ndarray,
    npts: int,
    args: SimpleNamespace,
) -> Tuple[float, float, float]:
    """Estimate scaling coefficient, intrinsic dimension, and Ricci term."""

    rstack = np.column_stack((np.ones_like(radii), np.log(radii)))
    pointwise_lfit_data = np.linalg.lstsq(rstack, np.log(volumes), rcond=None)
    pointwise_lfit = pointwise_lfit_data[0]
    scaling_coeff = np.exp(pointwise_lfit[0]) / npts
    dimension = pointwise_lfit[1]
    if args.miller:
        scaling_coeff = scaling_coeff * np.exp(0.5 * pointwise_lfit_data[1][0] ** 2)
    if args.ricci:
        residuals = pointwise_lfit_data[1]
        ricci = np.mean(-residuals * 6 * (dimension + 2) / radii**2)
    else:
        ricci = 0.0
    return scaling_coeff, dimension, ricci


def stratification_test(
    radii: np.ndarray,
    volumes: np.ndarray,
    ws: int = 10,
    alpha: float = 1e-3,
) -> Tuple[Optional[int], float]:
    """Sliding-window Welch t-test to spot stratifications."""

    dimvec = np.gradient(np.log(volumes)) / np.gradient(np.log(radii))
    for w in range(2 * ws, dimvec.shape[0] - 2 * ws):
        t1 = dimvec[w - 2 * ws : w - ws]
        t1 = t1[np.logical_and(np.abs(t1) > 1e-5, np.isfinite(t1))]
        t2 = dimvec[w + ws : w + 2 * ws]
        t2 = t2[np.logical_and(np.abs(t2) > 1e-5, np.isfinite(t2))]
        pvalue = scipy.stats.ttest_ind(t1, t2, equal_var=False).pvalue
        if pvalue < alpha:
            return w, pvalue
    return None, 1.0


def estimate_stratifications(
    dists_sorted: np.ndarray,
    vol_min: int,
    vol_max: int,
    npts: int,
    args: SimpleNamespace,
    ws: int = 10,
    alpha: float = 1e-3,
) -> Dict[str, List[float]]:
    """Detect stratifications by repeatedly fitting geometric statistics."""

    radii = dists_sorted[vol_min:vol_max]
    volumes = np.arange(vol_min, vol_max)
    output: Dict[str, List[float]] = {
        "scaling_coeffs": [],
        "dimensions": [],
        "riccis": [],
        "strat_radii": [],
        "strat_volumes": [],
        "pvalues": [],
    }
    vol_min_current = np.argmax(radii > 1e-10)
    for _ in range(args.nstrat):
        vol_max_current = radii.shape[0]
        strat_idx, pvalue = stratification_test(
            radii[vol_min_current:vol_max_current],
            volumes[vol_min_current:vol_max_current],
            ws,
            alpha / args.nstrat,
        )
        if strat_idx is not None:
            vol_max_current = strat_idx + vol_min_current
        scaling_coeff, dimension, ricci = geo_estimator(
            radii[vol_min_current:vol_max_current],
            volumes[vol_min_current:vol_max_current],
            npts,
            args,
        )
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


def run_fiber_bundle_test(
    embeddings: torch.Tensor,
    vol_min: int = 8,
    vol_max: int = 64,
    ws: int = 8,
    alpha: float = 1e-2,
    nstrat: int = 3,
) -> List[Dict[str, List[float]]]:
    """Wrapper around estimate_stratifications on the CLS embeddings."""
    coords = embeddings.cpu().numpy().astype(np.float64)
    dists = scipy.spatial.distance_matrix(coords, coords)
    dists_sorted = np.sort(dists, axis=0)
    npts = dists_sorted.shape[0]
    if npts < 2:
        return []
    vol_max = min(vol_max, npts - 1)
    vol_min = min(vol_min, max(1, vol_max - 2))
    if vol_max - vol_min < 5:
        vol_min = max(1, vol_max - 5)
    args = SimpleNamespace(nstrat=nstrat, miller=True, ricci=False)
    outputs: List[Dict[str, List[float]]] = []
    for i in range(npts):
        outputs.append(
            estimate_stratifications(
                dists_sorted[:, i], vol_min, vol_max, npts, args, ws=ws, alpha=alpha
            )
        )
    return outputs


def predict_patch_class(
    patch_img: torch.Tensor,
    bbox: torch.Tensor,
    model: TinyViT,
    device: torch.device,
    img_size: int,
    dataset: str,
    class_names: List[str] | None = None,
) -> tuple[int, str | None]:
    """Predict class for a patch region by cropping, resizing, and running the model."""
    x0, y0, x1, y1 = [int(v) for v in bbox.tolist()]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = max(x0 + 1, x1), max(y0 + 1, y1)
    crop = patch_img[:, y0:y1, x0:x1].unsqueeze(0)  # 1,C,H,W
    crop = F.interpolate(crop, size=(img_size, img_size), mode="bilinear", align_corners=False)
    mean, std = get_norm_stats(dataset, device=device)
    crop = (crop.to(device) - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
    with torch.no_grad():
        logits = model(crop)
        pred = int(logits.argmax(dim=-1).item())
    name = class_names[pred] if class_names and 0 <= pred < len(class_names) else None
    return pred, name


def summarize_stratifications(results: List[Dict[str, List[float]]], alpha: float = 1e-2) -> Dict[str, float]:
    """Aggregate per-token stratification outputs."""
    first_dims, min_pvals, irr_scores = [], [], []
    irregular_tokens = 0
    for res in results:
        if not res or not res["dimensions"]:
            continue
        first_dims.append(res["dimensions"][0])
        min_p = min(res["pvalues"])
        min_pvals.append(min_p)
        irr_scores.append(-np.log10(min_p + 1e-12))
        if min_p < alpha:
            irregular_tokens += 1
    summary = {
        "num_tokens": len(results),
        "tokens_with_strata": len(first_dims),
        "mean_dim": float(np.mean(first_dims)) if first_dims else float("nan"),
        "median_dim": float(np.median(first_dims)) if first_dims else float("nan"),
        "min_pvalue": float(np.min(min_pvals)) if min_pvals else float("nan"),
        "max_pvalue": float(np.max(min_pvals)) if min_pvals else float("nan"),
        "mean_irregularity": float(np.mean(irr_scores)) if irr_scores else float("nan"),
        "max_irregularity": float(np.max(irr_scores)) if irr_scores else float("nan"),
        "irregular_ratio": float(irregular_tokens / len(results)) if len(results) > 0 else float("nan"),
    }
    return summary


def compute_class_dim_means(
    fiber_results: List[Dict[str, List[float]]], labels: torch.Tensor, num_classes: int
) -> tuple[List[float], List[int]]:
    """Compute per-class mean of first-stratum dimensions."""
    dims = [res["dimensions"][0] if res and res.get("dimensions") else float("nan") for res in fiber_results]
    lbls = labels[: len(dims)].cpu().numpy()
    buckets: List[List[float]] = [[] for _ in range(num_classes)]
    for d, l in zip(dims, lbls):
        if not math.isfinite(d):
            continue
        if 0 <= l < num_classes:
            buckets[int(l)].append(d)
    means = [float(np.mean(b)) if b else float("nan") for b in buckets]
    counts = [len(b) for b in buckets]
    return means, counts


def compute_neighborhood_dimensions(
    fiber_results: List[Dict[str, List[float]]], bboxes: torch.Tensor, neighborhood_size: int
) -> List[float]:
    """Average first-stratum dimensions within a square neighborhood (side = neighborhood_size) around each patch."""
    if not fiber_results or bboxes is None or bboxes.numel() == 0:
        return []

    dims = np.array(
        [res["dimensions"][0] if res and res.get("dimensions") else np.nan for res in fiber_results],
        dtype=np.float64,
    )
    b_np = bboxes[: len(dims)].cpu().numpy()
    centers = np.column_stack(((b_np[:, 0] + b_np[:, 2]) * 0.5, (b_np[:, 1] + b_np[:, 3]) * 0.5))
    dist = scipy.spatial.distance_matrix(centers, centers)
    radius = neighborhood_size * 0.5

    neigh_avgs: List[float] = []
    for i in range(len(dims)):
        mask = dist[i] <= radius
        vals = dims[mask]
        vals = vals[np.isfinite(vals)]
        neigh_avgs.append(float(np.mean(vals)) if vals.size > 0 else float("nan"))
    return neigh_avgs


def project_embeddings_2d(embeddings: torch.Tensor) -> np.ndarray:
    """(Deprecated) PCA to 2D — kept for compatibility, not used."""
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=2)
    coords2d = centered @ v[:, :2]
    return coords2d.cpu().numpy()


def project_embeddings_3d(embeddings: torch.Tensor) -> np.ndarray:
    """PCA to 3D for plotting."""
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=3)
    coords3d = centered @ v[:, :3]
    return coords3d.cpu().numpy()


def tsne_embeddings_3d(embeddings: torch.Tensor, perplexity: float = 30.0, seed: int = 42) -> np.ndarray | None:
    """t-SNE to 3D for plotting. Returns None if sklearn is unavailable."""
    if not HAS_TSNE:
        return None
    emb_np = embeddings.cpu().numpy()
    tsne = TSNE(
        n_components=3,
        perplexity=min(perplexity, max(5, len(emb_np) - 1)),
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    coords = tsne.fit_transform(emb_np)
    return coords


def get_norm_stats(dataset: str, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = dataset.upper()
    if dataset in ["CIFAR10", "CIFAR100"]:
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device)
        std = torch.tensor([0.2470, 0.2435, 0.2616], device=device)
    elif dataset in ["SVHN"]:
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device)
        std = torch.tensor([0.2470, 0.2435, 0.2616], device=device)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        std = torch.tensor([0.229, 0.224, 0.225], device=device)
    return mean, std


def to_serializable(obj):
    """Convert numpy types for JSON dumping."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64, np.float16)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    return obj


def denormalize_images(imgs: torch.Tensor, dataset: str) -> torch.Tensor:
    """Best-effort inverse normalization for visualization."""
    dataset = dataset.upper()
    if dataset in ["CIFAR10", "CIFAR100"]:
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=imgs.device).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616], device=imgs.device).view(1, 3, 1, 1)
    elif dataset in ["SVHN"]:
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=imgs.device).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616], device=imgs.device).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.5, 0.5, 0.5], device=imgs.device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=imgs.device).view(1, 3, 1, 1)
    out = imgs * std + mean
    return out.clamp(0, 1)


def add_red_bbox(img_tensor: torch.Tensor, thickness: int = 2) -> Image.Image:
    """Draw a thin red border around the image to highlight irregularity."""
    np_img = (img_tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")
    pil_img = Image.fromarray(np_img)
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size
    for t in range(thickness):
        draw.rectangle([t, t, w - 1 - t, h - 1 - t], outline=(255, 0, 0))
    return pil_img


def add_heatmap_patch(
    img_tensor: torch.Tensor,
    bbox: torch.Tensor,
    value: float,
    max_value: float = 5.0,
<<<<<<< Updated upstream
) -> Image.Image:
    """Apply a heatmap tint to the patch region without drawing a solid box."""
=======
    neigh_value: float | None = None,
    neigh_max: float = 10.0,
    neighborhood_size: float | None = None,
) -> Image.Image:
    """Apply a heatmap tint to the patch region (red/yellow = irregularity, blue = neighborhood dim)."""
>>>>>>> Stashed changes
    np_img = img_tensor.permute(1, 2, 0).clamp(0, 1).numpy()
    h, w, _ = np_img.shape
    x0, y0, x1, y1 = [int(v) for v in bbox.tolist()]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return Image.fromarray((np_img * 255).astype("uint8"))

    # Apply blue neighborhood first (so the patch overlay stays yellow/red)
    if (
        neigh_value is not None
        and math.isfinite(neigh_value)
        and neigh_max > 0
        and neighborhood_size is not None
        and neighborhood_size > 0
    ):
        half_neigh = float(neighborhood_size) * 0.5
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        nbx0 = int(max(0, cx - half_neigh))
        nby0 = int(max(0, cy - half_neigh))
        nbx1 = int(min(w, cx + half_neigh))
        nby1 = int(min(h, cy + half_neigh))
        if nbx1 > nbx0 and nby1 > nby0:
            blue_norm = max(0.0, min(1.0, neigh_value / neigh_max))
            blue_alpha = 0.15 + 0.45 * blue_norm
            blue_color = np.array([0.2, 0.4, 1.0], dtype=np.float32)
            nb_patch = np_img[nby0:nby1, nbx0:nbx1, :]
            np_img[nby0:nby1, nbx0:nbx1, :] = (1 - blue_alpha) * nb_patch + blue_alpha * blue_color

    # Normalize irregularity to [0,1] and pick color from red -> yellow
    norm = max(0.0, min(1.0, value / max_value))
    color = np.array([1.0, norm, 0.0], dtype=np.float32)  # RGB in [0,1]
    alpha = 0.25 + 0.45 * norm

    patch = np_img[y0:y1, x0:x1, :]
    np_img[y0:y1, x0:x1, :] = (1 - alpha) * patch + alpha * color
    return Image.fromarray((np_img * 255).astype("uint8"))


def extract_patch_image(
    img_tensor: torch.Tensor,
    bbox: torch.Tensor,
    upscale: int = 128,
) -> Image.Image:
    """Crop a patch from the denormalized tensor and optionally upscale it."""

    np_img = img_tensor.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    h, w, _ = np_img.shape
    x0, y0, x1, y1 = [int(v) for v in bbox.tolist()]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        patch = np_img
    else:
        patch = np_img[y0:y1, x0:x1, :]
    pil_patch = Image.fromarray((patch * 255).astype("uint8"))
    if upscale and (pil_patch.width < upscale or pil_patch.height < upscale):
        if hasattr(Image, "Resampling"):
            pil_patch = pil_patch.resize((upscale, upscale), resample=Image.Resampling.BILINEAR)
        else:  # pragma: no cover - Pillow < 9.1 fallback
            pil_patch = pil_patch.resize((upscale, upscale), resample=Image.BILINEAR)
    return pil_patch


def make_patch_panel(previews: List[Dict[str, Any]], title: str) -> plt.Figure | None:
    """Assemble a horizontal panel of patch previews."""

    if not previews:
        return None
    cols = len(previews)
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3))
    if cols == 1:
        axes = [axes]
    for ax, preview in zip(axes, previews):
        ax.imshow(np.asarray(preview["image"]))
        ax.set_title(preview["caption"], fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def prepare_dimension_extreme_visuals(
    dims: np.ndarray,
    images: torch.Tensor,
    bboxes: torch.Tensor,
    labels: torch.Tensor,
    pred_labels: torch.Tensor,
    dataset: str,
    class_names: List[str] | None = None,
    top_k: int = 6,
) -> Dict[str, Dict[str, Any]]:
    """Select sample patches for the lowest/highest dimension tokens."""

    visuals: Dict[str, Dict[str, Any]] = {}
    if dims.size == 0 or images.numel() == 0 or bboxes.numel() == 0:
        return visuals
    valid_mask = np.isfinite(dims)
    valid_indices = np.where(valid_mask)[0]
    if valid_indices.size == 0:
        return visuals
    sorted_indices = valid_indices[np.argsort(dims[valid_indices])]
    if sorted_indices.size == 0:
        return visuals
    k = min(top_k, sorted_indices.size)
    low_indices = sorted_indices[:k]
    high_indices = sorted_indices[-k:]
    denorm = denormalize_images(images, dataset).cpu()

    labels_cpu = labels.cpu() if isinstance(labels, torch.Tensor) else labels
    preds_cpu = pred_labels.cpu() if isinstance(pred_labels, torch.Tensor) else pred_labels

    def resolve_name(index: int, tensor: torch.Tensor, fallback: str) -> str:
        if not isinstance(tensor, torch.Tensor) or tensor.numel() <= index:
            return fallback
        sample = tensor[index]
        if sample.dim() == 0:
            value = int(sample.item())
            if class_names and 0 <= value < len(class_names):
                return class_names[value]
            return str(value)
        return fallback

    def build_previews(indices: np.ndarray, label: str) -> Dict[str, Any]:
        previews: List[Dict[str, Any]] = []
        for idx in indices:
            patch_img = extract_patch_image(denorm[idx], bboxes[idx])
            lbl = resolve_name(idx, labels_cpu, "-")
            pred = resolve_name(idx, preds_cpu, "-" if preds_cpu is not None else "")
            caption = f"idx {idx} | dim {dims[idx]:.2f} | lbl {lbl} | pred {pred}"
            previews.append({"image": patch_img, "caption": caption})
        panel = make_patch_panel(previews, f"{label} dimension patches")
        return {"previews": previews, "figure": panel}

    visuals["low"] = build_previews(low_indices, "Low") if low_indices.size > 0 else {}
    visuals["high"] = build_previews(high_indices[::-1], "High") if high_indices.size > 0 else {}
    return visuals


# --------------------------------------------------------------------------- #
# Accelerate-aware train/eval helpers
# --------------------------------------------------------------------------- #
def train_one_epoch_accelerate(
    model: TinyViT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    task_type: str = "multiclass",
    grad_clip: float | None = None,
    label_smoothing: float = 0.0,
    epoch: int = 0,
    show_progress: bool = True,
):
    model.train()
    criterion = get_criterion(task_type, label_smoothing)

    if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    loss_sum = torch.tensor(0.0, device=accelerator.device)
    acc_sum = torch.tensor(0.0, device=accelerator.device)
    count_sum = torch.tensor(0.0, device=accelerator.device)

    iterator = loader
    if show_progress and accelerator.is_main_process:
        iterator = tqdm(loader, desc=f"Train {epoch:03d}", leave=False)

    for imgs, labels in iterator:
        imgs = imgs.to(accelerator.device, non_blocking=True)
        if task_type == "multilabel":
            labels = labels.to(accelerator.device, non_blocking=True).float()
        else:
            labels = labels.to(accelerator.device, non_blocking=True).long()

        with accelerator.autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)

        accelerator.backward(loss)
        if grad_clip is not None:
            accelerator.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            bs = labels.size(0)
            loss_sum += loss.detach() * bs
            if task_type == "multilabel":
                acc = multilabel_accuracy(logits.detach(), labels)
                acc_sum += acc * bs
            else:
                preds = logits.argmax(dim=-1)
                acc_sum += (preds == labels).float().sum()
            count_sum += bs

    # Reduce across processes
    loss_sum = accelerator.reduce(loss_sum, reduction="sum")
    acc_sum = accelerator.reduce(acc_sum, reduction="sum")
    count_sum = accelerator.reduce(count_sum, reduction="sum")

    avg_loss = (loss_sum / count_sum).item() if count_sum > 0 else float("nan")
    if task_type == "multilabel":
        avg_acc = (acc_sum / count_sum).item() if count_sum > 0 else float("nan")
    else:
        avg_acc = (acc_sum / count_sum).item() if count_sum > 0 else float("nan")
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate_accelerate(
    model: TinyViT,
    loader: DataLoader,
    accelerator: Accelerator,
    task_type: str = "multiclass",
    label_smoothing: float = 0.0,
    show_progress: bool = True,
):
    model.eval()
    criterion = get_criterion(task_type, label_smoothing)

    loss_sum = torch.tensor(0.0, device=accelerator.device)
    acc_sum = torch.tensor(0.0, device=accelerator.device)
    count_sum = torch.tensor(0.0, device=accelerator.device)

    iterator = loader
    if show_progress and accelerator.is_main_process:
        iterator = tqdm(loader, desc="Eval", leave=False)

    for imgs, labels in iterator:
        imgs = imgs.to(accelerator.device, non_blocking=True)
        if task_type == "multilabel":
            labels = labels.to(accelerator.device, non_blocking=True).float()
        else:
            labels = labels.to(accelerator.device, non_blocking=True).long()

        with accelerator.autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)

        bs = labels.size(0)
        loss_sum += loss.detach() * bs
        if task_type == "multilabel":
            acc = multilabel_accuracy(logits, labels)
            acc_sum += acc * bs
        else:
            preds = logits.argmax(dim=-1)
            acc_sum += (preds == labels).float().sum()
        count_sum += bs

    loss_sum = accelerator.reduce(loss_sum, reduction="sum")
    acc_sum = accelerator.reduce(acc_sum, reduction="sum")
    count_sum = accelerator.reduce(count_sum, reduction="sum")

    avg_loss = (loss_sum / count_sum).item() if count_sum > 0 else float("nan")
    avg_acc = (acc_sum / count_sum).item() if count_sum > 0 else float("nan")
    return avg_loss, avg_acc


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_progress(
    train_history: List[Dict[str, float]],
    fiber_history: List[Dict[str, float]],
    final_coords_3d: np.ndarray,
    final_colors: np.ndarray,
    out_path: Path,
) -> None:
    """Render training curves, fiber summary, and 3D embedding scatter (projected)."""
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    # Training curves
    epochs = [m["epoch"] for m in train_history]
    ax1.plot(epochs, [m["train_acc"] for m in train_history], label="train acc")
    ax1.plot(epochs, [m["eval_acc"] for m in train_history], label="val acc")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy")
    ax1.set_title("TinyViT training")
    ax1.legend()

    # Fiber bundle summary
    fiber_epochs = [m["epoch"] for m in fiber_history]
    ax2.plot(fiber_epochs, [m["mean_dim"] for m in fiber_history], marker="o", label="mean dim")
    ax2.plot(
        fiber_epochs,
        [m.get("mean_irregularity", float("nan")) for m in fiber_history],
        marker="x",
        label="mean irregularity (-log10 p)",
    )
    ax2.plot(
        fiber_epochs,
        [m.get("irregular_ratio", float("nan")) for m in fiber_history],
        marker="s",
        label="irregular ratio",
    )
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("value")
    ax2.set_title("Stratified estimator summary")
    ax2.legend()

    # Scatter of final embeddings (3D PCA)
    sc = ax3.scatter(
        final_coords_3d[:, 0],
        final_coords_3d[:, 1],
        final_coords_3d[:, 2],
        c=final_colors,
        cmap="viridis",
        s=12,
        alpha=0.85,
    )
    ax3.set_title("CLS embeddings (PCA 3D, final epoch)")
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zticks([])
    fig.colorbar(sc, ax=ax3, shrink=0.6, label="first-stratum dimension")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_embedding_figure_3d(
    coords3d: np.ndarray,
    dims: np.ndarray,
    title: str = "CLS embeddings (PCA 3D)",
) -> plt.Figure:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(coords3d[:, 0], coords3d[:, 1], coords3d[:, 2], c=dims, cmap="viridis", s=10, alpha=0.85)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    fig.colorbar(sc, ax=ax, shrink=0.6, label="first-stratum dim")
    fig.tight_layout()
    return fig


def make_embedding_figure_tsne(coords3d: np.ndarray, dims: np.ndarray) -> plt.Figure:
    return make_embedding_figure_3d(coords3d, dims, title="CLS embeddings (t-SNE 3D)")


def plot_dimension_histogram(dims: np.ndarray, bins: int = 30) -> plt.Figure | None:
    """Return a histogram of first-stratum dimensions for the collected tokens."""

    valid = dims[np.isfinite(dims)]
    if valid.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(valid, bins=bins, color="steelblue", edgecolor="black", alpha=0.85)
    ax.set_xlabel("local dimension estimate")
    ax.set_ylabel("token count")
    ax.set_title("Token dimension distribution")
    fig.tight_layout()
    return fig


def plot_patch_count_curve(image_ids: torch.Tensor | np.ndarray) -> plt.Figure | None:
    """Plot sorted patch counts per image id to reveal sampling coverage."""

    if isinstance(image_ids, torch.Tensor):
        ids = image_ids.view(-1).cpu().numpy()
    else:
        ids = np.asarray(image_ids)
    if ids.size == 0:
        return None
    unique_ids, counts = np.unique(ids, return_counts=True)
    if unique_ids.size == 0:
        return None
    order = np.argsort(counts)[::-1]
    sorted_counts = counts[order]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(sorted_counts)), sorted_counts, marker="o", linewidth=1.2)
    ax.set_xlabel("image rank (sorted by patch count)")
    ax.set_ylabel("patch tokens collected")
    ax.set_title("Patches per image (sorted)")
    fig.tight_layout()
    return fig


def plot_dimension_radius_scatter(results: List[Dict[str, List[float]]]) -> plt.Figure | None:
    """Scatter of first-layer radii vs estimated slopes (dimensions)."""

    radii, dims = [], []
    for res in results:
        if not res or not res.get("dimensions") or not res.get("strat_radii"):
            continue
        dim_val = res["dimensions"][0]
        radius_val = res["strat_radii"][0]
        if not np.isfinite(dim_val) or radius_val <= 0:
            continue
        radii.append(radius_val)
        dims.append(dim_val)
    if not radii:
        return None
    radii_np = np.array(radii)
    dims_np = np.array(dims)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(np.log10(radii_np), dims_np, s=18, alpha=0.7)
    ax.set_xlabel("log10 radius")
    ax.set_ylabel("estimated slope (dimension)")
    ax.set_title("Slope vs radius per token")
    fig.tight_layout()
    return fig


def select_irregular_images(
    images: torch.Tensor,
    labels: torch.Tensor,
    fiber_results: List[Dict[str, List[float]]],
    dataset: str,
    bboxes: torch.Tensor,
    neighborhood_dims: Optional[List[float]] = None,
    image_ids: Optional[torch.Tensor] = None,
    class_names: Optional[List[str]] = None,
    image_mean_dims: Optional[Dict[int, float]] = None,
    pred_labels: Optional[torch.Tensor] = None,
    top_k: int = 12,
) -> List[Dict[str, Any]]:
    """Select the top-k irregular tokens and assemble visualization metadata."""
    irregs = []
    for idx, res in enumerate(fiber_results):
        if not res or not res["pvalues"]:
            continue
        pval = res["pvalues"][0]
        irregularity = -np.log10(pval + 1e-12)
        dim = res["dimensions"][0] if res["dimensions"] else float("nan")
        neigh_dim = (
            neighborhood_dims[idx] if neighborhood_dims is not None and idx < len(neighborhood_dims) else float("nan")
        )
        img_id = int(image_ids[idx].item()) if image_ids is not None and idx < len(image_ids) else idx
        pred_lbl = int(pred_labels[idx].item()) if pred_labels is not None and idx < len(pred_labels) else -1
        irregs.append((irregularity, dim, idx, neigh_dim, img_id, pred_lbl))
    irregs.sort(reverse=True, key=lambda x: x[0])
    picks = irregs[:top_k]
    if not picks:
        return []
    imgs = denormalize_images(images, dataset).cpu()
    outputs = []
    for irregularity, dim, idx, neigh_dim, img_id, pred_lbl in picks:
        img = imgs[idx]
        lbl = labels[idx].item() if labels.numel() > idx else -1
        bbox = bboxes[idx]
        cls_name = None
        if class_names and 0 <= lbl < len(class_names):
            cls_name = class_names[lbl]
        pred_name = None
        if class_names and 0 <= pred_lbl < len(class_names):
            pred_name = class_names[pred_lbl]
        mean_dim_img = (
            float(image_mean_dims.get(img_id, float("nan"))) if image_mean_dims is not None else float("nan")
        )
        outputs.append(
            {
                "img": img,
                "irregularity": irregularity,
                "dim": dim,
                "neigh_dim": neigh_dim,
                "label": lbl,
                "label_name": cls_name,
                "pred_label": pred_lbl,
                "pred_label_name": pred_name,
                "token_id": idx,
                "image_id": img_id,
                "bbox": bbox,
                "image_mean_dim": mean_dim_img,
            }
        )
    return outputs


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="TinyViT + fiber bundle test")
    # Data / model
    p.add_argument("--dataset", type=str, default="CIFAR10")
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument(
        "--timm-model",
        type=str,
        default=None,
        help="Use a timm ViT model name instead of TinyViT (e.g., vit_tiny_patch16_224)",
    )
    p.add_argument(
        "--timm-pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load pretrained weights for timm model (default: True)",
    )
    p.add_argument(
        "--neighborhood-size",
        type=int,
        default=None,
        help="Square neighborhood side length (pixels) used to average dimensions; must exceed patch size",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--batch-size-test", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--subset-train", type=int, default=5000)
    p.add_argument("--subset-test", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument(
        "--embed-full-val",
        action="store_true",
        help="Use the entire validation set for embeddings/visualizations instead of capping at max_tokens",
    )
    # Train
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--embed-interval", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=2024)
    # Stratified estimator
    p.add_argument("--vol-min", type=int, default=8)
    p.add_argument("--vol-max", type=int, default=64)
    p.add_argument("--ws", type=int, default=8)
    p.add_argument("--alpha", type=float, default=5e-3)
    p.add_argument("--nstrat", type=int, default=3)
    # Output
    p.add_argument("--outdir", type=str, default=None)
    # wandb
    p.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p.add_argument("--project", type=str, default="tinyvit_fiber", help="wandb project")
    p.add_argument("--run-name", type=str, default=None, help="wandb run name")
    return p.parse_args()


def main():
    args = parse_args()
    if args.neighborhood_size is None:
        args.neighborhood_size = max(args.patch_size * 2, args.patch_size + 1)
    elif args.neighborhood_size <= args.patch_size:
        args.neighborhood_size = args.patch_size + 1
        print(
            f"[warn] neighborhood_size must exceed patch_size; overriding to {args.neighborhood_size}",
            flush=True,
        )
    accelerator = Accelerator()
    device = accelerator.device
    is_main = accelerator.is_main_process
    world_size = accelerator.num_processes

    base_dir = (
        Path(args.outdir)
        if args.outdir is not None
        else Path(f"tinyvit_runs/fiber_bundle_{args.dataset.lower()}")
    )
    embed_dir = base_dir / "embeddings"
    analysis_dir = base_dir / "fiber_analysis"
    if is_main:
        base_dir.mkdir(parents=True, exist_ok=True)
        embed_dir.mkdir(parents=True, exist_ok=True)
        analysis_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    seed_everything(args.seed + accelerator.process_index)
    if is_main and args.wandb:
        if wandb is None:
            print("[wandb] init failed: wandb is not installed")
            args.wandb = False
        else:
            try:
                wandb.init(
                    project=args.project,
                    name=args.run_name,
                    config=vars(args),
                    mode=os.environ.get("WANDB_MODE", "online"),
                )
            except Exception as e:
                print(f"[wandb] init failed: {e}")
                args.wandb = False

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # Data (global batch sizes divided per process)
    batch_size_train = max(1, args.batch_size // world_size)
    batch_size_test = max(1, args.batch_size_test // world_size)

    desired_img_size = args.img_size
    if args.timm_model and desired_img_size is None:
        desired_img_size = 224

    (
        train_loader,
        test_loader,
        num_classes,
        in_chans,
        img_size,
        task,
    ) = make_subset_loaders(
        dataset_name=args.dataset,
        root=args.root,
        img_size=desired_img_size,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        num_workers=args.num_workers,
        subset_train=args.subset_train,
        subset_test=args.subset_test,
        device=device,
    )
    # Resolve class names (fallback to known semantic lists when available)

    def _resolve_classes(ds, dataset_name: str):
        name = dataset_name.upper()
        known = {
            "CIFAR10": [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ],
            # FFHQ is faces only; provide a semantic placeholder
            "FFHQ": ["face"],
        }
        if name in known:
            return known[name]
        if hasattr(ds, "classes"):
            return ds.classes
        if hasattr(ds, "dataset"):
            return _resolve_classes(ds.dataset, name)
        return None

    class_names = _resolve_classes(test_loader.dataset, args.dataset)

    # Model + opt
    if args.timm_model:
        model = TimmViTWrapper(args.timm_model, num_classes, pretrained=args.timm_pretrained)
        timm_patch = resolve_patch_size(model)
        if timm_patch is not None:
            args.patch_size = timm_patch
            if is_main:
                print(f"[info] Using timm model {args.timm_model} with patch size {args.patch_size}")
    else:
        model = TinyViT(
            img_size=img_size,
            patch_size=args.patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=192,
            depth=8,
            num_heads=3,
            mlp_ratio=2.0,
            dropout_rate=0.1,
        )

    effective_lr = args.lr * world_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=args.wd)

    def lr_lambda(e: int) -> float:
        if e < args.warmup_epochs:
            return (e + 1) / max(1, args.warmup_epochs)
        t = (e - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )

    train_history: List[Dict[str, float]] = []
    fiber_history: List[Dict[str, float]] = []
    final_dims = None
    final_coords_3d = None
    final_tsne_3d = None

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch_accelerate(
            model,
            train_loader,
            optimizer,
            accelerator,
            task_type=task,
            grad_clip=args.grad_clip,
            label_smoothing=0.0,
            epoch=epoch,
            show_progress=True,
        )
        eval_loss, eval_acc = evaluate_accelerate(
            model,
            test_loader,
            accelerator,
            task_type=task,
            label_smoothing=0.0,
            show_progress=True,
        )
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        log_row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
        }
        if is_main:
            train_history.append(log_row)
            print(
                f"[Epoch {epoch:03d}] lr {lr_now:.2e} | "
                f"train {train_loss:.4f}/{train_acc:.4f} | "
                f"val {eval_loss:.4f}/{eval_acc:.4f}"
            )
            if args.wandb and wandb is not None:
                try:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "lr": lr_now,
                            "train/loss": train_loss,
                            "train/acc": train_acc,
                            "val/loss": eval_loss,
                            "val/acc": eval_acc,
                        }
                    )
                except Exception as e:
                    print(f"[wandb] log failed: {e}")

        if (epoch + 1) % args.embed_interval == 0 or epoch == 0 or epoch == args.epochs - 1:
            accelerator.wait_for_everyone()
            local_max = None if args.embed_full_val else max(1, math.ceil(args.max_tokens / world_size))
            unwrapped = accelerator.unwrap_model(model)
<<<<<<< Updated upstream
            patch_size_value = unwrapped.patch_embed.patch_size
=======
            patch_sz_eff = resolve_patch_size(unwrapped) or args.patch_size
>>>>>>> Stashed changes
            embeddings, labels, images, bboxes, img_ids, pred_labels = collect_patch_tokens(
                unwrapped,
                test_loader,
                accelerator.device,
<<<<<<< Updated upstream
                patch_size=patch_size_value,
=======
                patch_size=patch_sz_eff,
>>>>>>> Stashed changes
                max_tokens=local_max,
                show_progress=is_main,
            )
            all_embeddings = accelerator.gather_for_metrics(embeddings.to(accelerator.device))
            all_labels = accelerator.gather_for_metrics(labels.to(accelerator.device))
            all_images = accelerator.gather_for_metrics(images.to(accelerator.device))
            all_bboxes = accelerator.gather_for_metrics(bboxes.to(accelerator.device))
            all_img_ids = accelerator.gather_for_metrics(img_ids.to(accelerator.device))
            all_pred_labels = accelerator.gather_for_metrics(pred_labels.to(accelerator.device))

            if is_main and all_embeddings is not None:
                if not args.embed_full_val:
                    all_embeddings = all_embeddings[: args.max_tokens]
                    all_labels = all_labels[: args.max_tokens]
                    all_images = all_images[: args.max_tokens]
                    all_bboxes = all_bboxes[: args.max_tokens]
                    all_img_ids = all_img_ids[: args.max_tokens]
                    all_pred_labels = all_pred_labels[: args.max_tokens]
                all_embeddings = all_embeddings.cpu()
                all_labels = all_labels.cpu()
                all_images = all_images.cpu()
                all_bboxes = all_bboxes.cpu()
                all_img_ids = all_img_ids.cpu()
                all_pred_labels = all_pred_labels.cpu()
                torch.save(
                    {
                        "embeddings": all_embeddings,
                        "labels": all_labels,
                        "images": all_images,
                        "bboxes": all_bboxes,
                        "image_ids": all_img_ids,
                        "pred_labels": all_pred_labels,
                    },
                    embed_dir / f"epoch_{epoch:03d}.pt",
                )

                fiber_results = run_fiber_bundle_test(
                    all_embeddings,
                    vol_min=args.vol_min,
                    vol_max=args.vol_max,
                    ws=args.ws,
                    alpha=args.alpha,
                    nstrat=args.nstrat,
                )
                neighborhood_dims = compute_neighborhood_dimensions(
                    fiber_results, all_bboxes, args.neighborhood_size
                )
                # Build per-image mean dimension
                mean_dim_by_image = {}
                count_by_image = {}
                for dim_val, img_id_val in zip(
                    [res["dimensions"][0] if res["dimensions"] else float("nan") for res in fiber_results],
                    all_img_ids,
                ):
                    img_id_int = int(img_id_val.item())
                    if not math.isfinite(dim_val):
                        continue
                    mean_dim_by_image[img_id_int] = mean_dim_by_image.get(img_id_int, 0.0) + dim_val
                    count_by_image[img_id_int] = count_by_image.get(img_id_int, 0) + 1
                for k in list(mean_dim_by_image.keys()):
                    mean_dim_by_image[k] /= max(1, count_by_image[k])

                fiber_summary = summarize_stratifications(fiber_results, alpha=args.alpha)
                class_dim_means, class_dim_counts = compute_class_dim_means(
                    fiber_results, all_labels, num_classes
                )
                fiber_summary["class_dim_means"] = class_dim_means
                fiber_summary["class_dim_counts"] = class_dim_counts
                finite_neigh_dims = [d for d in neighborhood_dims if math.isfinite(d)]
                fiber_summary["mean_neighborhood_dim"] = (
                    float(np.mean(finite_neigh_dims)) if finite_neigh_dims else float("nan")
                )
                fiber_summary["median_neighborhood_dim"] = (
                    float(np.median(finite_neigh_dims)) if finite_neigh_dims else float("nan")
                )
                fiber_summary["neighborhood_size"] = args.neighborhood_size
                fiber_summary["epoch"] = epoch
                fiber_history.append(fiber_summary)

                fig_bar = None
                fig_bar_path = None
                if any(math.isfinite(m) for m in class_dim_means):
                    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
                    xs = np.arange(num_classes)
                    bars = ax_bar.bar(xs, class_dim_means, color="steelblue")
                    if class_names and len(class_names) == num_classes:
                        ax_bar.set_xticks(xs)
                        ax_bar.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
                    else:
                        ax_bar.set_xticks(xs)
                        ax_bar.set_xticklabels([str(i) for i in range(num_classes)], fontsize=8)
                    ax_bar.set_ylabel("first-stratum dim (mean)")
                    ax_bar.set_title("Per-class average dimension")
                    ax_bar.bar_label(bars, labels=[f"{c}" for c in class_dim_counts], fontsize=7)
                    fig_bar.tight_layout()
                    fig_bar_path = base_dir / f"class_dims_epoch_{epoch:03d}.png"

                with open(base_dir / f"fiber_epoch_{epoch:03d}.json", "w") as fp:
                    json.dump(to_serializable(fiber_results), fp, indent=2)

                final_coords_3d = project_embeddings_3d(all_embeddings)
                final_tsne_3d = tsne_embeddings_3d(all_embeddings)
                final_dims = np.array(
                    [res["dimensions"][0] if res["dimensions"] else np.nan for res in fiber_results]
                )

                analysis_paths: Dict[str, Path] = {}
                preview_data: Dict[str, List[Dict[str, Any]]] = {}

                dim_hist_fig = plot_dimension_histogram(final_dims)
                if dim_hist_fig is not None:
                    hist_path = analysis_dir / f"epoch_{epoch:03d}_dim_hist.png"
                    dim_hist_fig.savefig(hist_path, dpi=200)
                    plt.close(dim_hist_fig)
                    analysis_paths["fiber/dim_histogram"] = hist_path

                patch_count_fig = plot_patch_count_curve(all_img_ids)
                if patch_count_fig is not None:
                    patch_path = analysis_dir / f"epoch_{epoch:03d}_patch_count.png"
                    patch_count_fig.savefig(patch_path, dpi=200)
                    plt.close(patch_count_fig)
                    analysis_paths["fiber/patch_count_curve"] = patch_path

                slope_fig = plot_dimension_radius_scatter(fiber_results)
                if slope_fig is not None:
                    slope_path = analysis_dir / f"epoch_{epoch:03d}_slope_radius.png"
                    slope_fig.savefig(slope_path, dpi=200)
                    plt.close(slope_fig)
                    analysis_paths["fiber/dim_radius_scatter"] = slope_path

                extreme_visuals = prepare_dimension_extreme_visuals(
                    final_dims,
                    all_images,
                    all_bboxes,
                    all_labels,
                    all_pred_labels,
                    dataset=args.dataset,
                    class_names=class_names,
                    top_k=4,
                )
                for group_name in ("low", "high"):
                    group = extreme_visuals.get(group_name)
                    if not group:
                        continue
                    if group.get("figure") is not None:
                        panel_path = analysis_dir / f"epoch_{epoch:03d}_{group_name}_dim_panel.png"
                        group["figure"].savefig(panel_path, dpi=200)
                        plt.close(group["figure"])
                        key = f"embeddings/{group_name}_dim_panel"
                        analysis_paths[key] = panel_path
                    if group.get("previews"):
                        preview_data[group_name] = group["previews"]

                if args.wandb and wandb is not None:
                    try:
                        fig3d = make_embedding_figure_3d(final_coords_3d, final_dims)
                        log_dict = {
                            "epoch": epoch,
                            "fiber/mean_dim": fiber_summary["mean_dim"],
                            "fiber/median_dim": fiber_summary["median_dim"],
                            "fiber/min_pvalue": fiber_summary["min_pvalue"],
                            "fiber/max_pvalue": fiber_summary["max_pvalue"],
                            "fiber/mean_irregularity": fiber_summary["mean_irregularity"],
                            "fiber/max_irregularity": fiber_summary["max_irregularity"],
                            "fiber/irregular_ratio": fiber_summary["irregular_ratio"],
                            "fiber/mean_neighborhood_dim": fiber_summary["mean_neighborhood_dim"],
                            "fiber/median_neighborhood_dim": fiber_summary["median_neighborhood_dim"],
                            "embeddings/pca_3d": wandb.Image(fig3d, caption=f"Epoch {epoch}"),
                        }
                        if fig_bar is not None:
                            log_dict["fiber/class_dim_bar"] = wandb.Image(fig_bar, caption=f"Epoch {epoch}")
                        plt.close(fig3d)

                        if final_tsne_3d is not None:
                            fig_tsne = make_embedding_figure_tsne(final_tsne_3d, final_dims)
                            log_dict["embeddings/tsne_3d"] = wandb.Image(
                                fig_tsne, caption=f"Epoch {epoch}"
                            )
                            plt.close(fig_tsne)

                        for key, path in analysis_paths.items():
                            log_dict[key] = wandb.Image(str(path))

                        if preview_data.get("low"):
                            log_dict["embeddings/low_dim_patches"] = [
                                wandb.Image(item["image"], caption=item["caption"])
                                for item in preview_data["low"]
                            ]
                        if preview_data.get("high"):
                            log_dict["embeddings/high_dim_patches"] = [
                                wandb.Image(item["image"], caption=item["caption"])
                                for item in preview_data["high"]
                            ]

                        wandb.log(log_dict)

                        neighborhood_dims_vals = [
                            res["dimensions"][0] if res["dimensions"] else float("nan")
                            for res in fiber_results
                        ]
                        irregular = select_irregular_images(
                            all_images,
                            all_labels,
                            fiber_results,
                            args.dataset,
                            all_bboxes,
<<<<<<< Updated upstream
                            neighborhood_dims=neighborhood_dims_vals,
=======
                            neighborhood_dims=neighborhood_dims,
>>>>>>> Stashed changes
                            image_ids=all_img_ids,
                            class_names=class_names,
                            pred_labels=all_pred_labels,
                            image_mean_dims=mean_dim_by_image,
                            top_k=12,
                        )
                        if irregular:
<<<<<<< Updated upstream
=======
                            neigh_max = max(
                                [item["neigh_dim"] for item in irregular if math.isfinite(item["neigh_dim"])],
                                default=1.0,
                            )
                            if neigh_max <= 0:
                                neigh_max = 1.0
                            # Add patch-level predicted class for each irregular region
>>>>>>> Stashed changes
                            unwrapped_model = accelerator.unwrap_model(model)
                            for item in irregular:
                                pred_lbl, pred_name = predict_patch_class(
                                    item["img"],
                                    item["bbox"],
                                    unwrapped_model,
                                    accelerator.device,
                                    img_size,
                                    args.dataset,
                                    class_names=class_names,
                                )
                                item["patch_pred_label"] = pred_lbl
                                item["patch_pred_label_name"] = pred_name

<<<<<<< Updated upstream
                            heatmap_images = []
                            for item in irregular:
                                patch_pred_label = item.get(
                                    "patch_pred_label_name",
                                    item.get("patch_pred_label", ""),
                                )
                                caption = (
                                    f"token {item['token_id']}, "
                                    f"class {item.get('label_name', item['label'])}, "
                                    f"pred {item.get('pred_label_name', item['pred_label'])}, "
                                    f"patch_pred {patch_pred_label}, "
                                    f"dim {item['dim']:.2f}, "
                                    f"img_mean_dim {item['image_mean_dim']:.2f}, "
                                    f"irr {item['irregularity']:.2f}"
                                )
                                heatmap_images.append(
                                    wandb.Image(
                                        add_heatmap_patch(item["img"], item["bbox"], item["irregularity"]),
                                        caption=caption,
                                    )
                                )

                            wandb.log({"embeddings/irregular_samples": heatmap_images})
=======
                            wandb.log(
                                {
                                    "embeddings/irregular_samples": [
                                        wandb.Image(
                                            add_heatmap_patch(
                                                item["img"],
                                                item["bbox"],
                                                item["irregularity"],
                                                neigh_value=item["neigh_dim"],
                                                neigh_max=neigh_max,
                                                neighborhood_size=args.neighborhood_size,
                                            ),
                                            caption=(
                                                f"token {item['token_id']}, "
                                                f"class {item.get('label_name', item['label'])}, "
                                                f"pred {item.get('pred_label_name', item['pred_label'])}, "
                                                f"patch_pred {item.get('patch_pred_label_name', item.get('patch_pred_label', ''))}, "
                                                f"dim {item['dim']:.2f}, "
                                                f"neigh_dim {item['neigh_dim']:.2f}, "
                                                f"img_mean_dim {item['image_mean_dim']:.2f}, "
                                                f"irr {item['irregularity']:.2f}"
                                            ),
                                        )
                                        for item in irregular
                                    ]
                                }
                        )
>>>>>>> Stashed changes
                    except Exception as e:
                        print(f"[wandb] logging failed: {e}")

                if fig_bar is not None and fig_bar_path is not None:
                    fig_bar.savefig(fig_bar_path, dpi=200)
                    plt.close(fig_bar)

    accelerator.wait_for_everyone()
    if is_main:
        with open(base_dir / "train_history.json", "w") as fp:
            json.dump(train_history, fp, indent=2)
        with open(base_dir / "fiber_history.json", "w") as fp:
            json.dump(fiber_history, fp, indent=2)

        if final_coords_3d is not None and final_dims is not None:
            plot_progress(
                train_history,
                fiber_history,
                final_coords_3d,
                final_dims,
                base_dir / "fiber_bundle_summary.png",
            )
            print(f"Saved summary plot → {base_dir/'fiber_bundle_summary.png'}")

        if args.wandb and wandb is not None:
            try:
                wandb.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()
