#!/usr/bin/env python3
"""
TinyViT training + fiber bundle test using the stratified_estimator routines.

Runs a 100-epoch CIFAR-10 training (subsampled for speed on CPU), saves CLS
token embeddings every 10 epochs, applies the stratified estimator to detect
stratifications, and produces a visualization. Supports multi-GPU training
via Hugging Face Accelerate (use `accelerate launch`).
"""

import argparse
import io
import os
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.spatial  # noqa: E402
import scipy.stats  # noqa: E402
import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
try:
    from sklearn.manifold import TSNE  # noqa: E402

    HAS_TSNE = True
except ImportError:
    TSNE = None
    HAS_TSNE = False

from tinyvit import TinyViT, build_dataset, get_criterion, multilabel_accuracy


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
    max_tokens: int = 256,
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
    with torch.no_grad():
        for imgs, lbls in loader:
            if collected >= max_tokens:
                break
            imgs = imgs.to(device)
            feats = model.forward_features(imgs)
            logits = model.head(feats[:, 0])
            preds = logits.argmax(dim=-1).cpu()
            patch_tokens = feats[:, 1:, :].cpu()  # remove CLS
            B, P, E = patch_tokens.shape
            grid = int(math.sqrt(P))
            for i in range(B):
                for p in range(P):
                    if collected >= max_tokens:
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
                if collected >= max_tokens:
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
    emb_tensor = torch.stack(embeddings, dim=0)[:max_tokens]
    label_tensor = torch.stack(labels, dim=0)[:max_tokens]
    img_tensor = torch.stack(images, dim=0)[:max_tokens]
    bbox_tensor = torch.stack(bboxes, dim=0)[:max_tokens]
    imgid_tensor = torch.stack(image_ids, dim=0)[:max_tokens]
    pred_tensor = torch.stack(pred_labels, dim=0)[:max_tokens]
    return emb_tensor, label_tensor, img_tensor, bbox_tensor, imgid_tensor, pred_tensor


def geo_estimator(radii, volumes, npts, args):
    """Estimate scaling coefficient, dimension, and Ricci (from kb1dds/stratified_estimator)."""
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


def stratification_test(radii, volumes, ws=10, alpha=1e-3):
    """Sliding-window Welch t-test to spot stratifications (kb1dds/stratified_estimator)."""
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


def estimate_stratifications(dists_sorted, vol_min, vol_max, npts, args, ws=10, alpha=1e-3):
    """Detect stratifications (unchanged logic from kb1dds/stratified_estimator)."""
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


def add_heatmap_patch(img_tensor: torch.Tensor, bbox: torch.Tensor, value: float, max_value: float = 5.0) -> Image.Image:
    """Apply a heatmap tint to the patch region without drawing a solid box."""
    np_img = img_tensor.permute(1, 2, 0).clamp(0, 1).numpy()
    h, w, _ = np_img.shape
    x0, y0, x1, y1 = [int(v) for v in bbox.tolist()]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return Image.fromarray((np_img * 255).astype("uint8"))

    # Normalize irregularity to [0,1] and pick color from red -> yellow
    norm = max(0.0, min(1.0, value / max_value))
    color = np.array([1.0, norm, 0.0], dtype=np.float32)  # RGB in [0,1]
    alpha = 0.25 + 0.45 * norm

    patch = np_img[y0:y1, x0:x1, :]
    np_img[y0:y1, x0:x1, :] = (1 - alpha) * patch + alpha * color
    return Image.fromarray((np_img * 255).astype("uint8"))


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


def make_embedding_figure_3d(coords3d: np.ndarray, dims: np.ndarray, title: str = "CLS embeddings (PCA 3D)") -> plt.Figure:
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


def select_irregular_images(
    images: torch.Tensor,
    labels: torch.Tensor,
    fiber_results: List[Dict[str, List[float]]],
    dataset: str,
    bboxes: torch.Tensor,
    neighborhood_dims: List[float] | None = None,
    image_ids: torch.Tensor | None = None,
    class_names: List[str] | None = None,
    image_mean_dims: dict | None = None,
    pred_labels: torch.Tensor | None = None,
    top_k: int = 12,
):
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
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--batch-size-test", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--subset-train", type=int, default=5000)
    p.add_argument("--subset-test", type=int, default=1000)
    p.add_argument("--max-tokens", type=int, default=256)
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
    if is_main:
        base_dir.mkdir(parents=True, exist_ok=True)
        embed_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    seed_everything(args.seed + accelerator.process_index)
    if is_main and args.wandb:
        try:
            import wandb

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
        img_size=args.img_size,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        num_workers=args.num_workers,
        subset_train=args.subset_train,
        subset_test=args.subset_test,
        device=device,
    )
    # Resolve class names (works for torchvision datasets and Subsets)
    def _resolve_classes(ds):
        if hasattr(ds, "classes"):
            return ds.classes
        if hasattr(ds, "dataset"):
            return _resolve_classes(ds.dataset)
        return None

    class_names = _resolve_classes(test_loader.dataset)

    # Model + opt
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
    final_coords = None
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
            if args.wandb:
                try:
                    import wandb

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
            local_max = max(1, math.ceil(args.max_tokens / world_size))
            unwrapped = accelerator.unwrap_model(model)
            embeddings, labels, images, bboxes, img_ids, pred_labels = collect_patch_tokens(
                unwrapped,
                test_loader,
                accelerator.device,
                patch_size=model.module.patch_embed.patch_size if hasattr(model, "module") else model.patch_embed.patch_size,
                max_tokens=local_max,
            )
            all_embeddings = accelerator.gather_for_metrics(embeddings.to(accelerator.device))
            all_labels = accelerator.gather_for_metrics(labels.to(accelerator.device))
            all_images = accelerator.gather_for_metrics(images.to(accelerator.device))
            all_bboxes = accelerator.gather_for_metrics(bboxes.to(accelerator.device))
            all_img_ids = accelerator.gather_for_metrics(img_ids.to(accelerator.device))
            all_pred_labels = accelerator.gather_for_metrics(pred_labels.to(accelerator.device))

            if is_main and all_embeddings is not None:
                all_embeddings = all_embeddings[: args.max_tokens].cpu()
                all_labels = all_labels[: args.max_tokens].cpu()
                all_images = all_images[: args.max_tokens].cpu()
                all_bboxes = all_bboxes[: args.max_tokens].cpu()
                all_img_ids = all_img_ids[: args.max_tokens].cpu()
                all_pred_labels = all_pred_labels[: args.max_tokens].cpu()
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
                fiber_summary["epoch"] = epoch
                fiber_history.append(fiber_summary)

                with open(base_dir / f"fiber_epoch_{epoch:03d}.json", "w") as fp:
                    json.dump(to_serializable(fiber_results), fp, indent=2)

                final_coords_3d = project_embeddings_3d(all_embeddings)
                final_tsne_3d = tsne_embeddings_3d(all_embeddings)
                final_dims = np.array(
                    [res["dimensions"][0] if res["dimensions"] else np.nan for res in fiber_results]
                )

                if args.wandb:
                    try:
                        import wandb

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
                            "embeddings/pca_3d": wandb.Image(fig3d, caption=f"Epoch {epoch}"),
                        }
                        plt.close(fig3d)

                        if final_tsne_3d is not None:
                            fig_tsne = make_embedding_figure_tsne(final_tsne_3d, final_dims)
                            log_dict["embeddings/tsne_3d"] = wandb.Image(
                                fig_tsne, caption=f"Epoch {epoch}"
                            )
                            plt.close(fig_tsne)

                        wandb.log(log_dict)

                        irregular = select_irregular_images(
                            all_images,
                            all_labels,
                            fiber_results,
                            args.dataset,
                            all_bboxes,
                            neighborhood_dims=[res["dimensions"][0] if res["dimensions"] else float("nan") for res in fiber_results],
                            image_ids=all_img_ids,
                            class_names=class_names,
                            pred_labels=all_pred_labels,
                            image_mean_dims=mean_dim_by_image,
                            top_k=12,
                        )
                        if irregular:
                            wandb.log(
                                {
                                    "embeddings/irregular_samples": [
                                        wandb.Image(
                                            add_heatmap_patch(item["img"], item["bbox"], item["irregularity"]),
                                            caption=(
                                                f"token {item['token_id']}, "
                                                f"class {item.get('label_name', item['label'])}, "
                                                f"pred {item.get('pred_label_name', item['pred_label'])}, "
                                                f"dim {item['dim']:.2f}, "
                                                f"img_mean_dim {item['image_mean_dim']:.2f}, "
                                                f"irr {item['irregularity']:.2f}"
                                            ),
                                        )
                                        for item in irregular
                                    ]
                                }
                            )
                    except Exception as e:
                        print(f"[wandb] logging failed: {e}")

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

        if args.wandb:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()
