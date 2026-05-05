"""Matplotlib-dependent plotting and image utilities for fiber analysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from utils import denormalize_images

try:
    import scipy.stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    matplotlib = None
    plt = None
    HAS_MATPLOTLIB = False

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    TSNE = None
    HAS_TSNE = False


def _require_matplotlib():
    if plt is None:
        raise ImportError("matplotlib is required for fiber-bundle plotting outputs")
    return plt


def _corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank_positions(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    ranks = np.empty(order.shape[0], dtype=np.float64)
    ranks[order] = np.arange(order.shape[0], dtype=np.float64)
    return ranks


def matplotlib_supports_3d() -> bool:
    if plt is None:
        return False
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def project_embeddings_pca_3d(embeddings: torch.Tensor) -> np.ndarray:
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=3)
    return (centered @ v[:, :3]).cpu().numpy()


def project_embeddings_pca_2d(
    embeddings: torch.Tensor,
    *,
    mean: torch.Tensor | None = None,
    basis: torch.Tensor | None = None,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    emb = embeddings.detach().float().cpu()
    if emb.ndim != 2:
        raise ValueError("embeddings must be rank-2 for 2D projection")
    feat_dim = emb.shape[1]
    if mean is None:
        mean = emb.mean(dim=0, keepdim=True) if emb.shape[0] else torch.zeros((1, feat_dim), dtype=emb.dtype)
    centered = emb - mean
    if basis is None:
        rank = min(2, centered.shape[0], centered.shape[1])
        if rank > 0:
            _, _, v = torch.pca_lowrank(centered, q=rank)
            basis = v[:, :rank]
        else:
            basis = torch.zeros((feat_dim, 0), dtype=emb.dtype)
    coords = centered @ basis if basis.numel() else torch.zeros((emb.shape[0], 0), dtype=emb.dtype)
    if coords.shape[1] < 2:
        coords = torch.cat([coords, torch.zeros((coords.shape[0], 2 - coords.shape[1]), dtype=emb.dtype)], dim=1)
    return coords[:, :2].numpy(), mean, basis


def project_embeddings_tsne_3d(
    embeddings: torch.Tensor,
    perplexity: float = 30.0,
    seed: int = 42,
    max_points: int = 2048,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not HAS_TSNE:
        return None
    emb_np = embeddings.cpu().numpy()
    n = emb_np.shape[0]
    if n > max_points:
        idx = np.random.default_rng(seed).choice(n, size=max_points, replace=False)
        emb_np = emb_np[idx]
    else:
        idx = np.arange(n)
    tsne = TSNE(
        n_components=3,
        perplexity=min(perplexity, max(5, len(emb_np) - 1)),
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(emb_np), idx


# ---------------------------------------------------------------------------
# Patch image utilities
# ---------------------------------------------------------------------------

def extract_patch_image(
    img_tensor: torch.Tensor, bbox: torch.Tensor, upscale: int = 128
) -> Image.Image:
    np_img = img_tensor.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    h, w = np_img.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox.tolist()]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    patch = np_img[y0:y1, x0:x1, :] if x1 > x0 and y1 > y0 else np_img
    pil = Image.fromarray((patch * 255).astype("uint8"))
    if upscale and (pil.width < upscale or pil.height < upscale):
        pil = pil.resize(
            (upscale, upscale),
            resample=getattr(Image, "Resampling", Image).BILINEAR,
        )
    return pil


def add_heatmap_patch(
    img_tensor: torch.Tensor,
    bbox: torch.Tensor,
    value: float,
    max_value: float = 5.0,
    neigh_value: float | None = None,
    neigh_max: float = 10.0,
    neighborhood_size: float | None = None,
) -> Image.Image:
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
            np_img[nby0:nby1, nbx0:nbx1, :] = (
                (1 - blue_alpha) * np_img[nby0:nby1, nbx0:nbx1, :]
                + blue_alpha * np.array([0.2, 0.4, 1.0])
            )

    norm = max(0, min(1, value / max_value))
    color, alpha = np.array([1.0, norm, 0.0]), 0.25 + 0.45 * norm
    np_img[y0:y1, x0:x1, :] = (1 - alpha) * np_img[y0:y1, x0:x1, :] + alpha * color
    return Image.fromarray((np_img * 255).astype("uint8"))


def _make_patch_grid(
    patches: List[Image.Image],
    *,
    cols: int = 8,
    pad: int = 2,
    bg: tuple = (10, 10, 10),
) -> Image.Image:
    if not patches:
        return Image.new("RGB", (64, 64), bg)
    w, h = patches[0].size
    rows = math.ceil(len(patches) / cols)
    grid = Image.new(
        "RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), bg
    )
    for i, p in enumerate(patches):
        grid.paste(p, (pad + (i % cols) * (w + pad), pad + (i // cols) * (h + pad)))
    return grid


def _tensor01_to_pil(img01: torch.Tensor) -> Image.Image:
    np_img = (img01.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(np_img)


def _pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    np_img = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(np_img).permute(2, 0, 1)


def _draw_patch_box(
    img: Image.Image, bbox: np.ndarray, color: tuple = (255, 0, 0), width: int = 3
) -> Image.Image:
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
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(out.width, x1), min(out.height, y1)
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


def _format_label_text(
    label: torch.Tensor, class_names: List[str] | None, max_items: int = 3
) -> str:
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
        names = [
            class_names[i] if class_names and 0 <= i < len(class_names) else str(i)
            for i in pos[:max_items]
        ]
        extra = f" +{len(pos) - max_items}" if len(pos) > max_items else ""
        return f"labels {', '.join(names)}{extra}"
    return "label n/a"


def _format_top_label(
    top_label: int, top_share: float, class_names: List[str] | None
) -> str:
    if top_label < 0:
        return "top label n/a"
    name = class_names[top_label] if class_names and 0 <= top_label < len(class_names) else str(top_label)
    return f"top label {name} ({top_share:.0%})"


def _label_name(idx: int, class_names: List[str] | None) -> str:
    if class_names and 0 <= idx < len(class_names):
        return class_names[idx]
    return str(idx)


# ---------------------------------------------------------------------------
# Matplotlib plotting
# ---------------------------------------------------------------------------

def _add_embedding_scatter_subplot(fig, subplot_spec, coords: np.ndarray, colors: np.ndarray, *, title: str):
    plt_mod = _require_matplotlib()
    coords_np = np.asarray(coords, dtype=np.float64)
    if coords_np.ndim != 2 or coords_np.shape[0] == 0:
        coords_np = np.zeros((0, 3), dtype=np.float64)
    if coords_np.shape[1] < 3:
        coords_np = np.pad(coords_np, ((0, 0), (0, 3 - coords_np.shape[1])), mode="constant")

    if matplotlib_supports_3d():
        ax = fig.add_subplot(subplot_spec, projection="3d")
        sc = ax.scatter(coords_np[:, 0], coords_np[:, 1], coords_np[:, 2], c=colors, cmap="viridis", s=12, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
        ax.set_zlabel("component 3")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        return ax, sc

    ax = fig.add_subplot(subplot_spec)
    sc = ax.scatter(coords_np[:, 0], coords_np[:, 1], c=colors, cmap="viridis", s=12, alpha=0.85)
    ax.set_title(f"{title} (2D fallback)")
    ax.set_xlabel("component 1"); ax.set_ylabel("component 2")
    ax.grid(alpha=0.2, linewidth=0.5)
    return ax, sc


def save_training_summary_plot(
    train_history: List[Dict],
    fiber_history: List[Dict],
    final_coords_3d: np.ndarray,
    final_colors: np.ndarray,
    out_path: Path,
) -> None:
    plt_mod = _require_matplotlib()
    fig = plt_mod.figure(figsize=(18, 5))
    ax1, ax2 = fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2)
    epochs = [m["epoch"] for m in train_history]
    ax1.plot(epochs, [m["train_acc"] for m in train_history], label="train acc")
    ax1.plot(epochs, [m["eval_acc"] for m in train_history], label="val acc")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("accuracy"); ax1.set_title("Training"); ax1.legend()
    fiber_epochs = [m["epoch"] for m in fiber_history]
    ax2.plot(fiber_epochs, [m["mean_dim"] for m in fiber_history], "o-", label="mean dim")
    ax2.plot(fiber_epochs, [m.get("mean_neighborhood_dim", np.nan) for m in fiber_history], "s-", label="mean neigh dim")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("dimension"); ax2.set_title("Fiber Summary")
    ax2.legend(loc="upper left")
    ax2b = ax2.twinx()
    ax2b.plot(fiber_epochs, [m.get("hypothesis_score", np.nan) for m in fiber_history], "^-", color="tab:green", label="hypothesis")
    ax2b.plot(fiber_epochs, [m.get("irregular_ratio", np.nan) for m in fiber_history], "x--", color="tab:red", label="irregular")
    ax2b.set_ylabel("score / ratio")
    ax2b.legend(loc="lower right")
    ax3, sc = _add_embedding_scatter_subplot(fig, 133, final_coords_3d, final_colors, title="Embeddings (PCA)")
    fig.colorbar(sc, ax=ax3, shrink=0.6, label="dim")
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt_mod.close(fig)


def build_embedding_scatter_figure(
    coords3d: np.ndarray, dims: np.ndarray, title: str = "Embeddings (PCA 3D)"
):
    plt_mod = _require_matplotlib()
    fig = plt_mod.figure(figsize=(6, 5))
    ax, sc = _add_embedding_scatter_subplot(fig, 111, coords3d, dims, title=title)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="estimated local dimension")
    fig.text(
        0.02,
        0.02,
        "Color shows the first estimated local volume dimension for each token.",
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def build_tsne_embedding_figure(coords3d: np.ndarray, dims: np.ndarray):
    return build_embedding_scatter_figure(coords3d, dims, "Embeddings (t-SNE 3D)")


def save_polysemy_irregularity_plot(
    *,
    entropies: np.ndarray,
    fiber_results: List[Dict[str, Any]],
    out_dir: Path,
    prefix: str,
    alpha: float = 1e-2,
) -> tuple[Path | None, Dict[str, float]]:
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
    ent, irr, rejected = ent[mask], irr[mask], rejected[mask]

    pearson_r, pearson_p = (float("nan"), float("nan"))
    spearman_r, spearman_p = (float("nan"), float("nan"))
    if ent.size > 2:
        if scipy_stats is not None:
            pearson_r, pearson_p = scipy_stats.pearsonr(ent, irr)
            spearman_r, spearman_p = scipy_stats.spearmanr(ent, irr)
        else:
            pearson_r = _corrcoef_safe(ent, irr)
            spearman_r = _corrcoef_safe(_rank_positions(ent), _rank_positions(irr))

    ent_rej, ent_ok = ent[rejected], ent[~rejected]
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
    ax0.set_title(f"Do semantic mixtures align with geometric failures? (r={pearson_r:.2f}, rho={spearman_r:.2f})")
    ax0.legend(fontsize=8, frameon=False)
    box_data = [ent_ok, ent_rej] if ent_rej.size else [ent_ok]
    ax1.boxplot(box_data, labels=["non-reject", "reject"] if ent_rej.size else ["non-reject"], showfliers=False)
    ax1.set_ylabel("Polysemy entropy")
    ax1.set_title("Entropy split by fiber-test outcome")
    out_path = out_dir / f"{prefix}_polysemy_entropy_vs_irregularity.png"
    fig.suptitle("Polysemy vs Fiber Irregularity", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(out_path, dpi=200); plt_mod.close(fig)
    return out_path, stats


def save_polysemy_entropy_scatter_plot(
    polysemy_result: Dict[str, Any],
    *,
    out_dir: Path,
    prefix: str,
    annotate_top: int = 6,
) -> Path | None:
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
    ax.set_xlabel("Top-label share"); ax.set_ylabel("Label entropy")
    ax.set_title("Polysemy Anchors: Higher Entropy Means More Label Mixing")
    ax.set_xlim(0.0, 1.0)
    fig.colorbar(sc, ax=ax, shrink=0.8, label="unique labels in neighborhood")
    top_idx = np.argsort(-ent)[: max(1, annotate_top)]
    for i in top_idx:
        ax.text(share[i], ent[i], str(ids[i]), fontsize=7)
    out_path = out_dir / f"{prefix}_polysemy_entropy_scatter.png"
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt_mod.close(fig)
    polysemy_result.setdefault("paths", {})["polysemy/entropy_scatter"] = out_path
    return out_path


# ---------------------------------------------------------------------------
# Irregularity scoring (used by polysemy + visualization)
# ---------------------------------------------------------------------------

def _compute_irregularity_scores(
    fiber_results: List[Dict[str, Any]], alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def select_singular_tokens(
    *, fiber_results: List[Dict[str, Any]], alpha: float, top_k: int
) -> List[int]:
    min_p, irr, rejected = _compute_irregularity_scores(fiber_results, alpha)
    if irr.size == 0 or not np.any(rejected):
        return []
    idxs = np.where(rejected)[0]
    order = np.argsort(-irr[idxs])
    picks = idxs[order][: max(1, top_k)]
    return [int(i) for i in picks if math.isfinite(min_p[int(i)])]


def select_irregular_tokens(
    images: torch.Tensor,
    image_ids: torch.Tensor,
    labels: torch.Tensor,
    fiber_results: List[Dict],
    dataset: str,
    bboxes: torch.Tensor,
    neighborhood_dims: List[float] | None = None,
    class_names: List[str] | None = None,
    image_mean_dims: Dict[int, float] | None = None,
    pred_labels: torch.Tensor | None = None,
    top_k: int = 12,
) -> List[Dict[str, Any]]:
    """Select most irregular tokens for visualization.

    ``images`` is the **unique** image buffer; ``image_ids`` maps each token
    to its source image index.
    """
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
            int(pred_labels[idx]) if pred_labels is not None and idx < len(pred_labels) else -1,
        ))
    irregs.sort(reverse=True, key=lambda x: x[0])
    picks = irregs[:top_k]
    if not picks:
        return []
    imgs = denormalize_images(images, dataset).cpu()
    outputs = []
    for irr_val, dim, idx, neigh_dim, img_id, pred_lbl in picks:
        lbl = labels[idx].cpu() if idx < len(labels) else None
        if isinstance(lbl, torch.Tensor) and lbl.dim() == 0:
            lbl_val = int(lbl.item())
            cls_name = class_names[lbl_val] if class_names and 0 <= lbl_val < len(class_names) else str(lbl_val)
        elif isinstance(lbl, torch.Tensor) and lbl.dim() == 1:
            pos = (lbl > 0).nonzero().view(-1).tolist()
            cls_name = ", ".join(
                [class_names[i] if class_names and 0 <= i < len(class_names) else str(i) for i in pos[:6]]
            )
            lbl_val = pos
        else:
            lbl_val, cls_name = -1, None
        pred_name = class_names[pred_lbl] if class_names and 0 <= pred_lbl < len(class_names) else None
        # img_id indexes into unique images buffer
        source_img = imgs[img_id] if img_id < imgs.shape[0] else imgs[0]
        outputs.append({
            "img": source_img,
            "irregularity": irr_val,
            "dim": dim,
            "neigh_dim": neigh_dim,
            "label": lbl_val,
            "label_name": cls_name,
            "pred_label": pred_lbl,
            "pred_label_name": pred_name,
            "token_id": idx,
            "image_id": img_id,
            "bbox": bboxes[idx],
            "image_mean_dim": image_mean_dims.get(img_id, np.nan) if image_mean_dims else np.nan,
        })
    return outputs


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def _maybe_wandb_metric_table(wandb_module, prefix: str, metrics: Dict[str, Any]):
    if wandb_module is None or not hasattr(wandb_module, "Table"):
        return None
    rows = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
            rows.append([f"{prefix}/{key}", f"{float(value):.6g}"])
        elif isinstance(value, str):
            rows.append([f"{prefix}/{key}", value])
    return wandb_module.Table(columns=["metric", "value"], data=rows) if rows else None


def _wandb_image_or_none(*, wandb_module, key: str, build_fn):
    try:
        return build_fn()
    except Exception as exc:
        print(f"[wandb] skipped {key}: {exc}")
        return None


project_embeddings_3d = project_embeddings_pca_3d
project_embeddings_2d = project_embeddings_pca_2d
tsne_embeddings_3d = project_embeddings_tsne_3d
plot_progress = save_training_summary_plot
make_embedding_figure_3d = build_embedding_scatter_figure
make_embedding_figure_tsne = build_tsne_embedding_figure
make_polysemy_irregularity_plot = save_polysemy_irregularity_plot
make_polysemy_entropy_scatter = save_polysemy_entropy_scatter_plot
select_singular_token_indices = select_singular_tokens
select_irregular_images = select_irregular_tokens
