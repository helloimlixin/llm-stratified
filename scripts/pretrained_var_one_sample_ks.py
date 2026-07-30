"""KS-style uniformity probe for pretrained VAR next-scale code distributions.

The language-side claim is: near a singular/polysemous point, the next-token
distribution becomes flatter, approaching uniform over plausible continuations.
For pretrained VAR, this script tests the visual analogue using the full
teacher-forced 4096-way next-scale VQ-code distribution at final-scale patch
locations.  In VAR, the autoregressive unit is a scale map, not a raster token;
each logged "singular token" is a predicted fine-scale patch code under the
coarser-scale context.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from fiber.branching_ks import fiber_singularity_scores, ks_2samp  # noqa: E402
from fiber.geometry import analyze_stratification  # noqa: E402
from pretrained_var_generator import (  # noqa: E402
    build_pretrained_var,
    parse_class_labels,
    parse_patch_nums,
    resolve_model_defaults,
    save_grid,
    to_jsonable,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def _denormalize_images(images: torch.Tensor, dataset: str) -> torch.Tensor:
    dataset_name = str(dataset).upper()
    mean_values = CIFAR_MEAN if dataset_name in {"CIFAR10", "CIFAR100", "SVHN"} else IMAGENET_MEAN
    std_values = CIFAR_STD if dataset_name in {"CIFAR10", "CIFAR100", "SVHN"} else IMAGENET_STD
    mean = torch.tensor(mean_values, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(std_values, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def _kolmogorov_pvalue(statistic: np.ndarray, n: int) -> np.ndarray:
    d = np.asarray(statistic, dtype=np.float64)
    n = max(1, int(n))
    lam = (math.sqrt(n) + 0.12 + 0.11 / max(math.sqrt(n), 1e-12)) * d
    terms = np.zeros_like(lam, dtype=np.float64)
    for j in range(1, 101):
        terms += ((-1) ** (j - 1)) * np.exp(-2.0 * (j * j) * lam * lam)
    return np.clip(2.0 * terms, 0.0, 1.0)


def one_sample_ks_uniform_draws(
    probs: np.ndarray,
    *,
    draws: int = 512,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo one-sample KS test for categorical distributions.

    For each row p over V code IDs, draw ``draws`` synthetic next-scale codes from p,
    jitter each token uniformly inside its codebook bin, and test the resulting
    [0, 1] sample against Uniform(0, 1).  Smaller KS statistic means the VAR
    next-scale code distribution is closer to uniform.
    """
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("probs must have shape (n_contexts, vocab_size)")
    if arr.shape[1] < 2:
        raise ValueError("vocab_size must be at least 2")
    rng = np.random.default_rng(seed)
    arr = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1e-12)
    cdf = np.cumsum(arr, axis=1)
    u = rng.random(size=(arr.shape[0], max(1, int(draws))))
    idx = np.asarray([np.searchsorted(cdf[row], u[row], side="right") for row in range(arr.shape[0])])
    x = (idx.astype(np.float64) + rng.random(size=idx.shape)) / float(arr.shape[1])
    x.sort(axis=1)
    n = x.shape[1]
    empirical_hi = np.arange(1, n + 1, dtype=np.float64) / float(n)
    empirical_lo = np.arange(0, n, dtype=np.float64) / float(n)
    d_plus = np.max(empirical_hi[None, :] - x, axis=1)
    d_minus = np.max(x - empirical_lo[None, :], axis=1)
    stat = np.maximum(d_plus, d_minus)
    return stat.astype(np.float64), _kolmogorov_pvalue(stat, n)


def permuted_categorical_uniform_ks(
    probs: np.ndarray,
    *,
    permutations: int = 32,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Order-robust KS distance from each categorical distribution to uniform.

    VQ code IDs are arbitrary, so a single codebook ordering can be misleading.
    This computes the categorical CDF KS distance after random code permutations
    and returns robust summaries.  Uniform p has distance 0 for every
    permutation; peaked p stays large for most permutations.
    """
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("probs must have shape (n_contexts, vocab_size)")
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


def ranked_probability_uniform_ks(probs: np.ndarray) -> np.ndarray:
    """Order-free KS-style distance between sorted token mass and uniform mass.

    VQ token IDs are not ordinal, so the most defensible uniformity statistic is
    invariant to codebook permutation.  Sorting the probabilities turns the
    categorical distribution into a concentration curve.  Uniform mass has CDF
    k / V after sorting; concentrated mass bows away from that line.
    """
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("probs must have shape (n_contexts, vocab_size)")
    if arr.shape[1] < 2:
        raise ValueError("vocab_size must be at least 2")
    arr = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1e-12)
    sorted_probs = np.sort(arr, axis=1)[:, ::-1]
    cdf = np.cumsum(sorted_probs, axis=1)
    uniform_cdf = np.arange(1, arr.shape[1] + 1, dtype=np.float64) / float(arr.shape[1])
    return np.max(np.abs(cdf - uniform_cdf[None, :]), axis=1)


def topk_branch_uniform_ks(probs: np.ndarray, *, top_k: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniformity among the most plausible next-scale code branches.

    Full-vocabulary uniformity is usually too strong for image tokens: most VQ
    codes are implausible under a visual prefix.  This statistic conditions on
    the top-k plausible continuations, renormalizes their mass, and measures
    whether that local branch posterior is flat.  Lower KS means more
    polysemous/branch-like among plausible continuations.
    """
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("probs must have shape (n_contexts, vocab_size)")
    k = max(2, min(int(top_k), int(arr.shape[1])))
    arr = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1e-12)
    top = np.sort(arr, axis=1)[:, -k:][:, ::-1]
    top_mass = np.sum(top, axis=1)
    local = top / np.maximum(top_mass[:, None], 1e-12)
    cdf = np.cumsum(local, axis=1)
    uniform_cdf = np.arange(1, k + 1, dtype=np.float64) / float(k)
    ks = np.max(np.abs(cdf - uniform_cdf[None, :]), axis=1)
    local_entropy = -np.sum(local * np.log(np.clip(local, 1e-12, 1.0)), axis=1) / math.log(k)
    return ks, local_entropy, top_mass


def _trimmed_mean(values: np.ndarray, *, axis: int = 0, trim: float = 0.10) -> np.ndarray:
    arr = np.sort(np.asarray(values, dtype=np.float64), axis=axis)
    n = arr.shape[axis]
    cut = int(math.floor(max(0.0, min(0.45, trim)) * n))
    if cut > 0 and n > 2 * cut:
        slc = [slice(None)] * arr.ndim
        slc[axis] = slice(cut, n - cut)
        arr = arr[tuple(slc)]
    return np.mean(arr, axis=axis)


def _mean_or_nan(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
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


def _permutation_mean_diff_pvalue(a: np.ndarray, b: np.ndarray, *, reps: int = 5000, seed: int = 0) -> tuple[float, float]:
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
    for _ in range(max(1, int(reps))):
        perm = rng.permutation(pooled)
        diff = float(perm[: a.size].mean() - perm[a.size :].mean())
        if abs(diff) >= abs(observed):
            extreme += 1
    return observed, float((extreme + 1.0) / (float(reps) + 1.0))


def _permutation_rate_diff_pvalue(a: np.ndarray, b: np.ndarray, *, reps: int = 5000, seed: int = 0) -> tuple[float, float]:
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
    for _ in range(max(1, int(reps))):
        perm = rng.permutation(pooled)
        diff = float(perm[: a.size].mean() - perm[a.size :].mean())
        if abs(diff) >= abs(observed):
            extreme += 1
    return observed, float((extreme + 1.0) / (float(reps) + 1.0))


def _tail_mask(values: np.ndarray, fraction: float, *, largest: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mask = np.zeros(arr.shape, dtype=bool)
    idx = np.flatnonzero(np.isfinite(arr))
    if idx.size == 0:
        return mask
    count = max(1, int(math.ceil(float(fraction) * idx.size)))
    order = idx[np.argsort(arr[idx])]
    mask[order[-count:] if largest else order[:count]] = True
    return mask


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    ar = np.argsort(np.argsort(a[mask])).astype(np.float64)
    br = np.argsort(np.argsort(b[mask])).astype(np.float64)
    if ar.std() <= 1e-12 or br.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def _load_image_folder(image_dir: Path, *, image_size: int, limit: int = 0) -> tuple[torch.Tensor, list[str]]:
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"})
    if limit > 0:
        paths = paths[: int(limit)]
    if not paths:
        raise FileNotFoundError(f"No images found under {image_dir}")
    tensors: list[torch.Tensor] = []
    names: list[str] = []
    for path in paths:
        img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
        names.append(path.name)
    return torch.stack(tensors, dim=0), names


def _load_embedding_pack_images(
    path: Path,
    *,
    dataset: str,
    image_size: int,
    limit: int = 0,
) -> tuple[torch.Tensor, list[str]]:
    pack = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(pack, dict) or "images" not in pack:
        raise ValueError(f"Embedding pack {path} must be a dict with an 'images' tensor")
    images = torch.as_tensor(pack["images"], dtype=torch.float32)
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Embedding pack images must have shape (B,3,H,W), got {tuple(images.shape)}")
    if limit > 0:
        images = images[: int(limit)]
    images01 = _denormalize_images(images, dataset)
    if images01.shape[-2:] != (image_size, image_size):
        images01 = F.interpolate(images01, size=(image_size, image_size), mode="bilinear", align_corners=False)
    ids = pack.get("image_ids")
    names = []
    for idx in range(int(images01.shape[0])):
        if ids is not None and len(ids) > idx:
            try:
                names.append(f"embedding_pack_image_{int(ids[idx])}")
            except Exception:
                names.append(f"embedding_pack_image_{idx}")
        else:
            names.append(f"embedding_pack_image_{idx}")
    return images01.clamp(0.0, 1.0), names


def _hidden_states(var, label_b: torch.Tensor, x_blcv_wo_first_l: torch.Tensor) -> torch.Tensor:
    bg, ed = var.begin_ends[var.prog_si] if var.prog_si >= 0 else (0, var.L)
    batch = int(x_blcv_wo_first_l.shape[0])
    autocast_off = torch.amp.autocast(device_type="cuda", enabled=False) if label_b.is_cuda else nullcontext()
    with autocast_off:
        sos = cond_bd = var.class_emb(label_b)
        sos = sos.unsqueeze(1).expand(batch, var.first_l, -1) + var.pos_start.expand(batch, var.first_l, -1)
        if var.prog_si == 0:
            x_blc = sos
        else:
            x_blc = torch.cat((sos, var.word_embed(x_blcv_wo_first_l.float())), dim=1)
        x_blc = x_blc + var.lvl_embed(var.lvl_1L[:, :ed].expand(batch, -1)) + var.pos_1LC[:, :ed]

    attn_bias = var.attn_bias_for_masking[:, :, :ed, :ed]
    cond_bd_or_gss = var.shared_ada_lin(cond_bd)
    main_type = torch.matmul(x_blc.new_ones(8, 8), x_blc.new_ones(8, 8)).dtype
    x_blc = x_blc.to(dtype=main_type)
    cond_bd_or_gss = cond_bd_or_gss.to(dtype=main_type)
    attn_bias = attn_bias.to(dtype=main_type)
    for block in var.blocks:
        x_blc = block(x=x_blc, cond_BD=cond_bd_or_gss, attn_bias=attn_bias)
    return x_blc.float()


@torch.no_grad()
def collect_var_pack(
    vae,
    var,
    *,
    images01: torch.Tensor,
    patch_nums: tuple[int, ...],
    labels: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model_dtype = next(vae.parameters()).dtype
    pixel_values = F.interpolate(images01.to(device), size=(16 * patch_nums[-1], 16 * patch_nums[-1]), mode="bilinear", align_corners=False)
    pixel_values = pixel_values.mul(2.0).sub(1.0).clamp(-1.0, 1.0).to(dtype=model_dtype)
    idx_bl = vae.img_to_idxBl(pixel_values, v_patch_nums=patch_nums)
    x_in = vae.quantize.idxBl_to_var_input(idx_bl)
    labels = labels.to(device=device, dtype=torch.long)
    hidden = _hidden_states(var, labels, x_in)
    logits = var(labels, x_in).float()
    targets = torch.cat(idx_bl, dim=1).long()
    start, end = var.begin_ends[-1]
    return {
        "tokens": hidden[:, start:end, :].detach().cpu(),
        "logits": logits[:, start:end, :].detach().cpu(),
        "targets": targets[:, start:end].detach().cpu(),
    }


def _make_heatmap_grid(
    *,
    images: np.ndarray,
    maps: np.ndarray,
    out_path: Path,
    title: str,
    cmap: str,
) -> Path:
    n = min(int(images.shape[0]), 8)
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols + 0.8, 4.1 * rows), squeeze=False)
    finite = maps[:n][np.isfinite(maps[:n])]
    vmin = float(np.quantile(finite, 0.02)) if finite.size else 0.0
    vmax = float(np.quantile(finite, 0.98)) if finite.size else 1.0
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0
    im = None
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(np.clip(images[i].transpose(1, 2, 0), 0.0, 1.0))
        im = ax.imshow(maps[i], cmap=cmap, alpha=0.58, vmin=vmin, vmax=vmax, extent=(0, images.shape[-1], images.shape[-2], 0))
        ax.set_title(f"image {i}", fontsize=10)
    fig.suptitle(title, fontsize=15)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.024, pad=0.015)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_distribution_overlay(
    *,
    singular: np.ndarray,
    regular: np.ndarray,
    out_path: Path,
    xlabel: str,
    title: str,
) -> Path:
    singular = np.asarray(singular, dtype=np.float64)
    regular = np.asarray(regular, dtype=np.float64)
    singular = singular[np.isfinite(singular)]
    regular = regular[np.isfinite(regular)]
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        float(min(np.min(singular), np.min(regular))) if singular.size and regular.size else 0.0,
        float(max(np.max(singular), np.max(regular))) if singular.size and regular.size else 1.0,
        42,
    )
    ax.hist(regular, bins=bins, density=True, alpha=0.55, label=f"regular n={regular.size}", color="#4c78a8")
    ax.hist(singular, bins=bins, density=True, alpha=0.55, label=f"singular n={singular.size}", color="#f58518")
    ax.axvline(_mean_or_nan(regular), color="#4c78a8", linestyle="--", linewidth=2)
    ax.axvline(_mean_or_nan(singular), color="#f58518", linestyle="--", linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _plot_scatter(irregularity: np.ndarray, ks_stat: np.ndarray, entropy_norm: np.ndarray, out_path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    axes[0].scatter(irregularity, ks_stat, s=12, alpha=0.55, color="#4c78a8")
    axes[0].set_xlabel("fiber irregularity")
    axes[0].set_ylabel("branch KS D to uniform mass")
    axes[0].set_title("Uniformity vs singularity")
    axes[1].scatter(irregularity, entropy_norm, s=12, alpha=0.55, color="#54a24b")
    axes[1].set_xlabel("fiber irregularity")
    axes[1].set_ylabel("normalized entropy")
    axes[1].set_title("Entropy vs singularity")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _import_real_wandb():
    cached = sys.modules.pop("wandb", None)
    original_path = list(sys.path)
    repo_text = str(REPO_ROOT.resolve())
    try:
        sys.path = [item for item in sys.path if item not in {"", "."} and str(Path(item).resolve()) != repo_text]
        module = importlib.import_module("wandb")
        if not hasattr(module, "init"):
            raise ImportError("imported wandb module has no init()")
        return module
    except Exception:
        if cached is not None:
            sys.modules["wandb"] = cached
        raise
    finally:
        sys.path = original_path


def maybe_log_wandb(args: argparse.Namespace, summary: dict[str, Any], image_paths: dict[str, Path]) -> str | None:
    if not args.wandb:
        return None
    try:
        wandb = _import_real_wandb()
    except Exception as exc:
        print(f"[wandb] unavailable: {exc}", flush=True)
        return None
    try:
        wandb_dir = args.out_dir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            tags=[tag for tag in str(args.wandb_tags).split(",") if tag],
            config=summary,
            dir=str(wandb_dir),
            mode=args.wandb_mode or None,
        )
        payload = {f"pretrained_var_ks/{key}": value for key, value in summary.items() if isinstance(value, (int, float, str))}
        for key, path in image_paths.items():
            payload[f"pretrained_var_ks/{key}"] = wandb.Image(str(path)) if path.suffix.lower() == ".png" else str(path)
        wandb.log(payload, step=0)
        artifact = wandb.Artifact(f"{args.wandb_name or 'pretrained_var_ks'}_outputs", type="analysis")
        for path in image_paths.values():
            artifact.add_file(str(path))
        summary_path = args.out_dir / "pretrained_var_one_sample_ks_summary.json"
        if summary_path.exists():
            artifact.add_file(str(summary_path))
        wandb.log_artifact(artifact)
        url = getattr(run, "url", None)
        wandb.finish()
        return url
    except Exception as exc:
        print(f"[wandb] logging failed: {exc}", flush=True)
        try:
            wandb.finish()
        except Exception:
            pass
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Order-free KS-style uniformity probe for pretrained VAR logits.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--repo-id", default="FoundationVision/var")
    parser.add_argument("--vae-filename", default="vae_ch160v4096z32.pth")
    parser.add_argument("--var-filename", default=None)
    parser.add_argument("--var-repo-path", default=None)
    parser.add_argument("--model-depth", type=int, default=16, choices=[16, 20, 24, 30, 36])
    parser.add_argument("--patch-nums", default="auto")
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--embedding-pack", type=Path, default=None, help="Optional epoch_*.pt pack with normalized images.")
    parser.add_argument("--embedding-dataset", default="COCO", help="Dataset normalization used by --embedding-pack images.")
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--class-labels", default="980,437")
    parser.add_argument("--teacher-class-labels", default=None, help="Defaults to generation labels; use -1 for unconditional image-dir probing.")
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--ks-draws", type=int, default=512)
    parser.add_argument("--ks-permutations", type=int, default=24)
    parser.add_argument("--branch-top-k", type=int, default=32)
    parser.add_argument("--vol-min", type=int, default=8)
    parser.add_argument("--vol-max", type=int, default=64)
    parser.add_argument("--ws", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--nstrat", type=int, default=3)
    parser.add_argument("--tail-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--model-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--mmap-load", action="store_true", help="Memory-map checkpoint tensors during torch.load.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="stratified-manifold-learning")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-tags", default="pretrained-var,one-sample-ks")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = REPO_ROOT / "runs" / "local" / "pretrained_var_one_sample_ks" / stamp
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.wandb_mode == "disabled":
        args.wandb = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[str(args.model_dtype)]

    defaults = resolve_model_defaults(args.model_depth)
    var_filename = args.var_filename or str(defaults["filename"])
    patch_nums = parse_patch_nums(args.patch_nums, resolution=int(defaults["resolution"]))
    image_size = 16 * int(patch_nums[-1])
    labels = parse_class_labels(args.class_labels, samples=args.samples)
    if labels is None:
        labels = [int(x) for x in np.random.default_rng(args.seed).integers(0, 1000, size=int(args.samples))]

    print(f"[setup] loading pretrained VAR-d{args.model_depth} on {device}", flush=True)
    vae, var, _vae_path, _var_path = build_pretrained_var(
        depth=args.model_depth,
        repo_id=args.repo_id,
        vae_filename=args.vae_filename,
        var_filename=var_filename,
        patch_nums=patch_nums,
        shared_aln=bool(defaults["shared_aln"]),
        device=device,
        var_repo_path=args.var_repo_path,
        dtype=dtype if dtype is not torch.float32 else None,
        mmap_load=bool(args.mmap_load),
    )

    image_names: list[str]
    if args.embedding_pack is not None:
        images01, image_names = _load_embedding_pack_images(
            args.embedding_pack,
            dataset=args.embedding_dataset,
            image_size=image_size,
            limit=args.limit_images,
        )
        if args.teacher_class_labels is None or str(args.teacher_class_labels).strip() == "-1":
            teacher_values = [-1] * int(images01.shape[0])
        else:
            parsed = parse_class_labels(args.teacher_class_labels, samples=int(images01.shape[0]))
            teacher_values = parsed or [-1] * int(images01.shape[0])
    elif args.image_dir is not None:
        images01, image_names = _load_image_folder(args.image_dir, image_size=image_size, limit=args.limit_images)
        if args.teacher_class_labels is None:
            teacher_values = [-1] * int(images01.shape[0])
        else:
            if str(args.teacher_class_labels).strip() == "-1":
                teacher_values = [-1] * int(images01.shape[0])
            else:
                parsed = parse_class_labels(args.teacher_class_labels, samples=int(images01.shape[0]))
                teacher_values = parsed or [-1] * int(images01.shape[0])
    else:
        print(f"[sample] generating {len(labels)} VAR images for teacher-forced probing", flush=True)
        label_b = torch.tensor(labels, dtype=torch.long, device=device)
        with torch.inference_mode():
            images01 = var.autoregressive_infer_cfg(
                B=len(labels),
                label_B=label_b,
                cfg=float(args.cfg),
                top_k=int(args.top_k),
                top_p=float(args.top_p),
                g_seed=int(args.seed),
                more_smooth=False,
            ).detach().cpu()
        image_names = [f"generated_class_{label:03d}_{idx:03d}" for idx, label in enumerate(labels)]
        teacher_values = labels

    teacher_tensor = torch.tensor([var.num_classes if int(label) < 0 else int(label) for label in teacher_values], dtype=torch.long)
    image_paths: dict[str, Path] = {}
    image_paths["input_grid"] = save_grid(
        images01,
        args.out_dir / "pretrained_var_ks_inputs.png",
        labels=[int(x) for x in teacher_values],
        title="Images probed by pretrained VAR",
        cols=min(4, int(images01.shape[0])),
    )

    print("[var] collecting teacher-forced logits and hidden states", flush=True)
    pack = collect_var_pack(vae, var, images01=images01, patch_nums=patch_nums, labels=teacher_tensor, device=device)
    logits = pack["logits"].float()
    targets = pack["targets"].long()
    tokens = pack["tokens"].float()
    batch, grid_tokens, vocab_size = logits.shape
    grid_size = int(round(math.sqrt(grid_tokens)))
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    log_probs = flat_logits.log_softmax(dim=-1)
    probs = log_probs.exp().numpy().astype(np.float64)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1).numpy().astype(np.float64)
    entropy_norm = entropy / max(math.log(vocab_size), 1e-12)
    nll = (-log_probs.gather(-1, flat_targets[:, None]).squeeze(-1)).numpy().astype(np.float64)
    top2 = log_probs.exp().topk(k=2, dim=-1).values.numpy().astype(np.float64)
    top1_prob = top2[:, 0]
    top2_margin = top2[:, 0] - top2[:, 1]

    print("[ks] running uniformity diagnostics", flush=True)
    ordered_ks_stat, ordered_ks_pvalue = one_sample_ks_uniform_draws(probs, draws=args.ks_draws, seed=args.seed)
    ks_stat = ranked_probability_uniform_ks(probs)
    branch_ks, branch_entropy_norm, branch_topk_mass = topk_branch_uniform_ks(probs, top_k=args.branch_top_k)
    perm_ks = permuted_categorical_uniform_ks(probs, permutations=args.ks_permutations, seed=args.seed + 17)

    print("[fiber] estimating singularity scores in VAR hidden-token space", flush=True)
    fiber_results, _sorted_dists, _unsorted_dists = analyze_stratification(
        tokens.reshape(-1, tokens.shape[-1]),
        vol_min=args.vol_min,
        vol_max=args.vol_max,
        ws=args.ws,
        alpha=args.alpha,
        nstrat=args.nstrat,
    )
    singular = fiber_singularity_scores(fiber_results, alpha=args.alpha)
    irregularity = singular["irregularity"]
    rejected = singular["rejected"]
    quiet = ~rejected
    high_irregular = _tail_mask(irregularity, args.tail_fraction, largest=True)
    low_irregular = _tail_mask(irregularity, args.tail_fraction, largest=False)
    flat_branch = _tail_mask(branch_ks, args.tail_fraction, largest=False)
    sharp_branch = _tail_mask(branch_ks, args.tail_fraction, largest=True)
    high_branch_entropy = _tail_mask(branch_entropy_norm, args.tail_fraction, largest=True)
    flat_high_entropy_branch = flat_branch & high_branch_entropy

    ks_diff, ks_diff_p = _permutation_mean_diff_pvalue(ks_stat[rejected], ks_stat[quiet], seed=args.seed)
    entropy_diff, entropy_diff_p = _permutation_mean_diff_pvalue(entropy_norm[rejected], entropy_norm[quiet], seed=args.seed + 1)
    high_ks_diff, high_ks_diff_p = _permutation_mean_diff_pvalue(ks_stat[high_irregular], ks_stat[low_irregular], seed=args.seed + 2)
    high_entropy_diff, high_entropy_diff_p = _permutation_mean_diff_pvalue(
        entropy_norm[high_irregular],
        entropy_norm[low_irregular],
        seed=args.seed + 3,
    )
    flat_irregularity_diff, flat_irregularity_p = _permutation_mean_diff_pvalue(
        irregularity[flat_branch],
        irregularity[~flat_branch],
        seed=args.seed + 4,
    )
    flat_reject_rate_diff, flat_reject_rate_p = _permutation_rate_diff_pvalue(
        rejected[flat_branch].astype(np.float64),
        rejected[~flat_branch].astype(np.float64),
        seed=args.seed + 5,
    )
    flat_vs_sharp_irregularity_diff, flat_vs_sharp_irregularity_p = _permutation_mean_diff_pvalue(
        irregularity[flat_branch],
        irregularity[sharp_branch],
        seed=args.seed + 6,
    )
    flat_vs_sharp_reject_rate_diff, flat_vs_sharp_reject_rate_p = _permutation_rate_diff_pvalue(
        rejected[flat_branch].astype(np.float64),
        rejected[sharp_branch].astype(np.float64),
        seed=args.seed + 7,
    )

    summary = {
        "repo_id": args.repo_id,
        "model_depth": int(args.model_depth),
        "var_filename": var_filename,
        "reported_fid": float(defaults["fid"]),
        "reported_params": str(defaults["params"]),
        "device": str(device),
        "model_dtype": str(dtype).replace("torch.", ""),
        "mmap_load": bool(args.mmap_load),
        "source": "embedding_pack" if args.embedding_pack is not None else ("image_dir" if args.image_dir is not None else "generated"),
        "embedding_pack": str(args.embedding_pack) if args.embedding_pack is not None else None,
        "embedding_dataset": str(args.embedding_dataset),
        "image_dir": str(args.image_dir) if args.image_dir is not None else None,
        "image_names": image_names,
        "num_images": int(batch),
        "num_tokens": int(batch * grid_tokens),
        "grid_size": int(grid_size),
        "vocab_size": int(vocab_size),
        "ks_draws": int(args.ks_draws),
        "ks_permutations": int(args.ks_permutations),
        "branch_top_k": int(args.branch_top_k),
        "alpha": float(args.alpha),
        "fiber_violation_reject_count": int(rejected.sum()),
        "high_irregular_count": int(high_irregular.sum()),
        "low_irregular_count": int(low_irregular.sum()),
        "flat_branch_count": int(flat_branch.sum()),
        "sharp_branch_count": int(sharp_branch.sum()),
        "high_branch_entropy_count": int(high_branch_entropy.sum()),
        "flat_high_entropy_branch_count": int(flat_high_entropy_branch.sum()),
        "mean_ks_stat": _mean_or_nan(ks_stat),
        "mean_ordered_draw_ks_stat": _mean_or_nan(ordered_ks_stat),
        "mean_ordered_draw_ks_pvalue": _mean_or_nan(ordered_ks_pvalue),
        "mean_branch_ks": _mean_or_nan(branch_ks),
        "mean_branch_entropy_norm": _mean_or_nan(branch_entropy_norm),
        "mean_branch_topk_mass": _mean_or_nan(branch_topk_mass),
        "mean_entropy_norm": _mean_or_nan(entropy_norm),
        "mean_nll": _mean_or_nan(nll),
        "mean_top1_prob": _mean_or_nan(top1_prob),
        "mean_top2_margin": _mean_or_nan(top2_margin),
        "mean_permuted_order_ks_median": _mean_or_nan(perm_ks["median"]),
        "mean_ks_stat_rejected": _mean_or_nan(ks_stat[rejected]),
        "mean_ks_stat_regular": _mean_or_nan(ks_stat[quiet]),
        "diff_ks_rejected_minus_regular": ks_diff,
        "perm_p_ks_rejected_vs_regular": ks_diff_p,
        "cohen_d_ks_rejected_vs_regular": _cohen_d(ks_stat[rejected], ks_stat[quiet]),
        "mean_entropy_rejected": _mean_or_nan(entropy_norm[rejected]),
        "mean_entropy_regular": _mean_or_nan(entropy_norm[quiet]),
        "diff_entropy_rejected_minus_regular": entropy_diff,
        "perm_p_entropy_rejected_vs_regular": entropy_diff_p,
        "cohen_d_entropy_rejected_vs_regular": _cohen_d(entropy_norm[rejected], entropy_norm[quiet]),
        "mean_ks_stat_high_irregular": _mean_or_nan(ks_stat[high_irregular]),
        "mean_ks_stat_low_irregular": _mean_or_nan(ks_stat[low_irregular]),
        "diff_ks_high_minus_low_irregular": high_ks_diff,
        "perm_p_ks_high_vs_low_irregular": high_ks_diff_p,
        "cohen_d_ks_high_vs_low_irregular": _cohen_d(ks_stat[high_irregular], ks_stat[low_irregular]),
        "mean_entropy_high_irregular": _mean_or_nan(entropy_norm[high_irregular]),
        "mean_entropy_low_irregular": _mean_or_nan(entropy_norm[low_irregular]),
        "diff_entropy_high_minus_low_irregular": high_entropy_diff,
        "perm_p_entropy_high_vs_low_irregular": high_entropy_diff_p,
        "cohen_d_entropy_high_vs_low_irregular": _cohen_d(entropy_norm[high_irregular], entropy_norm[low_irregular]),
        "ks_2samp_stat_rejected_vs_regular": ks_2samp(ks_stat[rejected], ks_stat[quiet]).statistic,
        "ks_2samp_p_rejected_vs_regular": ks_2samp(ks_stat[rejected], ks_stat[quiet]).pvalue,
        "corr_irregularity_ks_spearman": _rank_corr(irregularity, ks_stat),
        "corr_irregularity_entropy_spearman": _rank_corr(irregularity, entropy_norm),
        "corr_irregularity_branch_ks_spearman": _rank_corr(irregularity, branch_ks),
        "corr_irregularity_branch_entropy_spearman": _rank_corr(irregularity, branch_entropy_norm),
        "mean_irregularity_flat_branch": _mean_or_nan(irregularity[flat_branch]),
        "mean_irregularity_not_flat_branch": _mean_or_nan(irregularity[~flat_branch]),
        "diff_irregularity_flat_branch_minus_rest": flat_irregularity_diff,
        "perm_p_irregularity_flat_branch_vs_rest": flat_irregularity_p,
        "reject_rate_flat_branch": _mean_or_nan(rejected[flat_branch].astype(np.float64)),
        "reject_rate_not_flat_branch": _mean_or_nan(rejected[~flat_branch].astype(np.float64)),
        "diff_reject_rate_flat_branch_minus_rest": flat_reject_rate_diff,
        "perm_p_reject_rate_flat_branch_vs_rest": flat_reject_rate_p,
        "mean_irregularity_sharp_branch": _mean_or_nan(irregularity[sharp_branch]),
        "diff_irregularity_flat_branch_minus_sharp_branch": flat_vs_sharp_irregularity_diff,
        "perm_p_irregularity_flat_branch_vs_sharp_branch": flat_vs_sharp_irregularity_p,
        "reject_rate_sharp_branch": _mean_or_nan(rejected[sharp_branch].astype(np.float64)),
        "diff_reject_rate_flat_branch_minus_sharp_branch": flat_vs_sharp_reject_rate_diff,
        "perm_p_reject_rate_flat_branch_vs_sharp_branch": flat_vs_sharp_reject_rate_p,
        "claim_direction_supported_flat_branch": bool(
            np.isfinite(flat_irregularity_diff)
            and np.isfinite(flat_reject_rate_diff)
            and flat_irregularity_diff > 0.0
            and flat_reject_rate_diff > 0.0
        ),
        "claim_direction_supported_rejected": bool(np.isfinite(ks_diff) and np.isfinite(entropy_diff) and ks_diff < 0.0 and entropy_diff > 0.0),
        "claim_direction_supported_high_irregular": bool(
            np.isfinite(high_ks_diff) and np.isfinite(high_entropy_diff) and high_ks_diff < 0.0 and high_entropy_diff > 0.0
        ),
    }

    ks_maps = ks_stat.reshape(batch, grid_size, grid_size)
    branch_ks_maps = branch_ks.reshape(batch, grid_size, grid_size)
    entropy_maps = entropy_norm.reshape(batch, grid_size, grid_size)
    irregularity_maps = irregularity.reshape(batch, grid_size, grid_size)
    images_np = images01.numpy().astype(np.float64)
    image_paths["ks_heatmaps"] = _make_heatmap_grid(
        images=images_np,
        maps=ks_maps,
        out_path=args.out_dir / "ks_to_uniform_heatmaps.png",
        title="Order-free ranked KS distance to uniform next-scale code mass",
        cmap="viridis",
    )
    image_paths["branch_ks_heatmaps"] = _make_heatmap_grid(
        images=images_np,
        maps=branch_ks_maps,
        out_path=args.out_dir / "branch_topk_ks_heatmaps.png",
        title=f"Top-{int(args.branch_top_k)} branch KS distance to uniform local mass",
        cmap="viridis",
    )
    image_paths["entropy_heatmaps"] = _make_heatmap_grid(
        images=images_np,
        maps=entropy_maps,
        out_path=args.out_dir / "entropy_heatmaps.png",
        title="Normalized VAR next-scale code entropy",
        cmap="magma",
    )
    image_paths["irregularity_heatmaps"] = _make_heatmap_grid(
        images=images_np,
        maps=irregularity_maps,
        out_path=args.out_dir / "fiber_irregularity_heatmaps.png",
        title="VAR hidden-token fiber irregularity",
        cmap="inferno",
    )
    image_paths["ks_distribution"] = _plot_distribution_overlay(
        singular=ks_stat[rejected],
        regular=ks_stat[quiet],
        out_path=args.out_dir / "ks_distribution_singular_vs_regular.png",
        xlabel="ranked probability KS D to uniform categorical mass",
        title="Singular fine-scale patches should shift left if next-scale code mass flattens",
    )
    image_paths["branch_ks_distribution"] = _plot_distribution_overlay(
        singular=branch_ks[rejected],
        regular=branch_ks[quiet],
        out_path=args.out_dir / "branch_ks_distribution_singular_vs_regular.png",
        xlabel=f"top-{int(args.branch_top_k)} branch KS D to uniform local mass",
        title="Singular tokens should shift left if plausible branches flatten",
    )
    image_paths["scatter"] = _plot_scatter(
        irregularity=irregularity,
        ks_stat=branch_ks,
        entropy_norm=branch_entropy_norm,
        out_path=args.out_dir / "irregularity_vs_ks_entropy.png",
    )

    token_rows = []
    for idx in range(int(batch * grid_tokens)):
        token_rows.append(
            {
                "token_index": idx,
                "image_id": int(idx // grid_tokens),
                "patch_id": int(idx % grid_tokens),
                "row": int((idx % grid_tokens) // grid_size),
                "col": int((idx % grid_tokens) % grid_size),
                "ranked_ks_stat": float(ks_stat[idx]),
                "branch_ks_stat": float(branch_ks[idx]),
                "branch_entropy_norm": float(branch_entropy_norm[idx]),
                "branch_topk_mass": float(branch_topk_mass[idx]),
                "flat_branch_tail": bool(flat_branch[idx]),
                "flat_high_entropy_branch_tail": bool(flat_high_entropy_branch[idx]),
                "ordered_draw_ks_stat": float(ordered_ks_stat[idx]),
                "ordered_draw_ks_pvalue": float(ordered_ks_pvalue[idx]),
                "permuted_order_ks_median": float(perm_ks["median"][idx]),
                "entropy_norm": float(entropy_norm[idx]),
                "nll": float(nll[idx]),
                "top1_prob": float(top1_prob[idx]),
                "top2_margin": float(top2_margin[idx]),
                "irregularity": float(irregularity[idx]),
                "fiber_violation_reject": bool(rejected[idx]),
            }
        )
    payload = {"summary": summary, "figures": {key: str(path) for key, path in image_paths.items()}, "tokens": token_rows}
    summary_path = args.out_dir / "pretrained_var_one_sample_ks_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(to_jsonable(payload), fp, indent=2)
    image_paths["summary"] = summary_path

    url = maybe_log_wandb(args, summary, image_paths)
    print(json.dumps(to_jsonable({"summary": summary, "figures": payload["figures"], "summary_path": summary_path}), indent=2), flush=True)
    if url:
        print(f"[wandb] {url}", flush=True)


if __name__ == "__main__":
    main()
