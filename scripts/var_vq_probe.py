"""Probe the VQ visual vocabulary used by VAR.

This is a more primitive companion to the VAR autoregressive probe. Instead of
studying teacher-forced transformer hidden states, it studies the learned VQ-VAE
codebook directly and then projects codebook-level geometry back onto COCO image
patch assignments.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import create_data_loaders  # noqa: E402
from fiber.figure_io import save_figure  # noqa: E402
from fiber.geometry import (  # noqa: E402
    min_change_pvalue,
    min_fiber_violation_pvalue,
    run_fiber_bundle_test,
    summarize_stratifications,
)
from utils import denormalize_images  # noqa: E402


DEFAULT_PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)


def _resolve_var_repo_path(var_repo_path: str | None = None) -> Path:
    candidates: list[Path] = []
    if var_repo_path:
        candidates.append(Path(var_repo_path))
    env_path = os.environ.get("VAR_REPO_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([REPO_ROOT / "external" / "VAR", Path.cwd() / "external" / "VAR"])
    for candidate in candidates:
        if (candidate / "models" / "vqvae.py").exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not find FoundationVision/VAR; expected external/VAR.")


def _import_var_models(var_repo_path: str | None = None) -> ModuleType:
    repo_path = _resolve_var_repo_path(var_repo_path)
    repo_str = str(repo_path)
    saved_models = sys.modules.get("models")
    saved_path = list(sys.path)
    try:
        sys.path.insert(0, repo_str)
        if saved_models is not None:
            del sys.modules["models"]
        return importlib.import_module("models")
    finally:
        if saved_models is not None:
            sys.modules["models"] = saved_models
        else:
            sys.modules.pop("models", None)
        sys.path[:] = saved_path


def _load_torch_state_dict(path: str, *, map_location: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _parse_patch_nums(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError("patch nums cannot be empty")
    return values


def _load_vae(
    *,
    device: torch.device,
    repo_id: str,
    vae_filename: str,
    patch_nums: tuple[int, ...],
    var_repo_path: str | None,
):
    var_models = _import_var_models(var_repo_path)
    vae = var_models.VQVAE(
        vocab_size=4096,
        z_channels=32,
        ch=160,
        test_mode=True,
        share_quant_resi=4,
        v_patch_nums=patch_nums,
    ).to(device)
    vae_path = hf_hub_download(repo_id=repo_id, filename=vae_filename)
    vae.load_state_dict(_load_torch_state_dict(vae_path, map_location=device), strict=True)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    return vae


def _first_dims(results: list[dict[str, list[float]]]) -> np.ndarray:
    return np.asarray(
        [res["dimensions"][0] if res and res.get("dimensions") else np.nan for res in results],
        dtype=np.float64,
    )


def _min_pvalues(results: list[dict[str, list[float]]], *, kind: str) -> np.ndarray:
    out = np.full(len(results), np.nan, dtype=np.float64)
    for idx, res in enumerate(results):
        if kind == "fiber":
            val = min_fiber_violation_pvalue(res)
        elif kind == "change":
            val = min_change_pvalue(res)
        else:
            raise ValueError(kind)
        if math.isfinite(val):
            out[idx] = val
    return out


def _irregularity(results: list[dict[str, list[float]]], *, alpha: float) -> np.ndarray:
    pvals = _min_pvalues(results, kind="fiber")
    out = np.zeros(pvals.shape, dtype=np.float64)
    mask = np.isfinite(pvals) & (pvals < alpha)
    out[mask] = -np.log10(np.clip(pvals[mask], 1e-12, None))
    return out


def _pca2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def _finite_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _summarize_usage(final_ids: np.ndarray, *, vocab_size: int) -> dict[str, Any]:
    counts = np.bincount(final_ids.reshape(-1), minlength=vocab_size).astype(np.int64)
    used = counts > 0
    probs = counts[used].astype(np.float64) / max(1, int(counts.sum()))
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum()) if probs.size else 0.0
    top = np.argsort(-counts)[:20]
    return {
        "num_tokens": int(counts.sum()),
        "vocab_size": int(vocab_size),
        "used_codes": int(used.sum()),
        "used_fraction": float(used.mean()),
        "effective_vocab": float(np.exp(entropy)),
        "entropy": entropy,
        "top_codes": [
            {"code": int(code), "count": int(counts[code]), "fraction": float(counts[code] / max(1, counts.sum()))}
            for code in top
            if int(counts[code]) > 0
        ],
    }


@torch.no_grad()
def _collect_final_vq_ids(
    *,
    vae,
    loader,
    dataset_name: str,
    device: torch.device,
    patch_nums: tuple[int, ...],
    num_images: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_size = 16 * int(patch_nums[-1])
    all_ids: list[np.ndarray] = []
    all_images: list[np.ndarray] = []
    all_dataset_indices: list[int] = []
    remaining = int(num_images)
    for batch in loader:
        imgs = batch[0]
        batch_indices = batch[2] if len(batch) > 2 else torch.arange(imgs.shape[0])
        take = min(remaining, int(imgs.shape[0]))
        imgs = imgs[:take].to(device, non_blocking=True)
        imgs01 = denormalize_images(imgs, dataset_name)
        if imgs01.shape[-2:] != (expected_size, expected_size):
            imgs01 = F.interpolate(
                imgs01,
                size=(expected_size, expected_size),
                mode="bilinear",
                align_corners=False,
            )
        pixels = imgs01.mul(2.0).sub(1.0).clamp(-1.0, 1.0)
        idx_bl = vae.img_to_idxBl(pixels, v_patch_nums=patch_nums)
        final = idx_bl[-1].detach().cpu().numpy().astype(np.int64)
        all_ids.append(final)
        all_images.append(imgs01.detach().cpu().permute(0, 2, 3, 1).numpy())
        all_dataset_indices.extend(int(x) for x in batch_indices[:take].detach().cpu().tolist())
        remaining -= take
        if remaining <= 0:
            break
    if not all_ids:
        raise RuntimeError("No images were collected for the VQ probe.")
    return (
        np.concatenate(all_ids, axis=0),
        np.concatenate(all_images, axis=0),
        np.asarray(all_dataset_indices, dtype=np.int64),
    )


def _plot_codebook_pca(
    *,
    codebook: np.ndarray,
    dims: np.ndarray,
    irregularity: np.ndarray,
    usage_counts: np.ndarray | None,
    out_path: Path,
) -> None:
    xy = _pca2(codebook)
    usage = np.zeros(codebook.shape[0], dtype=np.float64) if usage_counts is None else np.asarray(usage_counts, dtype=np.float64)

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "figure.titlesize": 22,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), constrained_layout=True)
    panels = [
        (dims, "local dimension", "viridis"),
        (irregularity, "fiber violation -log10(p)", "magma"),
        (np.log1p(usage), "log(1 + COCO usage)", "cividis"),
    ]
    for ax, (values, title, cmap) in zip(axes, panels):
        finite = np.isfinite(values)
        scatter = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=np.where(finite, values, np.nan),
            s=np.where(usage > 0, 18.0, 7.0),
            alpha=np.where(usage > 0, 0.86, 0.28),
            cmap=cmap,
            linewidths=0.0,
        )
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.15)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.025)
    fig.suptitle("VAR VQ Codebook Geometry")
    save_figure(fig, out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_patch_projection(
    *,
    images: np.ndarray,
    final_ids: np.ndarray,
    code_values: np.ndarray,
    title: str,
    colorbar_label: str,
    out_path: Path,
    cmap: str,
) -> None:
    num_images = min(16, int(images.shape[0]))
    grid = int(round(math.sqrt(final_ids.shape[1])))
    values = code_values[final_ids[:num_images].reshape(-1)].reshape(num_images, grid, grid)
    finite = values[np.isfinite(values)]
    vmin = float(np.nanpercentile(finite, 2)) if finite.size else 0.0
    vmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.titlesize": 22,
        }
    )
    fig, axes = plt.subplots(4, 4, figsize=(18, 18.5), constrained_layout=True)
    overlay_mappable = None
    for panel_idx, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if panel_idx >= num_images:
            continue
        ax.imshow(np.clip(images[panel_idx], 0.0, 1.0))
        overlay_mappable = ax.imshow(
            values[panel_idx],
            cmap=cmap,
            alpha=0.55,
            interpolation="nearest",
            extent=(0, images.shape[2], images.shape[1], 0),
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"image {panel_idx} | {grid}x{grid} VQ tokens")
        h, w = images.shape[1], images.shape[2]
        for pos in range(1, grid):
            ax.axhline(pos * h / grid, color="white", linewidth=0.35, alpha=0.55)
            ax.axvline(pos * w / grid, color="white", linewidth=0.35, alpha=0.55)
    if overlay_mappable is not None:
        cbar = fig.colorbar(overlay_mappable, ax=axes.ravel().tolist(), fraction=0.030, pad=0.018)
        cbar.set_label(colorbar_label)
    fig.suptitle(title)
    save_figure(fig, out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_usage_histogram(*, counts: np.ndarray, out_path: Path) -> None:
    used_counts = counts[counts > 0]
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "figure.titlesize": 22,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    axes[0].hist(used_counts, bins=min(50, max(8, int(np.sqrt(max(1, used_counts.size))))), color="#4c78a8")
    axes[0].set_title("Used-code count distribution")
    axes[0].set_xlabel("count in sampled COCO tokens")
    axes[0].set_ylabel("number of VQ codes")
    axes[1].plot(np.sort(counts)[::-1], color="#f58518", linewidth=2.0)
    axes[1].set_title("Ranked VQ code usage")
    axes[1].set_xlabel("code rank")
    axes[1].set_ylabel("count")
    axes[1].set_yscale("symlog")
    axes[1].grid(alpha=0.2)
    fig.suptitle("VAR VQ Usage on COCO")
    save_figure(fig, out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _run_geometry(name: str, embeddings: np.ndarray, *, args: argparse.Namespace) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
    results, _sorted_dists, _unsorted_dists = run_fiber_bundle_test(
        tensor,
        vol_min=args.vol_min,
        vol_max=args.vol_max,
        ws=args.ws,
        alpha=args.alpha,
        nstrat=args.nstrat,
    )
    summary = summarize_stratifications(results, alpha=args.alpha)
    dims = _first_dims(results)
    irregularity = _irregularity(results, alpha=args.alpha)
    summary["name"] = name
    summary["dimension_irregularity_corr"] = _finite_corr(dims, irregularity)
    return summary, dims, irregularity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--dataset", default="COCO")
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--repo-id", default="FoundationVision/var")
    parser.add_argument("--vae-filename", default="vae_ch160v4096z32.pth")
    parser.add_argument("--var-repo-path", default=None)
    parser.add_argument("--patch-nums", default=",".join(str(x) for x in DEFAULT_PATCH_NUMS))
    parser.add_argument("--vol-min", type=int, default=8)
    parser.add_argument("--vol-max", type=int, default=64)
    parser.add_argument("--ws", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--nstrat", type=int, default=3)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_nums = _parse_patch_nums(args.patch_nums)
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO_ROOT / "runs" / "local" / "coco_var_vq_probe" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    vae = _load_vae(
        device=device,
        repo_id=args.repo_id,
        vae_filename=args.vae_filename,
        patch_nums=patch_nums,
        var_repo_path=args.var_repo_path,
    )
    codebook = vae.quantize.embedding.weight.detach().float().cpu().numpy()

    _train_loader, test_loader, _num_classes, _in_chans, img_size, _task = create_data_loaders(
        dataset_name=args.dataset,
        root=args.data_root,
        img_size=224,
        batch_size_train=args.batch_size,
        batch_size_test=args.batch_size,
        num_workers=args.num_workers,
        subset_test=args.num_images,
        device=device,
        distributed=False,
    )
    final_ids, images, dataset_indices = _collect_final_vq_ids(
        vae=vae,
        loader=test_loader,
        dataset_name=args.dataset,
        device=device,
        patch_nums=patch_nums,
        num_images=args.num_images,
    )
    counts = np.bincount(final_ids.reshape(-1), minlength=codebook.shape[0]).astype(np.int64)
    used_ids = np.flatnonzero(counts > 0)

    codebook_summary, codebook_dims, codebook_irregularity = _run_geometry(
        "var_vq_codebook",
        codebook,
        args=args,
    )
    used_summary, used_dims, used_irregularity = _run_geometry(
        "var_vq_used_codes",
        codebook[used_ids],
        args=args,
    )

    usage_summary = _summarize_usage(final_ids, vocab_size=codebook.shape[0])
    usage_summary["dataset_indices"] = dataset_indices.astype(int).tolist()
    usage_summary["image_size"] = int(img_size)
    usage_summary["vq_grid"] = [int(patch_nums[-1]), int(patch_nums[-1])]

    used_dim_lookup = np.full(codebook.shape[0], np.nan, dtype=np.float64)
    used_irr_lookup = np.zeros(codebook.shape[0], dtype=np.float64)
    used_dim_lookup[used_ids] = used_dims
    used_irr_lookup[used_ids] = used_irregularity

    outputs = {
        "codebook_pca": str(out_dir / "var_vq_codebook_geometry.png"),
        "usage": str(out_dir / "var_vq_usage.png"),
        "codebook_dimension_projection": str(out_dir / "var_vq_codebook_dimension_on_images.png"),
        "codebook_irregularity_projection": str(out_dir / "var_vq_codebook_irregularity_on_images.png"),
        "used_code_dimension_projection": str(out_dir / "var_vq_used_code_dimension_on_images.png"),
        "used_code_irregularity_projection": str(out_dir / "var_vq_used_code_irregularity_on_images.png"),
    }
    _plot_codebook_pca(
        codebook=codebook,
        dims=codebook_dims,
        irregularity=codebook_irregularity,
        usage_counts=counts,
        out_path=Path(outputs["codebook_pca"]),
    )
    _plot_usage_histogram(counts=counts, out_path=Path(outputs["usage"]))
    _plot_patch_projection(
        images=images,
        final_ids=final_ids,
        code_values=codebook_dims,
        title="VAR VQ Codebook Local Dimension Projected to COCO Patches",
        colorbar_label="codebook local dimension",
        out_path=Path(outputs["codebook_dimension_projection"]),
        cmap="viridis",
    )
    _plot_patch_projection(
        images=images,
        final_ids=final_ids,
        code_values=codebook_irregularity,
        title="VAR VQ Codebook Fiber Violations Projected to COCO Patches",
        colorbar_label="codebook irregularity -log10(p)",
        out_path=Path(outputs["codebook_irregularity_projection"]),
        cmap="magma",
    )
    _plot_patch_projection(
        images=images,
        final_ids=final_ids,
        code_values=used_dim_lookup,
        title="Used VAR VQ Codes: Local Dimension Projected to COCO Patches",
        colorbar_label="used-code local dimension",
        out_path=Path(outputs["used_code_dimension_projection"]),
        cmap="viridis",
    )
    _plot_patch_projection(
        images=images,
        final_ids=final_ids,
        code_values=used_irr_lookup,
        title="Used VAR VQ Codes: Fiber Violations Projected to COCO Patches",
        colorbar_label="used-code irregularity -log10(p)",
        out_path=Path(outputs["used_code_irregularity_projection"]),
        cmap="magma",
    )

    report = {
        "args": vars(args),
        "patch_nums": list(patch_nums),
        "codebook": codebook_summary,
        "used_codes": used_summary,
        "usage": usage_summary,
        "outputs": outputs,
    }
    report_path = out_dir / "var_vq_probe_summary.json"
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(json.dumps({"summary": str(report_path), "codebook": codebook_summary, "used_codes": used_summary, "usage": usage_summary}, indent=2))


if __name__ == "__main__":
    main()
