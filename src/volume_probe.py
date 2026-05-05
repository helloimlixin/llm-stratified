#!/usr/bin/env python3
"""
No-training volume estimation for token, patch-embedding, and raw-patch spaces.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from torchvision.utils import make_grid
    from torchvision.transforms.functional import to_pil_image
except Exception:  # pragma: no cover
    make_grid = None
    to_pil_image = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    matplotlib = None
    plt = None

from datasets import create_data_loaders
from fiber.geometry import (
    normalize_volume_range,
    run_fiber_bundle_test_from_sorted_dists,
    sorted_distance_matrix,
    summarize_stratifications,
)
from models import DinoV2Wrapper, TinyViT, TimmViTWrapper, resolve_patch_size
from utils import denormalize_images, seed_everything, to_serializable


def _flatten_patches(imgs: torch.Tensor, patch_size: int, patch_stride: int | None = None) -> torch.Tensor:
    b, c, h, w = imgs.shape
    ps = int(patch_size)
    stride = int(patch_stride or patch_size)
    if min(h, w) < ps or stride <= 0:
        return torch.empty(0, c * ps * ps)
    patches = torch.nn.functional.unfold(imgs.contiguous(), kernel_size=ps, stride=stride)
    return patches.transpose(1, 2).reshape(b * patches.shape[-1], c * ps * ps)


def _resolve_prefix_tokens(model: torch.nn.Module) -> int:
    if hasattr(model, "num_prefix_tokens"):
        return int(getattr(model, "num_prefix_tokens"))
    return 2 if getattr(model, "has_dist_token", False) else 1


def _prepare_probe_inputs(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    dataset: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "prepare_images_for_features"):
        return model.prepare_images_for_features(imgs, dataset)
    return imgs, denormalize_images(imgs, dataset)


def _forward_feature_pack(model: torch.nn.Module, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
    if hasattr(model, "forward_feature_pack"):
        return model.forward_feature_pack(imgs)
    return {
        "tokens": model.forward_features(imgs),
        "patch_embeddings": _forward_patch_embed(model, imgs),
    }


def _flatten_token_tensor(tokens: torch.Tensor, prefix_tokens: int) -> torch.Tensor:
    start_idx = max(0, min(prefix_tokens, int(tokens.shape[1])))
    return tokens[:, start_idx:, :].reshape(-1, tokens.shape[-1])


def _representation_prefix_tokens(model: torch.nn.Module, rep_name: str) -> int:
    if rep_name == "tokens" or rep_name.startswith("tokens_"):
        return _resolve_prefix_tokens(model)
    if rep_name == "patch_embeddings" and bool(getattr(model, "patch_embeddings_include_prefix_tokens", False)):
        return _resolve_prefix_tokens(model)
    return 0


def _pixel_rep_name(patch_size: int, patch_stride: int) -> str:
    return "patch_pixels" if int(patch_stride) == int(patch_size) else f"patch_pixels_stride_{int(patch_stride)}"


def _stable_seed_offset(name: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(name))) % 100000


def _forward_patch_embed(model: torch.nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    pe = getattr(model, "patch_embed", None)
    if pe is None and hasattr(model, "backbone"):
        pe = getattr(model.backbone, "patch_embed", None)
    if pe is None:
        raise RuntimeError("Model has no patch_embed for patch embedding extraction.")
    out = pe(imgs)
    if isinstance(out, tuple):
        out = out[0]
    if out.dim() == 4:
        out = out.flatten(2).transpose(1, 2)
    return out


@torch.no_grad()
def collect_representations(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    dataset: str,
    patch_size: int,
    pixel_patch_stride: int | None,
    max_tokens: int,
    show_progress: bool,
    viz_images: int = 16,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    rep_buffers: dict[str, list[torch.Tensor]] = {}
    viz_imgs = []
    seen, warned = 0, False
    pixel_stride = int(pixel_patch_stride or patch_size)
    overlap_name = _pixel_rep_name(patch_size, pixel_stride)
    iterator = tqdm(loader, desc="Collect tokens", leave=False) if show_progress and tqdm else loader
    for batch in iterator:
        imgs = batch[0].to(device)
        model_imgs, denorm = _prepare_probe_inputs(model, imgs, dataset)
        feature_pack = _forward_feature_pack(model, model_imgs)
        flattened = {
            name: _flatten_token_tensor(feats, _representation_prefix_tokens(model, name)).detach().cpu()
            for name, feats in feature_pack.items()
        }
        patch_pix_flat = _flatten_patches(denorm, patch_size, patch_stride=patch_size).to(dtype=torch.float32).cpu()
        overlap_flat = _flatten_patches(denorm, patch_size, patch_stride=pixel_stride).to(dtype=torch.float32).cpu()

        max_viz = max(0, int(viz_images))
        if max_viz and len(viz_imgs) < max_viz:
            take = min(max_viz - len(viz_imgs), int(denorm.shape[0]))
            if take > 0:
                for i in range(take):
                    viz_imgs.append(denorm[i].detach().cpu())

        aligned_counts = [patch_pix_flat.shape[0]] + [rep.shape[0] for rep in flattened.values()]
        count = min(aligned_counts) if aligned_counts else 0
        if not warned and any(rep.shape[0] != count for rep in flattened.values()):
            print(
                "[warn] Patch counts mismatch across representations; truncating to common length "
                f"{count} ({', '.join(f'{name} {rep.shape[0]}' for name, rep in flattened.items())}, pixels {patch_pix_flat.shape[0]})."
            )
            warned = True

        for name, rep in flattened.items():
            rep_buffers.setdefault(name, []).append(rep[:count])
        rep_buffers.setdefault("patch_pixels", []).append(patch_pix_flat[:count])
        if overlap_name != "patch_pixels":
            rep_buffers.setdefault(overlap_name, []).append(overlap_flat)
        seen += count
        if seen >= max_tokens:
            break

    if not rep_buffers:
        return {}, torch.empty(0, 3, 1, 1)
    reps = {name: torch.cat(parts, dim=0)[:max_tokens] for name, parts in rep_buffers.items() if parts}
    return reps, torch.stack(viz_imgs, dim=0) if viz_imgs else torch.empty(0, 3, 1, 1)


def _run_volume_estimation(
    embeddings: torch.Tensor,
    *,
    vol_min: int,
    vol_max: int,
    ws: int,
    alpha: float,
    nstrat: int,
) -> tuple[dict, np.ndarray, list[dict], dict | None]:
    coords = embeddings.cpu().numpy().astype(np.float64)
    dists_sorted = sorted_distance_matrix(coords)
    npts = int(dists_sorted.shape[0])
    if npts < 2:
        return {"num_tokens": npts, "tokens_with_strata": 0}, np.array([], dtype=np.float64), [], None, None

    vol_min_adj, vol_max_adj = normalize_volume_range(npts, vol_min, vol_max)
    results = run_fiber_bundle_test_from_sorted_dists(
        dists_sorted,
        vol_min=vol_min_adj,
        vol_max=vol_max_adj,
        ws=ws,
        alpha=alpha,
        nstrat=nstrat,
    )
    summary = summarize_stratifications(results, alpha=alpha)
    dims = np.array(
        [res["dimensions"][0] if res and res.get("dimensions") else np.nan for res in results],
        dtype=np.float64,
    )
    knn_curve = None
    if vol_max_adj > vol_min_adj:
        radii_mat = dists_sorted[vol_min_adj:vol_max_adj, :]  # (k, npts)
        qs = np.array([0.1, 0.5, 0.9], dtype=np.float64)
        qvals = np.quantile(radii_mat, qs, axis=1)  # (q, k)
        knn_curve = {
            "k_min": int(vol_min_adj),
            "k_max": int(vol_max_adj - 1),
            "k_values": list(range(int(vol_min_adj), int(vol_max_adj))),
            "quantiles": qs.tolist(),
            "radii": {f"q{int(q * 100)}": qvals[i].astype(float).tolist() for i, q in enumerate(qs.tolist())},
        }
    return summary, dims, results, knn_curve, dists_sorted


def _result_min_pvalues(results: list[dict]) -> np.ndarray:
    values = np.full(len(results), np.nan, dtype=np.float64)
    for idx, res in enumerate(results):
        if not res or not res.get("pvalues"):
            continue
        finite = np.asarray(res["pvalues"], dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            values[idx] = float(np.min(finite))
    return values


def _result_irregularity(min_pvalues: np.ndarray) -> np.ndarray:
    out = np.full(min_pvalues.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(min_pvalues)
    out[finite] = -np.log10(np.clip(min_pvalues[finite], 1e-12, None))
    return out


def _select_visual_anchor_indices(scores: np.ndarray, *, limit: int) -> np.ndarray:
    if limit <= 0:
        return np.empty(0, dtype=np.int64)
    finite = np.flatnonzero(np.isfinite(scores))
    if finite.size == 0:
        return np.empty(0, dtype=np.int64)

    picks: list[int] = []
    high = finite[np.argsort(-scores[finite])]
    low = finite[np.argsort(scores[finite])]
    median = float(np.median(scores[finite]))
    middle = finite[np.argsort(np.abs(scores[finite] - median))]

    for candidate_set in (high, middle, low):
        for idx in candidate_set.tolist():
            if idx not in picks:
                picks.append(int(idx))
            if len(picks) >= limit:
                return np.asarray(picks, dtype=np.int64)
    return np.asarray(picks, dtype=np.int64)


def _project_embeddings_2d(
    embeddings: torch.Tensor,
    *,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(embeddings.shape[0])
    if n <= 0:
        return np.empty(0, dtype=np.int64), np.zeros((0, 2), dtype=np.float32)

    if 0 < max_points < n:
        rng = np.random.default_rng(int(seed))
        sample_idx = np.sort(rng.choice(n, size=max_points, replace=False).astype(np.int64))
    else:
        sample_idx = np.arange(n, dtype=np.int64)

    sample = embeddings[sample_idx].detach().float().cpu()
    if sample.numel() == 0:
        return sample_idx, np.zeros((0, 2), dtype=np.float32)

    centered = sample - sample.mean(dim=0, keepdim=True)
    rank = min(2, int(centered.shape[0]), int(centered.shape[1]))
    if rank > 0:
        _u, _s, v = torch.pca_lowrank(centered, q=rank)
        coords = centered @ v[:, :rank]
    else:
        coords = torch.zeros((sample.shape[0], 0), dtype=sample.dtype)

    if coords.shape[1] < 2:
        coords = torch.cat(
            [coords, torch.zeros((coords.shape[0], 2 - coords.shape[1]), dtype=coords.dtype)],
            dim=1,
        )
    return sample_idx, coords[:, :2].numpy()


def _scatter_metric(ax, coords: np.ndarray, values: np.ndarray, *, title: str, cmap: str, colorbar_label: str) -> None:
    finite = np.isfinite(values)
    if np.any(~finite):
        ax.scatter(
            coords[~finite, 0],
            coords[~finite, 1],
            s=8,
            c="#d0d0d0",
            alpha=0.5,
            linewidths=0,
        )
    if np.any(finite):
        scatter = ax.scatter(
            coords[finite, 0],
            coords[finite, 1],
            s=10,
            c=values[finite],
            cmap=cmap,
            alpha=0.85,
            linewidths=0,
        )
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.20, linewidth=0.5)


def _save_patch_nn_grid(
    *,
    rep: torch.Tensor,
    display_source: torch.Tensor,
    rep_name: str,
    out_path: Path,
    patch_size: int,
    seed: int,
    anchors: int,
    nn_k: int,
    anchor_indices: np.ndarray | None = None,
) -> bool:
    if make_grid is None or to_pil_image is None:
        return False
    n = int(rep.shape[0])
    if n <= 1 or anchors <= 0 or nn_k <= 0:
        return False
    ps = int(patch_size)
    c = int(display_source.shape[1] // max(1, ps * ps))
    if c <= 0 or display_source.shape[1] != c * ps * ps:
        return False

    if anchor_indices is None:
        rng = np.random.default_rng(int(seed) + _stable_seed_offset(rep_name))
        take = min(int(anchors), n)
        anchor_idx = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
    else:
        anchor_idx = np.asarray(anchor_indices, dtype=np.int64)
        anchor_idx = anchor_idx[np.logical_and(anchor_idx >= 0, anchor_idx < n)]
        if anchor_idx.size == 0:
            return False
        anchor_idx = anchor_idx[: min(int(anchors), anchor_idx.size)]

    nn_k_eff = min(int(nn_k), n - 1)
    rep_f = rep.to(dtype=torch.float32)
    patches = display_source.to(dtype=torch.float32)
    rows = []
    for anchor in anchor_idx.tolist():
        d = ((rep_f - rep_f[int(anchor)]) ** 2).sum(dim=1)
        nn = torch.topk(d, k=nn_k_eff + 1, largest=False).indices
        rows.append(patches[nn].reshape(-1, c, ps, ps).clamp(0, 1))
    if not rows:
        return False

    grid = make_grid(torch.cat(rows, dim=0), nrow=nn_k_eff + 1, padding=2)
    to_pil_image(grid).save(out_path)
    return True


def _save_representation_dashboard(
    *,
    embeddings: torch.Tensor,
    dims: np.ndarray,
    min_pvalues: np.ndarray,
    irregularity: np.ndarray,
    summary: dict,
    alpha: float,
    out_path: Path,
    seed: int,
    max_points: int,
) -> bool:
    if plt is None:
        return False

    finite_dims = dims[np.isfinite(dims)]
    finite_irregularity = irregularity[np.isfinite(irregularity)]
    if embeddings.shape[0] <= 0 or (finite_dims.size == 0 and finite_irregularity.size == 0):
        return False

    sample_idx, coords = _project_embeddings_2d(
        embeddings,
        max_points=max_points,
        seed=int(seed),
    )
    if coords.shape[0] == 0:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_dim_hist, ax_irr_hist, ax_irr_scatter, ax_dim_scatter = axes.flatten()

    if finite_dims.size:
        ax_dim_hist.hist(finite_dims, bins=min(30, max(8, int(np.sqrt(finite_dims.size)))), color="#2f5d8a", alpha=0.9)
    ax_dim_hist.set_title("Local Dimension Distribution")
    ax_dim_hist.set_xlabel("estimated dimension")
    ax_dim_hist.set_ylabel("token / patch count")
    ax_dim_hist.grid(alpha=0.20, linewidth=0.5)

    if finite_irregularity.size:
        ax_irr_hist.hist(
            finite_irregularity,
            bins=min(30, max(8, int(np.sqrt(finite_irregularity.size)))),
            color="#a63d40",
            alpha=0.9,
        )
    rejected = int(np.sum(np.isfinite(min_pvalues) & (min_pvalues < float(alpha))))
    total = int(np.sum(np.isfinite(min_pvalues)))
    ax_irr_hist.set_title(f"Irregularity Distribution ({rejected}/{total} rejected)")
    ax_irr_hist.set_xlabel("-log10(min p-value)")
    ax_irr_hist.set_ylabel("token / patch count")
    ax_irr_hist.grid(alpha=0.20, linewidth=0.5)

    _scatter_metric(
        ax_irr_scatter,
        coords,
        irregularity[sample_idx],
        title="PCA Projection Colored by Fiber-Test Irregularity",
        cmap="magma",
        colorbar_label="-log10(min p-value)",
    )
    _scatter_metric(
        ax_dim_scatter,
        coords,
        dims[sample_idx],
        title="PCA Projection Colored by Local Dimension",
        cmap="viridis",
        colorbar_label="estimated local dimension",
    )

    fig.suptitle(
        "Volume Probe Dashboard: mean_dim="
        f"{summary.get('mean_dim', float('nan')):.2f}, "
        "irregular_ratio="
        f"{summary.get('irregular_ratio', float('nan')):.3f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def _save_scaling_curves_figure(
    *,
    dists_sorted: np.ndarray | None,
    results: list[dict],
    irregularity: np.ndarray,
    out_path: Path,
    seed: int,
    vol_min: int,
    vol_max: int,
    anchors: int,
) -> bool:
    if plt is None or dists_sorted is None or anchors <= 0:
        return False

    anchor_idx = _select_visual_anchor_indices(irregularity, limit=anchors)
    if anchor_idx.size == 0:
        return False

    k_values = np.arange(int(vol_min), int(vol_max), dtype=np.int64)
    if k_values.size == 0:
        return False

    cols = min(2, int(anchor_idx.size))
    rows = int(math.ceil(anchor_idx.size / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)

    for ax, idx in zip(axes.flatten(), anchor_idx.tolist()):
        radii = np.clip(np.asarray(dists_sorted[k_values, int(idx)], dtype=np.float64), 1e-12, None)
        ax.plot(np.log10(radii), np.log10(k_values.astype(np.float64)), color="#2f5d8a", linewidth=2)
        res = results[int(idx)] if int(idx) < len(results) else {}
        for strat_radius in np.asarray((res or {}).get("strat_radii") or [], dtype=np.float64):
            if np.isfinite(strat_radius) and strat_radius > 0:
                ax.axvline(np.log10(strat_radius), color="#a63d40", linestyle="--", linewidth=1)
        dims = [float(val) for val in (res or {}).get("dimensions") or [] if np.isfinite(float(val))]
        min_p = float(np.nanmin(np.asarray((res or {}).get("pvalues") or [np.nan], dtype=np.float64)))
        title = f"anchor {idx} | irr {irregularity[int(idx)]:.2f}"
        if dims:
            title += " | dims " + ",".join(f"{dim:.1f}" for dim in dims[:3])
        if np.isfinite(min_p):
            title += f" | p={min_p:.2e}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("log10(radius)")
        ax.set_ylabel("log10(neighbor count k)")
        ax.grid(alpha=0.25)

    for ax in axes.flatten()[anchor_idx.size :]:
        ax.axis("off")

    fig.suptitle("Local Volume Scaling Curves: Slope Estimates Local Dimension", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def run_volume_probe(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    dataset: str,
    patch_size: int,
    pixel_patch_stride: int | None,
    max_tokens: int,
    vol_min: int,
    vol_max: int,
    ws: int,
    alpha: float,
    nstrat: int,
    out_dir: Path,
    save_full: bool = False,
    progress: bool = False,
    seed: int = 1337,
    viz_images: int = 16,
    viz_patches: int = 64,
    viz_nn_anchors: int = 3,
    viz_nn_k: int = 8,
    viz_projection_points: int = 1024,
    viz_curve_anchors: int = 6,
    config: dict | None = None,
) -> dict:
    representations, example_images = collect_representations(
        model=model,
        loader=loader,
        device=device,
        dataset=dataset,
        patch_size=patch_size,
        pixel_patch_stride=pixel_patch_stride,
        max_tokens=max_tokens,
        show_progress=progress,
        viz_images=viz_images,
    )
    patch_pixels = representations.get("patch_pixels", torch.empty(0, 1))
    overlap_pixel_names = [name for name in representations if name.startswith("patch_pixels_stride_")]

    out_dir.mkdir(parents=True, exist_ok=True)
    results_payload = {
        "config": config or {},
        "representations": {},
    }

    # Lightweight visualizations (saved to disk; can be logged to W&B by the caller).
    viz_files: dict[str, str] = {}
    if make_grid is not None and to_pil_image is not None:
        try:
            if isinstance(example_images, torch.Tensor) and example_images.numel() > 0:
                imgs = example_images[: max(0, int(viz_images))].clamp(0, 1)
                if imgs.numel() > 0:
                    grid = make_grid(imgs, nrow=min(4, int(imgs.shape[0])), padding=2)
                    out_path = out_dir / "example_images.png"
                    to_pil_image(grid).save(out_path)
                    viz_files["example_images"] = out_path.name
        except Exception as e:
            print(f"[viz] failed to save example_images: {e}")

        try:
            ps = int(patch_size)
            for rep_name in ["patch_pixels"] + overlap_pixel_names:
                rep = representations.get(rep_name)
                if rep is None:
                    continue
                n = int(rep.shape[0])
                if viz_patches <= 0 or n <= 0 or ps <= 0:
                    continue
                c = int(rep.shape[1] // max(1, ps * ps))
                if c <= 0 or rep.shape[1] != c * ps * ps:
                    continue
                rng = np.random.default_rng(int(seed) + _stable_seed_offset(rep_name))
                take = min(int(viz_patches), n)
                idx = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
                patch_imgs = rep[idx].to(dtype=torch.float32).reshape(take, c, ps, ps).clamp(0, 1)
                grid = make_grid(patch_imgs, nrow=min(8, take), padding=2)
                out_path = out_dir / f"example_{rep_name}.png"
                to_pil_image(grid).save(out_path)
                viz_files[f"example_{rep_name}"] = out_path.name
        except Exception as e:
            print(f"[viz] failed to save example_patches: {e}")

        try:
            for rep_name, rep in representations.items():
                display_source = rep if rep_name.startswith("patch_pixels_stride_") else patch_pixels
                out_path = out_dir / f"nn_{rep_name}.png"
                if _save_patch_nn_grid(
                    rep=rep,
                    display_source=display_source,
                    rep_name=rep_name,
                    out_path=out_path,
                    patch_size=patch_size,
                    seed=seed,
                    anchors=viz_nn_anchors,
                    nn_k=viz_nn_k,
                ):
                    viz_files[f"nn_{rep_name}"] = out_path.name
        except Exception as e:
            print(f"[viz] failed to save nn grids: {e}")

    for name, emb in representations.items():
        summary, dims, results, knn_curve, dists_sorted = _run_volume_estimation(
            emb,
            vol_min=vol_min,
            vol_max=vol_max,
            ws=ws,
            alpha=alpha,
            nstrat=nstrat,
        )
        dims_path = out_dir / f"{name}_dims.npy"
        np.save(dims_path, dims)
        results_path = None
        if save_full:
            results_path = out_dir / f"{name}_fiber_results.json"
            with open(results_path, "w") as fp:
                json.dump(to_serializable(results), fp, indent=2)

        min_pvalues = _result_min_pvalues(results)
        irregularity = _result_irregularity(min_pvalues)
        rep_viz: dict[str, str] = {}
        try:
            detail_path = out_dir / f"detail_{name}.png"
            if _save_representation_dashboard(
                embeddings=emb,
                dims=dims,
                min_pvalues=min_pvalues,
                irregularity=irregularity,
                summary=summary,
                alpha=alpha,
                out_path=detail_path,
                seed=int(seed) + _stable_seed_offset(name),
                max_points=viz_projection_points,
            ):
                rep_viz["detail"] = detail_path.name
                viz_files[f"detail_{name}"] = detail_path.name

            k_min = int(knn_curve["k_min"]) if isinstance(knn_curve, dict) and knn_curve.get("k_min") is not None else int(vol_min)
            k_max_exclusive = (
                int(knn_curve["k_max"]) + 1
                if isinstance(knn_curve, dict) and knn_curve.get("k_max") is not None
                else int(vol_max)
            )
            scaling_path = out_dir / f"scaling_{name}.png"
            if _save_scaling_curves_figure(
                dists_sorted=dists_sorted,
                results=results,
                irregularity=irregularity,
                out_path=scaling_path,
                seed=int(seed) + _stable_seed_offset(name),
                vol_min=k_min,
                vol_max=k_max_exclusive,
                anchors=viz_curve_anchors,
            ):
                rep_viz["scaling"] = scaling_path.name
                viz_files[f"scaling_{name}"] = scaling_path.name

            display_source = emb if name.startswith("patch_pixels_stride_") else patch_pixels
            irregular_idx = _select_visual_anchor_indices(irregularity, limit=viz_nn_anchors)
            nn_irregular_path = out_dir / f"nn_irregular_{name}.png"
            if _save_patch_nn_grid(
                rep=emb,
                display_source=display_source,
                rep_name=f"{name}_irregular",
                out_path=nn_irregular_path,
                patch_size=patch_size,
                seed=int(seed) + _stable_seed_offset(name),
                anchors=viz_nn_anchors,
                nn_k=viz_nn_k,
                anchor_indices=irregular_idx,
            ):
                rep_viz["nn_irregular"] = nn_irregular_path.name
                viz_files[f"nn_irregular_{name}"] = nn_irregular_path.name
        except Exception as e:
            print(f"[viz] failed to save detailed visuals for {name}: {e}")

        results_payload["representations"][name] = {
            "summary": summary,
            "knn_curve": knn_curve,
            "dims_path": dims_path.name,
            "results_path": results_path.name if results_path else None,
            "viz": rep_viz,
        }

    if viz_files:
        results_payload["viz"] = viz_files

    with open(out_dir / "volume_summary.json", "w") as fp:
        json.dump(to_serializable(results_payload), fp, indent=2)
    return results_payload


def _build_model(
    *, args: argparse.Namespace, num_classes: int, in_chans: int, img_size: int
) -> tuple[torch.nn.Module, int]:
    if args.feature_backbone == "dinov2":
        model = DinoV2Wrapper(model_name=args.dinov2_model, token_layers=args.dinov2_layers)
        return model, int(model.patch_size)
    if args.timm_model:
        model = TimmViTWrapper(args.timm_model, num_classes, pretrained=args.timm_pretrained)
        patch_size = resolve_patch_size(model) or args.patch_size
    else:
        if args.patch_size is None:
            raise ValueError("--patch-size is required for TinyViT.")
        model = TinyViT(
            img_size,
            args.patch_size,
            in_chans,
            num_classes,
            args.embed_dim,
            args.depth,
            args.num_heads,
            args.mlp_ratio,
            args.dropout,
        )
        patch_size = args.patch_size

    if patch_size is None:
        raise ValueError("Unable to resolve patch size; pass --patch-size explicitly.")
    if args.patch_size and patch_size != args.patch_size and args.timm_model:
        print(f"[warn] Using timm patch size {patch_size} (overriding --patch-size {args.patch_size}).")
    return model, int(patch_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-training volume estimation baselines.")
    parser.add_argument("--dataset", default="CIFAR10")
    parser.add_argument("--root", default="./data")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subset-test", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--vol-min", type=int, default=8)
    parser.add_argument("--vol-max", type=int, default=64)
    parser.add_argument("--ws", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=5e-3)
    parser.add_argument("--nstrat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--outdir", default="./runs/volume_probe")
    parser.add_argument("--timm-model", default=None)
    parser.add_argument("--timm-pretrained", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-full", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--viz-images", type=int, default=16)
    parser.add_argument("--viz-patches", type=int, default=64)
    parser.add_argument("--viz-nn-anchors", type=int, default=3)
    parser.add_argument("--viz-nn-k", type=int, default=8)
    parser.add_argument("--viz-projection-points", type=int, default=1024)
    parser.add_argument("--viz-curve-anchors", type=int, default=6)
    parser.add_argument("--feature-backbone", choices=["model", "dinov2"], default="model")
    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    parser.add_argument("--dinov2-layers", type=int, nargs="*", default=None)
    parser.add_argument("--pixel-patch-stride", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive.")
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    _, test_loader, num_classes, in_chans, final_img_size, _task = create_data_loaders(
        args.dataset,
        args.root,
        args.img_size,
        args.batch_size,
        args.batch_size,
        args.num_workers,
        None,
        args.subset_test,
        device,
        distributed=False,
        rank=0,
        world_size=1,
    )

    model, patch_size = _build_model(args=args, num_classes=num_classes, in_chans=in_chans, img_size=final_img_size)
    model = model.to(device)
    model.eval()

    config = {
        "dataset": args.dataset,
        "root": args.root,
        "img_size": final_img_size,
        "patch_size": patch_size,
        "max_tokens": args.max_tokens,
        "vol_min": args.vol_min,
        "vol_max": args.vol_max,
        "ws": args.ws,
        "alpha": args.alpha,
        "nstrat": args.nstrat,
        "seed": args.seed,
        "device": str(device),
        "feature_backbone": args.feature_backbone,
        "timm_model": args.timm_model,
        "timm_pretrained": bool(args.timm_pretrained),
        "dinov2_model": args.dinov2_model,
        "dinov2_layers": list(args.dinov2_layers) if args.dinov2_layers is not None else None,
        "pixel_patch_stride": args.pixel_patch_stride,
        "viz_projection_points": args.viz_projection_points,
        "viz_curve_anchors": args.viz_curve_anchors,
    }
    out_dir = Path(args.outdir)
    run_volume_probe(
        model=model,
        loader=test_loader,
        device=device,
        dataset=args.dataset,
        patch_size=patch_size,
        pixel_patch_stride=args.pixel_patch_stride,
        max_tokens=args.max_tokens,
        vol_min=args.vol_min,
        vol_max=args.vol_max,
        ws=args.ws,
        alpha=args.alpha,
        nstrat=args.nstrat,
        out_dir=out_dir,
        save_full=args.save_full,
        progress=args.progress,
        seed=args.seed,
        viz_images=args.viz_images,
        viz_patches=args.viz_patches,
        viz_nn_anchors=args.viz_nn_anchors,
        viz_nn_k=args.viz_nn_k,
        viz_projection_points=args.viz_projection_points,
        viz_curve_anchors=args.viz_curve_anchors,
        config=config,
    )
    print(f"Saved volume estimates -> {out_dir}")


if __name__ == "__main__":
    main()
