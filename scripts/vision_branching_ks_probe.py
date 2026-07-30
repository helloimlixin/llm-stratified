"""Probe visual branch flattening with robust KS diagnostics and W&B logging.

The main model-facing path consumes saved ``embeddings/epoch_000.pt`` artifacts
from the existing fiber runs.  A no-Torch image-folder backend is also provided
for smoke tests and quick visual checks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fiber.branching_ks import (  # noqa: E402
    augmentation_branch_instability,
    branch_metrics,
    branch_posteriors,
    extract_image_folder_patch_features,
    fiber_singularity_scores,
    fit_kmeans,
    ks_2samp,
    quantile_group_mask,
    sliced_ks_test,
    standardize_features,
)


PLOT_BG = (255, 255, 255)
INK = (28, 32, 38)
MUTED = (104, 113, 128)
BLUE = (47, 99, 164)
RED = (204, 80, 72)
GREEN = (58, 142, 95)
ORANGE = (216, 132, 48)


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_serializable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local ML env
        raise RuntimeError(
            "Loading saved embedding artifacts requires torch. Use --image-dir "
            "for the dependency-light backend, or run this script in the ML env."
        ) from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _load_fiber_results(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of fiber results in {path}")
    return data


def _artifact_images_to_pil(artifact: dict[str, Any], *, dataset: str | None) -> list[Image.Image] | None:
    images = artifact.get("images")
    if images is None:
        return None
    try:
        import torch

        tensor = images.detach().float().cpu() if hasattr(images, "detach") else torch.as_tensor(images).float()
        if dataset:
            try:
                from utils import denormalize_images

                tensor = denormalize_images(tensor, dataset)
            except Exception:
                tensor = tensor.clamp(0, 1)
        else:
            tensor = tensor.clamp(0, 1)
        out: list[Image.Image] = []
        for img in tensor:
            arr = (img.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
            out.append(Image.fromarray(arr, mode="RGB"))
        return out
    except Exception:
        return None


def _load_artifact_backend(args: argparse.Namespace) -> dict[str, Any]:
    artifact = _torch_load(Path(args.embeddings))
    features = _tensor_to_numpy(artifact["embeddings"]).astype(np.float64)
    images = _artifact_images_to_pil(artifact, dataset=args.dataset)
    image_ids = _tensor_to_numpy(artifact["image_ids"]).astype(np.int64)
    bboxes = _tensor_to_numpy(artifact["bboxes"]).astype(np.int64)
    patch_indices = np.arange(features.shape[0], dtype=np.int64)
    if "patch_indices" in artifact:
        patch_indices = _tensor_to_numpy(artifact["patch_indices"]).astype(np.int64)
    fiber_results = _load_fiber_results(args.fiber_results)
    return {
        "backend": "torch_artifact",
        "features": features,
        "all_variant_features": None,
        "variant_groups": None,
        "images": images,
        "image_ids": image_ids,
        "bboxes": bboxes,
        "patch_indices": patch_indices,
        "image_names": [f"image_{int(v)}" for v in image_ids],
        "fiber_results": fiber_results,
    }


def _load_image_backend(args: argparse.Namespace) -> dict[str, Any]:
    loaded = extract_image_folder_patch_features(
        args.image_dir,
        image_size=args.image_size,
        grid=args.grid,
        augmentations=args.augmentations,
        seed=args.seed,
    )
    loaded["backend"] = "image_folder"
    loaded["fiber_results"] = None
    return loaded


def _font(size: int = 14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_title(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, *, size: int = 18) -> None:
    draw.text(xy, text, fill=INK, font=_font(size))


def _scale(values: np.ndarray, lo: float, hi: float, pix_lo: int, pix_hi: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if abs(hi - lo) < 1e-12:
        return np.full_like(values, (pix_lo + pix_hi) / 2.0)
    return pix_lo + (values - lo) * (pix_hi - pix_lo) / (hi - lo)


def _save_cdf_plot(path: Path, regular: np.ndarray, singular: np.ndarray, *, metric: str, ks_stat: float) -> Path:
    width, height = 980, 620
    margin_l, margin_r, margin_t, margin_b = 90, 42, 82, 82
    img = Image.new("RGB", (width, height), PLOT_BG)
    draw = ImageDraw.Draw(img)
    _draw_title(draw, (margin_l, 28), f"{metric}: singular vs regular CDF", size=24)
    draw.text((margin_l, 58), f"KS D = {ks_stat:.3f}", fill=MUTED, font=_font(15))

    x0, y0 = margin_l, height - margin_b
    x1, y1 = width - margin_r, margin_t
    draw.line((x0, y0, x1, y0), fill=INK, width=2)
    draw.line((x0, y0, x0, y1), fill=INK, width=2)
    values = np.concatenate([regular[np.isfinite(regular)], singular[np.isfinite(singular)]])
    lo = float(np.nanmin(values)) if values.size else 0.0
    hi = float(np.nanmax(values)) if values.size else 1.0
    if abs(hi - lo) < 1e-9:
        hi = lo + 1.0

    def cdf_points(values_in: np.ndarray) -> list[tuple[int, int]]:
        vals = np.sort(values_in[np.isfinite(values_in)])
        if vals.size == 0:
            return []
        xs = _scale(vals, lo, hi, x0, x1)
        ys = _scale(np.arange(1, vals.size + 1) / vals.size, 0.0, 1.0, y0, y1)
        return [(int(x), int(y)) for x, y in zip(xs, ys)]

    for pts, color in ((cdf_points(regular), BLUE), (cdf_points(singular), RED)):
        if len(pts) > 1:
            draw.line(pts, fill=color, width=4)
    draw.text((x0, y0 + 18), f"{lo:.3g}", fill=MUTED, font=_font(13))
    draw.text((x1 - 80, y0 + 18), f"{hi:.3g}", fill=MUTED, font=_font(13))
    draw.text((x0 - 42, y1 - 8), "1.0", fill=MUTED, font=_font(13))
    draw.text((x0 - 42, y0 - 8), "0.0", fill=MUTED, font=_font(13))
    draw.rectangle((x1 - 240, y1 + 12, x1 - 20, y1 + 82), outline=(226, 230, 236), width=1)
    draw.line((x1 - 220, y1 + 34, x1 - 174, y1 + 34), fill=BLUE, width=4)
    draw.text((x1 - 164, y1 + 25), "regular", fill=INK, font=_font(15))
    draw.line((x1 - 220, y1 + 61, x1 - 174, y1 + 61), fill=RED, width=4)
    draw.text((x1 - 164, y1 + 52), "singular", fill=INK, font=_font(15))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def _save_scatter_plot(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    singular: np.ndarray,
    *,
    x_label: str,
    y_label: str,
    title: str,
) -> Path:
    width, height = 980, 660
    margin_l, margin_r, margin_t, margin_b = 92, 44, 84, 88
    img = Image.new("RGB", (width, height), PLOT_BG)
    draw = ImageDraw.Draw(img)
    _draw_title(draw, (margin_l, 28), title, size=24)
    finite = np.isfinite(x) & np.isfinite(y)
    x_f, y_f, s_f = x[finite], y[finite], singular[finite]
    if x_f.size == 0:
        img.save(path)
        return path
    xlo, xhi = float(np.min(x_f)), float(np.max(x_f))
    ylo, yhi = float(np.min(y_f)), float(np.max(y_f))
    if abs(xhi - xlo) < 1e-9:
        xhi = xlo + 1.0
    if abs(yhi - ylo) < 1e-9:
        yhi = ylo + 1.0
    x0, y0 = margin_l, height - margin_b
    x1, y1 = width - margin_r, margin_t
    draw.line((x0, y0, x1, y0), fill=INK, width=2)
    draw.line((x0, y0, x0, y1), fill=INK, width=2)
    xs = _scale(x_f, xlo, xhi, x0, x1)
    ys = _scale(y_f, ylo, yhi, y0, y1)
    for px, py, is_singular in zip(xs, ys, s_f):
        color = RED if is_singular else BLUE
        r = 5 if is_singular else 3
        draw.ellipse((int(px) - r, int(py) - r, int(px) + r, int(py) + r), fill=color)
    draw.text((x0, y0 + 28), x_label, fill=INK, font=_font(16))
    draw.text((x0 - 76, y1 - 28), y_label, fill=INK, font=_font(16))
    draw.text((x0, y0 + 8), f"{xlo:.3g}", fill=MUTED, font=_font(13))
    draw.text((x1 - 80, y0 + 8), f"{xhi:.3g}", fill=MUTED, font=_font(13))
    draw.text((x0 - 56, y0 - 8), f"{ylo:.3g}", fill=MUTED, font=_font(13))
    draw.text((x0 - 56, y1 - 8), f"{yhi:.3g}", fill=MUTED, font=_font(13))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def _save_hist_plot(path: Path, values: np.ndarray, null: np.ndarray, *, observed: float, title: str) -> Path:
    width, height = 900, 540
    margin_l, margin_r, margin_t, margin_b = 80, 44, 82, 72
    img = Image.new("RGB", (width, height), PLOT_BG)
    draw = ImageDraw.Draw(img)
    _draw_title(draw, (margin_l, 28), title, size=23)
    finite = values[np.isfinite(values)]
    null_f = null[np.isfinite(null)]
    all_vals = np.concatenate([finite, null_f, np.asarray([observed])])
    lo = float(np.min(all_vals)) if all_vals.size else 0.0
    hi = float(np.max(all_vals)) if all_vals.size else 1.0
    if abs(hi - lo) < 1e-9:
        hi = lo + 1.0
    bins = np.linspace(lo, hi, 22)
    x0, y0 = margin_l, height - margin_b
    x1, y1 = width - margin_r, margin_t
    draw.line((x0, y0, x1, y0), fill=INK, width=2)
    draw.line((x0, y0, x0, y1), fill=INK, width=2)
    for vals, color, alpha_scale in ((null_f, (196, 205, 219), 1.0), (finite, BLUE, 1.0)):
        if vals.size == 0:
            continue
        hist, _ = np.histogram(vals, bins=bins)
        max_h = max(1, int(hist.max()))
        for idx, count in enumerate(hist):
            bx0 = int(_scale(np.asarray([bins[idx]]), lo, hi, x0, x1)[0])
            bx1 = int(_scale(np.asarray([bins[idx + 1]]), lo, hi, x0, x1)[0])
            by = int(y0 - (count / max_h) * (y0 - y1) * 0.82)
            draw.rectangle((bx0 + 1, by, max(bx0 + 2, bx1 - 1), y0), fill=color)
    obs_x = int(_scale(np.asarray([observed]), lo, hi, x0, x1)[0])
    draw.line((obs_x, y0, obs_x, y1), fill=RED, width=4)
    draw.text((obs_x + 8, y1 + 8), "observed median", fill=RED, font=_font(14))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def _blend(color: tuple[int, int, int], value: float) -> tuple[int, int, int]:
    v = max(0.0, min(1.0, float(value)))
    return (
        int((1.0 - v) * 35 + v * color[0]),
        int((1.0 - v) * 42 + v * color[1]),
        int((1.0 - v) * 54 + v * color[2]),
    )


def _save_heatmap_grid(
    path: Path,
    images: list[Image.Image] | None,
    image_ids: np.ndarray,
    bboxes: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    max_images: int,
) -> Path | None:
    if not images:
        return None
    finite = np.isfinite(values)
    if not np.any(finite):
        return None
    unique, counts = np.unique(image_ids[finite], return_counts=True)
    ranked = unique[np.argsort(-counts)][: max(1, int(max_images))]
    cols = min(4, len(ranked))
    rows = int(math.ceil(len(ranked) / cols))
    panel = 280
    title_h = 72
    img = Image.new("RGB", (cols * panel, rows * panel + title_h), PLOT_BG)
    draw = ImageDraw.Draw(img)
    _draw_title(draw, (18, 18), title, size=24)
    lo = float(np.nanpercentile(values[finite], 5))
    hi = float(np.nanpercentile(values[finite], 95))
    if abs(hi - lo) < 1e-9:
        hi = lo + 1.0
    for idx, image_idx in enumerate(ranked):
        row, col = divmod(idx, cols)
        x_off, y_off = col * panel, title_h + row * panel
        base = images[int(image_idx)].convert("RGB").resize((panel, panel))
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        token_idxs = np.where((image_ids == int(image_idx)) & finite)[0]
        for token_idx in token_idxs:
            x0, y0, x1, y1 = bboxes[token_idx].astype(float)
            sx, sy = panel / max(1.0, images[int(image_idx)].width), panel / max(1.0, images[int(image_idx)].height)
            norm = (float(values[token_idx]) - lo) / max(hi - lo, 1e-12)
            color = _blend(ORANGE, norm)
            od.rectangle((x0 * sx, y0 * sy, x1 * sx, y1 * sy), fill=(*color, 112), outline=(*color, 210))
        base = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
        img.paste(base, (x_off, y_off))
        draw.text((x_off + 8, y_off + 8), f"image {int(image_idx)}", fill=INK, font=_font(14))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def _save_patch_gallery(
    path: Path,
    images: list[Image.Image] | None,
    image_ids: np.ndarray,
    bboxes: np.ndarray,
    scores: np.ndarray,
    *,
    title: str,
    top_k: int = 16,
) -> Path | None:
    if not images:
        return None
    finite = np.where(np.isfinite(scores))[0]
    if finite.size == 0:
        return None
    ranked = finite[np.argsort(-scores[finite])[: max(1, int(top_k))]]
    cell = 154
    cols = min(4, len(ranked))
    rows = int(math.ceil(len(ranked) / cols))
    title_h = 76
    img = Image.new("RGB", (cols * cell, rows * (cell + 34) + title_h), PLOT_BG)
    draw = ImageDraw.Draw(img)
    _draw_title(draw, (18, 18), title, size=23)
    for pos, token_idx in enumerate(ranked):
        row, col = divmod(pos, cols)
        x_off, y_off = col * cell, title_h + row * (cell + 34)
        image_idx = int(image_ids[token_idx])
        if image_idx < 0 or image_idx >= len(images):
            continue
        x0, y0, x1, y1 = [int(v) for v in bboxes[token_idx]]
        patch = images[image_idx].crop((x0, y0, x1, y1)).resize((cell - 16, cell - 16))
        img.paste(patch, (x_off + 8, y_off + 8))
        draw.text((x_off + 8, y_off + cell - 4), f"tok {int(token_idx)}  s={scores[token_idx]:.2f}", fill=INK, font=_font(13))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return float("nan")
    xx, yy = x[mask], y[mask]
    if np.std(xx) <= 1e-12 or np.std(yy) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def _import_real_wandb() -> Any:
    cached = sys.modules.pop("wandb", None)
    original_path = list(sys.path)
    repo_text = str(REPO_ROOT.resolve())
    try:
        sys.path = [
            item for item in sys.path
            if item not in {"", "."} and str(Path(item).resolve()) != repo_text
        ]
        import importlib

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


def _init_wandb(args: argparse.Namespace, config: dict[str, Any]):
    if not args.wandb:
        return None
    try:
        wandb = _import_real_wandb()
    except Exception as exc:
        print(f"[wandb] unavailable: {exc}")
        return None
    init_kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_name,
        "tags": [tag for tag in args.wandb_tags.split(",") if tag],
        "config": config,
        "dir": str(args.out_dir / "wandb"),
    }
    mode = args.wandb_mode
    if mode:
        init_kwargs["mode"] = mode
    try:
        return wandb.init(**init_kwargs)
    except Exception as exc:
        if mode == "offline":
            print(f"[wandb] failed in offline mode: {exc}")
            return None
        print(f"[wandb] online init failed, retrying offline: {exc}")
        init_kwargs["mode"] = "offline"
        try:
            return wandb.init(**init_kwargs)
        except Exception as offline_exc:
            print(f"[wandb] offline init failed: {offline_exc}")
            return None


def _build_probe(args: argparse.Namespace) -> dict[str, Any]:
    loaded = _load_artifact_backend(args) if args.embeddings else _load_image_backend(args)
    features = loaded["features"].astype(np.float64)
    if args.max_tokens and features.shape[0] > args.max_tokens:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(features.shape[0], size=args.max_tokens, replace=False))
        for key in ("features", "image_ids", "bboxes", "patch_indices"):
            loaded[key] = loaded[key][keep]
        features = loaded["features"]

    if loaded.get("all_variant_features") is not None:
        prototype_features = loaded["all_variant_features"]
    else:
        prototype_features = features
    centers, _ = fit_kmeans(
        standardize_features(prototype_features),
        n_clusters=args.prototypes,
        seed=args.seed,
        iters=args.kmeans_iters,
    )

    posteriors = branch_posteriors(features, centers, temperature=args.temperature, top_k=args.top_prototypes)
    metrics = branch_metrics(posteriors)
    instability = np.zeros(features.shape[0], dtype=np.float64)
    singular_source = "fiber_violation"
    fiber_scores = None
    if loaded.get("all_variant_features") is not None:
        variant_post = branch_posteriors(
            loaded["all_variant_features"], centers, temperature=args.temperature, top_k=args.top_prototypes
        )
        instability = augmentation_branch_instability(variant_post, loaded["variant_groups"])
        singular_source = "augmentation_branch_instability"
    if loaded.get("fiber_results"):
        fiber_scores = fiber_singularity_scores(loaded["fiber_results"][: features.shape[0]], alpha=args.alpha)
        singular = fiber_scores["rejected"]
        if not np.any(singular):
            singular = quantile_group_mask(fiber_scores["irregularity"], upper_quantile=args.singular_quantile)
            singular_source = "fiber_irregularity_quantile"
    else:
        branch_score = 0.5 * metrics["branch_entropy_norm"] + 0.5 * instability
        singular = quantile_group_mask(branch_score, upper_quantile=args.singular_quantile)

    if np.all(singular) or not np.any(singular):
        singular = quantile_group_mask(metrics["branch_entropy_norm"], upper_quantile=args.singular_quantile)
        singular_source = "branch_entropy_quantile"

    regular = ~singular
    ks_entropy = ks_2samp(metrics["branch_entropy_norm"][regular], metrics["branch_entropy_norm"][singular])
    ks_margin = ks_2samp(metrics["branch_margin"][regular], metrics["branch_margin"][singular])
    ks_flatness = ks_2samp(metrics["branch_flatness"][regular], metrics["branch_flatness"][singular])
    sliced = sliced_ks_test(
        features,
        singular,
        projections=args.projections,
        permutations=args.permutations,
        seed=args.seed,
    )
    branch_score = 0.5 * metrics["branch_entropy_norm"] + 0.5 * instability
    summary = {
        "backend": loaded["backend"],
        "singular_source": singular_source,
        "num_tokens": int(features.shape[0]),
        "num_singular": int(np.sum(singular)),
        "num_regular": int(np.sum(regular)),
        "prototype_count": int(centers.shape[0]),
        "mean_entropy_singular": float(np.nanmean(metrics["branch_entropy_norm"][singular])),
        "mean_entropy_regular": float(np.nanmean(metrics["branch_entropy_norm"][regular])),
        "mean_margin_singular": float(np.nanmean(metrics["branch_margin"][singular])),
        "mean_margin_regular": float(np.nanmean(metrics["branch_margin"][regular])),
        "mean_instability_singular": float(np.nanmean(instability[singular])),
        "mean_instability_regular": float(np.nanmean(instability[regular])),
        "ks_entropy_D": ks_entropy.statistic,
        "ks_entropy_p": ks_entropy.pvalue,
        "ks_margin_D": ks_margin.statistic,
        "ks_margin_p": ks_margin.pvalue,
        "ks_flatness_D": ks_flatness.statistic,
        "ks_flatness_p": ks_flatness.pvalue,
        "sliced_ks_median_D": sliced.median_statistic,
        "sliced_ks_trimmed_mean_D": sliced.trimmed_mean_statistic,
        "sliced_ks_max_D": sliced.max_statistic,
        "sliced_ks_permutation_p": sliced.permutation_pvalue,
        "entropy_irregularity_pearson": (
            _pearson(metrics["branch_entropy_norm"], fiber_scores["irregularity"])
            if fiber_scores is not None
            else _pearson(metrics["branch_entropy_norm"], instability)
        ),
        "entropy_instability_spearman": _pearson(_rank(metrics["branch_entropy_norm"]), _rank(instability)),
    }
    return {
        **loaded,
        "features": features,
        "centers": centers,
        "posteriors": posteriors,
        "metrics": metrics,
        "instability": instability,
        "branch_score": branch_score,
        "singular": singular,
        "regular": regular,
        "fiber_scores": fiber_scores,
        "ks": {
            "entropy": ks_entropy,
            "margin": ks_margin,
            "flatness": ks_flatness,
            "sliced": sliced,
        },
        "summary": summary,
    }


def _write_outputs(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Path]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = result["metrics"]
    singular = result["singular"]
    regular = result["regular"]
    images = result.get("images")
    image_ids = result["image_ids"]
    bboxes = result["bboxes"]
    fiber_scores = result.get("fiber_scores")
    irregularity = (
        fiber_scores["irregularity"]
        if fiber_scores is not None
        else result["instability"]
    )
    paths: dict[str, Path] = {}
    paths["entropy_cdf"] = _save_cdf_plot(
        out_dir / "branch_entropy_cdf.png",
        metrics["branch_entropy_norm"][regular],
        metrics["branch_entropy_norm"][singular],
        metric="Normalized branch entropy",
        ks_stat=result["ks"]["entropy"].statistic,
    )
    paths["margin_cdf"] = _save_cdf_plot(
        out_dir / "branch_margin_cdf.png",
        metrics["branch_margin"][regular],
        metrics["branch_margin"][singular],
        metric="Top-1 minus top-2 branch margin",
        ks_stat=result["ks"]["margin"].statistic,
    )
    paths["entropy_scatter"] = _save_scatter_plot(
        out_dir / "branch_entropy_vs_singularity.png",
        irregularity,
        metrics["branch_entropy_norm"],
        singular,
        x_label="fiber irregularity or augmentation instability",
        y_label="normalized branch entropy",
        title="Visual Branch Flattening",
    )
    paths["sliced_ks"] = _save_hist_plot(
        out_dir / "sliced_ks_projection_statistics.png",
        result["ks"]["sliced"].projection_statistics,
        result["ks"]["sliced"].null_statistics,
        observed=result["ks"]["sliced"].median_statistic,
        title="Sliced KS over random 1D feature projections",
    )
    heatmap = _save_heatmap_grid(
        out_dir / "branch_entropy_heatmaps.png",
        images,
        image_ids,
        bboxes,
        metrics["branch_entropy_norm"],
        title="Branch entropy projected back to patches",
        max_images=args.max_images,
    )
    if heatmap:
        paths["entropy_heatmap"] = heatmap
    instability_heatmap = _save_heatmap_grid(
        out_dir / "branch_instability_heatmaps.png",
        images,
        image_ids,
        bboxes,
        result["instability"],
        title="Augmentation branch instability projected to patches",
        max_images=args.max_images,
    )
    if instability_heatmap:
        paths["instability_heatmap"] = instability_heatmap
    gallery = _save_patch_gallery(
        out_dir / "top_branch_flattening_patches.png",
        images,
        image_ids,
        bboxes,
        result["branch_score"],
        title="Highest branch-flattening patches",
        top_k=args.gallery_tokens,
    )
    if gallery:
        paths["patch_gallery"] = gallery

    summary_path = out_dir / "vision_branching_ks_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(to_serializable(result["summary"]), fp, indent=2)
    paths["summary"] = summary_path

    token_table_path = out_dir / "vision_branching_tokens.csv"
    with token_table_path.open("w", encoding="utf-8") as fp:
        fp.write("token,image_id,patch_index,singular,branch_entropy_norm,branch_margin,instability,branch_score\n")
        for idx in range(result["features"].shape[0]):
            fp.write(
                f"{idx},{int(image_ids[idx])},{int(result['patch_indices'][idx])},{int(bool(singular[idx]))},"
                f"{metrics['branch_entropy_norm'][idx]:.8f},{metrics['branch_margin'][idx]:.8f},"
                f"{result['instability'][idx]:.8f},{result['branch_score'][idx]:.8f}\n"
            )
    paths["tokens_csv"] = token_table_path
    return paths


def _log_wandb(args: argparse.Namespace, result: dict[str, Any], paths: dict[str, Path]) -> str | None:
    run = _init_wandb(args, result["summary"])
    if run is None:
        return None
    import wandb

    try:
        payload: dict[str, Any] = {f"vision_branching/{k}": v for k, v in result["summary"].items()}
        for key, path in paths.items():
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                payload[f"vision_branching/{key}"] = wandb.Image(str(path))
        if hasattr(wandb, "Table"):
            rows = []
            metrics = result["metrics"]
            for idx in np.argsort(-result["branch_score"])[: min(args.gallery_tokens, result["features"].shape[0])]:
                rows.append([
                    int(idx),
                    int(result["image_ids"][idx]),
                    int(result["patch_indices"][idx]),
                    bool(result["singular"][idx]),
                    float(metrics["branch_entropy_norm"][idx]),
                    float(metrics["branch_margin"][idx]),
                    float(result["instability"][idx]),
                    float(result["branch_score"][idx]),
                ])
            payload["vision_branching/top_tokens"] = wandb.Table(
                columns=[
                    "token",
                    "image_id",
                    "patch_index",
                    "singular",
                    "branch_entropy_norm",
                    "branch_margin",
                    "instability",
                    "branch_score",
                ],
                data=rows,
            )
        wandb.log(payload, step=0)
        if hasattr(wandb, "Artifact"):
            artifact = wandb.Artifact(f"{args.wandb_name or 'vision_branching_ks'}_outputs", type="analysis")
            for path in paths.values():
                artifact.add_file(str(path))
            wandb.log_artifact(artifact)
        url = getattr(run, "url", None)
        wandb.finish()
        return url
    except Exception as exc:
        print(f"[wandb] logging failed: {exc}")
        try:
            wandb.finish()
        except Exception:
            pass
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual branch-flattening KS probe with W&B logging.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--embeddings", type=Path, help="Saved embeddings/epoch_000.pt artifact.")
    source.add_argument("--image-dir", type=Path, help="Directory of images for dependency-light smoke runs.")
    parser.add_argument("--fiber-results", type=Path, help="Matching checkpoints/fiber_epoch_000.json.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name for denormalizing torch artifact images.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/local/vision_branching_ks"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--augmentations", type=int, default=6)
    parser.add_argument("--prototypes", type=int, default=16)
    parser.add_argument("--top-prototypes", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--kmeans-iters", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--singular-quantile", type=float, default=0.85)
    parser.add_argument("--projections", type=int, default=96)
    parser.add_argument("--permutations", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--gallery-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="stratified-manifold-learning")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="vision-branching-ks")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.wandb_mode == "disabled":
        args.wandb = False
    result = _build_probe(args)
    paths = _write_outputs(args, result)
    url = _log_wandb(args, result, paths)
    print(json.dumps(to_serializable(result["summary"]), indent=2))
    print("[outputs]")
    for key, path in paths.items():
        print(f"{key}: {path}")
    if url:
        print(f"[wandb] {url}")
    elif args.wandb:
        print("[wandb] no run URL; check offline logs under the output directory")


if __name__ == "__main__":
    main()
