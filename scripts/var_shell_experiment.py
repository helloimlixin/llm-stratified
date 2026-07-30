"""Run the calibrated radial-shell test on a pretrained VAR visual-token pipeline.

The experiment has two linked parts:

1. Test the normalized VAR VQ codebook with the fitted-dimension multinomial
   shell likelihood-ratio statistic used in the paper.
2. Teacher-force real ImageNet images and freshly generated VAR images, then
   relate each target code's shell deviance to the generator's local branch
   flatness, entropy, likelihood, and confidence.

VAR predicts visual token maps coarse-to-fine, so this is an architectural
control for the raster-order GPT experiment rather than another checkpoint of
the same factorization.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from pretrained_var_generator import (  # noqa: E402
    build_pretrained_var,
    parse_patch_nums,
    resolve_model_defaults,
    save_grid,
)
from vq_gpt_shell_visualizations import (  # noqa: E402
    fit_radial_dimension,
)
from radial_shell_statistics import (  # noqa: E402
    calibrate_fitted_shell_deviance,
    fitted_shell_test_from_distances,
    monte_carlo_pvalue,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return float("nan")
    left_rank = np.argsort(np.argsort(left[mask])).astype(np.float64)
    right_rank = np.argsort(np.argsort(right[mask])).astype(np.float64)
    if float(left_rank.std()) <= 0.0 or float(right_rank.std()) <= 0.0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _permutation_p(
    values: np.ndarray,
    selected: np.ndarray,
    *,
    alternative: str,
    reps: int,
    seed: int,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    selected = np.asarray(selected, dtype=bool)
    finite = np.isfinite(values)
    values = values[finite]
    selected = selected[finite]
    if not selected.any() or selected.all():
        return float("nan")
    observed = float(values[selected].mean() - values[~selected].mean())
    rng = np.random.default_rng(int(seed))
    labels = selected.copy()
    extreme = 0
    for _ in range(int(reps)):
        rng.shuffle(labels)
        diff = float(values[labels].mean() - values[~labels].mean())
        if alternative == "higher":
            extreme += int(diff >= observed)
        elif alternative == "lower":
            extreme += int(diff <= observed)
        else:
            extreme += int(abs(diff) >= abs(observed))
    return float((extreme + 1.0) / (int(reps) + 1.0))


def compare_groups(
    values: np.ndarray,
    selected: np.ndarray,
    *,
    alternative: str,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    selected = np.asarray(selected, dtype=bool)
    finite = np.isfinite(values)
    left = values[finite & selected]
    right = values[finite & ~selected]
    return {
        "selected_count": int(left.size),
        "rest_count": int(right.size),
        "selected_mean": _finite_mean(left),
        "rest_mean": _finite_mean(right),
        "selected_minus_rest": float(left.mean() - right.mean()) if left.size and right.size else float("nan"),
        "alternative": alternative,
        "permutation_p": _permutation_p(
            values,
            selected,
            alternative=alternative,
            reps=int(reps),
            seed=int(seed),
        ),
    }


def compare_groups_by_image(
    values: np.ndarray,
    selected: np.ndarray,
    *,
    num_images: int,
    tokens_per_image: int,
    alternative: str,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(int(num_images), int(tokens_per_image))
    selected = np.asarray(selected, dtype=bool).reshape(int(num_images), int(tokens_per_image))
    differences: list[float] = []
    for image_idx in range(int(num_images)):
        finite = np.isfinite(values[image_idx])
        active = finite & selected[image_idx]
        rest = finite & ~selected[image_idx]
        if active.any() and rest.any():
            differences.append(float(values[image_idx, active].mean() - values[image_idx, rest].mean()))
    diffs = np.asarray(differences, dtype=np.float64)
    if diffs.size == 0:
        return {
            "num_images": 0,
            "mean_image_difference": float("nan"),
            "median_image_difference": float("nan"),
            "wins_expected_direction": 0,
            "exact_sign_flip_p": float("nan"),
            "alternative": alternative,
            "image_differences": [],
        }
    observed = float(diffs.mean())
    if diffs.size <= 20:
        patterns = np.arange(1 << int(diffs.size), dtype=np.uint32)[:, None]
        bits = (patterns >> np.arange(int(diffs.size), dtype=np.uint32)[None, :]) & 1
        signs = np.where(bits > 0, 1.0, -1.0)
        null = signs @ diffs / float(diffs.size)
        if alternative == "higher":
            pvalue = float(np.mean(null >= observed - 1e-15))
            wins = int(np.sum(diffs > 0.0))
        elif alternative == "lower":
            pvalue = float(np.mean(null <= observed + 1e-15))
            wins = int(np.sum(diffs < 0.0))
        else:
            pvalue = float(np.mean(np.abs(null) >= abs(observed) - 1e-15))
            wins = int(np.sum(np.abs(diffs) > 0.0))
    else:
        pvalue = float("nan")
        wins = int(np.sum(diffs > 0.0)) if alternative == "higher" else int(np.sum(diffs < 0.0))
    return {
        "num_images": int(diffs.size),
        "mean_image_difference": observed,
        "median_image_difference": float(np.median(diffs)),
        "wins_expected_direction": wins,
        "exact_sign_flip_p": pvalue,
        "alternative": alternative,
        "image_differences": diffs.tolist(),
    }


@torch.inference_mode()
def nearest_neighbor_distances(
    embeddings: torch.Tensor,
    *,
    neighbors: int,
    chunk_size: int = 512,
) -> np.ndarray:
    embeddings = F.normalize(embeddings.detach().float(), dim=1)
    total = int(embeddings.shape[0])
    if not 2 <= int(neighbors) < total:
        raise ValueError("neighbors must be between 2 and codebook_size - 1")
    chunks: list[torch.Tensor] = []
    for start in range(0, total, int(chunk_size)):
        end = min(start + int(chunk_size), total)
        distances = torch.cdist(embeddings[start:end], embeddings)
        local_rows = torch.arange(end - start, device=embeddings.device)
        global_rows = torch.arange(start, end, device=embeddings.device)
        distances[local_rows, global_rows] = torch.inf
        nearest = distances.topk(k=int(neighbors), dim=1, largest=False, sorted=True).values
        chunks.append(nearest.cpu())
    return torch.cat(chunks, dim=0).numpy().astype(np.float64)


def analyze_codebook(
    codebook: torch.Tensor,
    *,
    neighbors: int,
    bins: int,
    alpha: float,
    calibration_trials: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    normalized = F.normalize(codebook.detach().float(), dim=1)
    distances = nearest_neighbor_distances(normalized.to(device), neighbors=int(neighbors))
    total = int(distances.shape[0])
    tested = int(neighbors) - 1
    critical, null_statistics = calibrate_fitted_shell_deviance(
        samples=tested,
        bins=int(bins),
        alpha=float(alpha),
        trials=int(calibration_trials),
        seed=int(seed) + 900,
    )
    dimensions = np.full(total, np.nan, dtype=np.float64)
    statistics = np.full(total, np.nan, dtype=np.float64)
    scores = np.full(total, np.nan, dtype=np.float64)
    pvalues = np.full(total, np.nan, dtype=np.float64)
    radii = np.full(total, np.nan, dtype=np.float64)
    shell_count_rows = np.zeros((total, int(bins)), dtype=np.int64)
    for code_id in range(total):
        local = distances[code_id]
        radius = float(local[-1])
        inner = local[:-1]
        dimensions[code_id] = fit_radial_dimension(inner, radius)
        shell_count_rows[code_id], statistics[code_id] = fitted_shell_test_from_distances(
            inner,
            radius=radius,
            bins=int(bins),
        )
        scores[code_id] = statistics[code_id] / critical
        pvalues[code_id] = monte_carlo_pvalue(statistics[code_id], null_statistics)
        radii[code_id] = radius
    rejected = np.isfinite(pvalues) & (pvalues <= float(alpha))
    coords = normalized.cpu().numpy().astype(np.float64)
    centered = coords - coords.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    pca = centered @ vt[:2].T
    summary = {
        "num_codes": total,
        "embedding_dimension": int(normalized.shape[1]),
        "normalization": "L2-normalized codebook vectors",
        "neighbors": int(neighbors),
        "tested_inner_radii": tested,
        "bins": int(bins),
        "alpha": float(alpha),
        "calibration_trials": int(calibration_trials),
        "shell_lrt_critical": float(critical),
        "shell_lrt_reject_count": int(rejected.sum()),
        "shell_lrt_reject_fraction": float(rejected.mean()),
        "mean_dimension_hat": _finite_mean(dimensions),
        "median_dimension_hat": float(np.nanmedian(dimensions)),
        "median_shell_lrt_score": float(np.nanmedian(scores)),
        "shell_lrt_score_quantiles": {
            "q50": float(np.nanquantile(scores, 0.50)),
            "q90": float(np.nanquantile(scores, 0.90)),
            "q95": float(np.nanquantile(scores, 0.95)),
            "q99": float(np.nanquantile(scores, 0.99)),
        },
    }
    arrays = {
        "distances": distances,
        "dimensions": dimensions,
        "statistics": statistics,
        "scores": scores,
        "pvalues": pvalues,
        "rejected": rejected,
        "radii": radii,
        "shell_counts": shell_count_rows,
        "pca": pca,
        "normalized_codebook": normalized.cpu().numpy(),
    }
    return summary, arrays


def load_dataset_records(path: Path, *, image_size: int, limit: int) -> tuple[torch.Tensor, list[int], list[str]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = rows[: int(limit)] if int(limit) > 0 else rows
    images: list[torch.Tensor] = []
    labels: list[int] = []
    names: list[str] = []
    for row in rows:
        image_path = Path(row["path"])
        image = ImageOps.fit(
            Image.open(image_path).convert("RGB"),
            (int(image_size), int(image_size)),
            method=Image.Resampling.BICUBIC,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        images.append(torch.from_numpy(array).permute(2, 0, 1))
        labels.append(int(row["class_label"]))
        names.append(str(row.get("relative_path", image_path.name)))
    if not images:
        raise ValueError("dataset record file did not provide any images")
    return torch.stack(images, dim=0), labels, names


@torch.inference_mode()
def teacher_forced_metrics(
    vae,
    var,
    *,
    images: torch.Tensor,
    labels: list[int],
    patch_nums: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    branch_top_k: int,
) -> dict[str, np.ndarray]:
    all_targets: list[np.ndarray] = []
    all_entropy: list[np.ndarray] = []
    all_branch_ks: list[np.ndarray] = []
    all_branch_entropy: list[np.ndarray] = []
    all_top_mass: list[np.ndarray] = []
    all_nll: list[np.ndarray] = []
    all_top1: list[np.ndarray] = []
    model_dtype = next(vae.parameters()).dtype
    for start in range(0, int(images.shape[0]), int(batch_size)):
        end = min(start + int(batch_size), int(images.shape[0]))
        batch = images[start:end].to(device=device, dtype=model_dtype)
        pixels = batch.mul(2.0).sub(1.0).clamp(-1.0, 1.0)
        idx_bl = vae.img_to_idxBl(pixels, v_patch_nums=patch_nums)
        x_in = vae.quantize.idxBl_to_var_input(idx_bl)
        label_tensor = torch.tensor(labels[start:end], dtype=torch.long, device=device)
        logits = var(label_tensor, x_in)
        scale_start, scale_end = var.begin_ends[-1]
        final_logits = logits[:, scale_start:scale_end, :].float()
        targets = idx_bl[-1].long()
        log_probs = final_logits.log_softmax(dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1) / math.log(int(probs.shape[-1]))
        nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        top1 = probs.amax(dim=-1)
        k = max(2, min(int(branch_top_k), int(probs.shape[-1])))
        top = probs.topk(k=k, dim=-1, largest=True, sorted=True).values
        top_mass = top.sum(dim=-1)
        local = top / top_mass.unsqueeze(-1).clamp_min(1e-12)
        uniform_cdf = torch.arange(1, k + 1, device=device, dtype=local.dtype) / float(k)
        branch_ks = (local.cumsum(dim=-1) - uniform_cdf).abs().amax(dim=-1)
        branch_entropy = -(local * local.clamp_min(1e-12).log()).sum(dim=-1) / math.log(k)
        all_targets.append(targets.cpu().numpy().astype(np.int64))
        all_entropy.append(entropy.cpu().numpy().astype(np.float64))
        all_branch_ks.append(branch_ks.cpu().numpy().astype(np.float64))
        all_branch_entropy.append(branch_entropy.cpu().numpy().astype(np.float64))
        all_top_mass.append(top_mass.cpu().numpy().astype(np.float64))
        all_nll.append(nll.cpu().numpy().astype(np.float64))
        all_top1.append(top1.cpu().numpy().astype(np.float64))
    return {
        "targets": np.concatenate(all_targets, axis=0),
        "entropy_norm": np.concatenate(all_entropy, axis=0),
        "branch_ks": np.concatenate(all_branch_ks, axis=0),
        "branch_entropy_norm": np.concatenate(all_branch_entropy, axis=0),
        "branch_topk_mass": np.concatenate(all_top_mass, axis=0),
        "nll": np.concatenate(all_nll, axis=0),
        "top1_prob": np.concatenate(all_top1, axis=0),
    }


def summarize_positions(
    metrics: dict[str, np.ndarray],
    *,
    codebook_arrays: dict[str, np.ndarray],
    source: str,
    permutation_reps: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    targets = metrics["targets"].reshape(-1)
    code_scores = codebook_arrays["scores"][targets]
    code_rejected = codebook_arrays["rejected"][targets]
    code_dimensions = codebook_arrays["dimensions"][targets]
    flattened = {key: np.asarray(value).reshape(-1) for key, value in metrics.items() if key != "targets"}
    comparisons = {
        "branch_ks": compare_groups(
            flattened["branch_ks"],
            code_rejected,
            alternative="lower",
            reps=permutation_reps,
            seed=seed + 1,
        ),
        "branch_entropy_norm": compare_groups(
            flattened["branch_entropy_norm"],
            code_rejected,
            alternative="higher",
            reps=permutation_reps,
            seed=seed + 2,
        ),
        "full_entropy_norm": compare_groups(
            flattened["entropy_norm"],
            code_rejected,
            alternative="higher",
            reps=permutation_reps,
            seed=seed + 3,
        ),
        "nll": compare_groups(
            flattened["nll"],
            code_rejected,
            alternative="higher",
            reps=permutation_reps,
            seed=seed + 4,
        ),
        "top1_prob": compare_groups(
            flattened["top1_prob"],
            code_rejected,
            alternative="lower",
            reps=permutation_reps,
            seed=seed + 5,
        ),
    }
    num_images = int(metrics["targets"].shape[0])
    tokens_per_image = int(metrics["targets"].shape[1])
    image_cluster_comparisons = {
        "branch_ks": compare_groups_by_image(
            flattened["branch_ks"],
            code_rejected,
            num_images=num_images,
            tokens_per_image=tokens_per_image,
            alternative="lower",
        ),
        "branch_entropy_norm": compare_groups_by_image(
            flattened["branch_entropy_norm"],
            code_rejected,
            num_images=num_images,
            tokens_per_image=tokens_per_image,
            alternative="higher",
        ),
        "full_entropy_norm": compare_groups_by_image(
            flattened["entropy_norm"],
            code_rejected,
            num_images=num_images,
            tokens_per_image=tokens_per_image,
            alternative="higher",
        ),
        "nll": compare_groups_by_image(
            flattened["nll"],
            code_rejected,
            num_images=num_images,
            tokens_per_image=tokens_per_image,
            alternative="higher",
        ),
        "top1_prob": compare_groups_by_image(
            flattened["top1_prob"],
            code_rejected,
            num_images=num_images,
            tokens_per_image=tokens_per_image,
            alternative="lower",
        ),
    }
    summary = {
        "source": source,
        "num_images": num_images,
        "tokens_per_image": tokens_per_image,
        "num_positions": int(targets.size),
        "unique_target_codes": int(np.unique(targets).size),
        "target_rejected_code_fraction": float(code_rejected.mean()),
        "mean_target_code_shell_lrt_score": _finite_mean(code_scores),
        "median_target_code_shell_lrt_score": float(np.nanmedian(code_scores)),
        "mean_target_code_dimension": _finite_mean(code_dimensions),
        "mean_branch_ks": _finite_mean(flattened["branch_ks"]),
        "mean_branch_entropy_norm": _finite_mean(flattened["branch_entropy_norm"]),
        "mean_full_entropy_norm": _finite_mean(flattened["entropy_norm"]),
        "mean_nll": _finite_mean(flattened["nll"]),
        "mean_top1_prob": _finite_mean(flattened["top1_prob"]),
        "spearman_code_score_branch_ks": _rank_correlation(code_scores, flattened["branch_ks"]),
        "spearman_code_score_branch_entropy": _rank_correlation(code_scores, flattened["branch_entropy_norm"]),
        "spearman_code_score_full_entropy": _rank_correlation(code_scores, flattened["entropy_norm"]),
        "spearman_code_score_nll": _rank_correlation(code_scores, flattened["nll"]),
        "comparisons": comparisons,
        "image_cluster_comparisons": image_cluster_comparisons,
    }
    arrays = dict(flattened)
    arrays.update(
        {
            "targets": targets,
            "code_scores": code_scores,
            "code_rejected": code_rejected,
            "code_dimensions": code_dimensions,
        }
    )
    return summary, arrays


def _metric_maps(metrics: dict[str, np.ndarray], codebook_arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    targets = metrics["targets"]
    grid = int(round(math.sqrt(int(targets.shape[1]))))
    radial = codebook_arrays["scores"][targets].reshape(targets.shape[0], grid, grid)
    entropy = metrics["branch_entropy_norm"].reshape(targets.shape[0], grid, grid)
    return radial, entropy


def write_heatmap_gallery(
    path: Path,
    *,
    real_images: torch.Tensor,
    real_labels: list[int],
    real_metrics: dict[str, np.ndarray],
    generated_images: torch.Tensor,
    generated_labels: list[int],
    generated_metrics: dict[str, np.ndarray],
    codebook_arrays: dict[str, np.ndarray],
) -> str:
    real_radial, real_entropy = _metric_maps(real_metrics, codebook_arrays)
    gen_radial, gen_entropy = _metric_maps(generated_metrics, codebook_arrays)
    rows = min(4, int(real_images.shape[0]), int(generated_images.shape[0]))
    radial_values = np.concatenate([real_radial[:rows].reshape(-1), gen_radial[:rows].reshape(-1)])
    entropy_values = np.concatenate([real_entropy[:rows].reshape(-1), gen_entropy[:rows].reshape(-1)])
    radial_max = float(np.nanquantile(radial_values, 0.98))
    entropy_min = float(np.nanquantile(entropy_values, 0.02))
    entropy_max = float(np.nanquantile(entropy_values, 0.98))
    fig, axes = plt.subplots(rows, 6, figsize=(18, 3.15 * rows), constrained_layout=True)
    if rows == 1:
        axes = axes[None, :]
    radial_mappable = None
    entropy_mappable = None
    for row in range(rows):
        groups = [
            (real_images[row], real_labels[row], "ImageNet", real_radial[row], real_entropy[row]),
            (generated_images[row], generated_labels[row], "VAR sample", gen_radial[row], gen_entropy[row]),
        ]
        for group_idx, (image, label, source, radial, entropy) in enumerate(groups):
            base = 3 * group_idx
            image_np = image.permute(1, 2, 0).numpy()
            axes[row, base].imshow(np.clip(image_np, 0.0, 1.0))
            axes[row, base].set_title(f"{source} | class {label}", fontsize=10)
            axes[row, base + 1].imshow(np.clip(image_np, 0.0, 1.0))
            radial_mappable = axes[row, base + 1].imshow(
                radial,
                cmap="viridis",
                alpha=0.62,
                interpolation="nearest",
                extent=(0, image_np.shape[1], image_np.shape[0], 0),
                vmin=0.0,
                vmax=max(radial_max, 1.0),
            )
            axes[row, base + 1].set_title("shell-deviance ratio", fontsize=10)
            axes[row, base + 2].imshow(np.clip(image_np, 0.0, 1.0))
            entropy_mappable = axes[row, base + 2].imshow(
                entropy,
                cmap="magma",
                alpha=0.62,
                interpolation="nearest",
                extent=(0, image_np.shape[1], image_np.shape[0], 0),
                vmin=entropy_min,
                vmax=entropy_max,
            )
            axes[row, base + 2].set_title("top-32 branch entropy", fontsize=10)
        for ax in axes[row]:
            ax.axis("off")
    if radial_mappable is not None:
        fig.colorbar(radial_mappable, ax=axes[:, [1, 4]].ravel().tolist(), fraction=0.012, pad=0.008)
    if entropy_mappable is not None:
        fig.colorbar(entropy_mappable, ax=axes[:, [2, 5]].ravel().tolist(), fraction=0.012, pad=0.008)
    fig.suptitle("VAR visual tokens: radial geometry and coarse-to-fine branch uncertainty", fontsize=16)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def write_dashboard(
    path: Path,
    *,
    codebook_summary: dict[str, Any],
    codebook_arrays: dict[str, np.ndarray],
    real_summary: dict[str, Any],
    generated_summary: dict[str, Any],
    all_targets: np.ndarray,
    llamagen_summary_path: Path | None,
) -> str:
    scores = codebook_arrays["scores"]
    dims = codebook_arrays["dimensions"]
    pca = codebook_arrays["pca"]
    rejected = codebook_arrays["rejected"]
    counts = np.bincount(all_targets.reshape(-1), minlength=int(scores.size))
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), constrained_layout=True)
    axes[0, 0].hist(scores[np.isfinite(scores)], bins=55, color="#4c78a8", alpha=0.92)
    axes[0, 0].axvline(1.0, color="#d62728", linewidth=2)
    axes[0, 0].set_title("VAR codebook radial scores")
    axes[0, 0].set_xlabel("shell deviance / 5% Monte Carlo critical value")
    scatter = axes[0, 1].scatter(pca[:, 0], pca[:, 1], c=np.clip(scores, 0.0, 5.0), s=7, cmap="viridis", linewidths=0)
    axes[0, 1].set_title("Normalized VQ codebook PCA")
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")
    fig.colorbar(scatter, ax=axes[0, 1], fraction=0.046, pad=0.03, label="clipped shell-deviance ratio")
    axes[0, 2].hist(dims[np.isfinite(dims)], bins=50, color="#72b7b2", alpha=0.92)
    axes[0, 2].set_title("Fitted radial dimensions")
    axes[0, 2].set_xlabel("dimension estimate")

    pipeline_names = ["VAR"]
    reject_rates = [float(codebook_summary["shell_lrt_reject_fraction"])]
    median_dims = [float(codebook_summary["median_dimension_hat"])]
    if llamagen_summary_path is not None and llamagen_summary_path.exists():
        llama = json.loads(llamagen_summary_path.read_text(encoding="utf-8"))["codebook_shell_test"]
        pipeline_names.insert(0, "LlamaGen")
        reject_rates.insert(0, float(llama["shell_lrt_reject_fraction"]))
        median_dims.insert(0, float(llama["median_dimension_hat"]))
    x = np.arange(len(pipeline_names))
    width = 0.36
    axes[1, 0].bar(x - width / 2, reject_rates, width, color="#f58518", label="reject fraction")
    axes[1, 0].bar(x + width / 2, np.asarray(median_dims) / max(max(median_dims), 1e-12), width, color="#54a24b", label="median dimension, normalized")
    axes[1, 0].set_xticks(x, pipeline_names)
    axes[1, 0].set_ylim(0.0, 1.08)
    axes[1, 0].set_title("Visual-token pipeline comparison")
    axes[1, 0].legend(fontsize=8)

    metric_names = ["branch KS", "branch H", "full H", "NLL", "top-1 p"]
    keys = ["branch_ks", "branch_entropy_norm", "full_entropy_norm", "nll", "top1_prob"]
    real_diffs = [real_summary["comparisons"][key]["selected_minus_rest"] for key in keys]
    gen_diffs = [generated_summary["comparisons"][key]["selected_minus_rest"] for key in keys]
    x = np.arange(len(keys))
    axes[1, 1].bar(x - width / 2, real_diffs, width, color="#4c78a8", label="ImageNet")
    axes[1, 1].bar(x + width / 2, gen_diffs, width, color="#e45756", label="generated")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_xticks(x, metric_names, rotation=25, ha="right")
    axes[1, 1].set_title("Rejected-code positions minus rest")
    axes[1, 1].legend(fontsize=8)

    used = counts > 0
    axes[1, 2].scatter(scores[~used], np.zeros((~used).sum()), s=5, alpha=0.18, color="#bab0ac", label="unused")
    axes[1, 2].scatter(scores[used], np.log1p(counts[used]), s=10, alpha=0.65, c=np.where(rejected[used], "#e45756", "#4c78a8"), label="used")
    axes[1, 2].axvline(1.0, color="#d62728", linewidth=1.5)
    axes[1, 2].set_xlabel("codebook shell-deviance ratio")
    axes[1, 2].set_ylabel("log(1 + target usage)")
    axes[1, 2].set_title("Code geometry versus token usage")
    for ax in axes.ravel():
        ax.grid(alpha=0.15)
    fig.suptitle("Pretrained VAR VQ tokenizer and next-scale generator", fontsize=17)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.model_dtype]
    defaults = resolve_model_defaults(int(args.model_depth))
    patch_nums = parse_patch_nums(args.patch_nums, resolution=int(defaults["resolution"]))
    image_size = 16 * int(patch_nums[-1])
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    print(f"[model] loading VAR-d{args.model_depth} on {device} ({args.model_dtype})", flush=True)
    vae, var, vae_path, var_path = build_pretrained_var(
        depth=int(args.model_depth),
        repo_id=args.repo_id,
        vae_filename=args.vae_filename,
        var_filename=args.var_filename or str(defaults["filename"]),
        patch_nums=patch_nums,
        shared_aln=bool(defaults["shared_aln"]),
        device=device,
        var_repo_path=args.var_repo_path,
        dtype=dtype if dtype is not torch.float32 else None,
        mmap_load=bool(args.mmap_load),
    )

    print("[codebook] computing nearest-neighbor radial tests", flush=True)
    codebook_summary, codebook_arrays = analyze_codebook(
        vae.quantize.embedding.weight,
        neighbors=int(args.neighbors),
        bins=int(args.bins),
        alpha=float(args.alpha),
        calibration_trials=int(args.calibration_trials),
        seed=int(args.seed),
        device=device,
    )

    real_images, real_labels, real_names = load_dataset_records(
        Path(args.dataset_records),
        image_size=image_size,
        limit=int(args.real_images),
    )
    print(f"[real] teacher-forcing {len(real_labels)} ImageNet images", flush=True)
    real_metrics = teacher_forced_metrics(
        vae,
        var,
        images=real_images,
        labels=real_labels,
        patch_nums=patch_nums,
        device=device,
        batch_size=int(args.batch_size),
        branch_top_k=int(args.branch_top_k),
    )

    generated_labels = [int(part) for part in str(args.generated_classes).split(",") if part.strip()]
    if len(generated_labels) < int(args.generated_images):
        repeats = int(math.ceil(int(args.generated_images) / max(len(generated_labels), 1)))
        generated_labels = (generated_labels * repeats)[: int(args.generated_images)]
    else:
        generated_labels = generated_labels[: int(args.generated_images)]
    print(f"[generate] sampling {len(generated_labels)} fresh VAR images", flush=True)
    generated_batches: list[torch.Tensor] = []
    for start in range(0, len(generated_labels), int(args.generation_batch_size)):
        batch_labels = generated_labels[start:start + int(args.generation_batch_size)]
        with torch.inference_mode():
            generated_batches.append(
                var.autoregressive_infer_cfg(
                    B=len(batch_labels),
                    label_B=torch.tensor(batch_labels, dtype=torch.long, device=device),
                    cfg=float(args.cfg),
                    top_k=int(args.top_k),
                    top_p=float(args.top_p),
                    g_seed=int(args.seed) + start,
                    more_smooth=False,
                ).detach().float().cpu()
            )
    generated_images = torch.cat(generated_batches, dim=0)
    save_grid(
        generated_images,
        out_dir / "var_generated_samples.png",
        labels=generated_labels,
        title=f"Fresh pretrained VAR-d{args.model_depth} samples",
        cols=min(4, len(generated_labels)),
    )
    print("[generated] re-encoding and teacher-forcing generated images", flush=True)
    generated_metrics = teacher_forced_metrics(
        vae,
        var,
        images=generated_images,
        labels=generated_labels,
        patch_nums=patch_nums,
        device=device,
        batch_size=int(args.batch_size),
        branch_top_k=int(args.branch_top_k),
    )

    real_summary, real_arrays = summarize_positions(
        real_metrics,
        codebook_arrays=codebook_arrays,
        source="imagenet_validation",
        permutation_reps=int(args.permutation_reps),
        seed=int(args.seed) + 100,
    )
    generated_summary, generated_arrays = summarize_positions(
        generated_metrics,
        codebook_arrays=codebook_arrays,
        source="var_generated",
        permutation_reps=int(args.permutation_reps),
        seed=int(args.seed) + 200,
    )
    all_targets = np.concatenate([real_metrics["targets"].reshape(-1), generated_metrics["targets"].reshape(-1)])
    figures = {
        "generated_samples": str(out_dir / "var_generated_samples.png"),
        "heatmap_gallery": write_heatmap_gallery(
            out_dir / "var_imagenet_generated_shell_gallery.png",
            real_images=real_images,
            real_labels=real_labels,
            real_metrics=real_metrics,
            generated_images=generated_images,
            generated_labels=generated_labels,
            generated_metrics=generated_metrics,
            codebook_arrays=codebook_arrays,
        ),
        "dashboard": write_dashboard(
            out_dir / "var_shell_dashboard.png",
            codebook_summary=codebook_summary,
            codebook_arrays=codebook_arrays,
            real_summary=real_summary,
            generated_summary=generated_summary,
            all_targets=all_targets,
            llamagen_summary_path=Path(args.llamagen_summary) if args.llamagen_summary else None,
        ),
    }

    records = []
    for source, arrays, image_count in (
        ("imagenet_validation", real_arrays, int(real_images.shape[0])),
        ("var_generated", generated_arrays, int(generated_images.shape[0])),
    ):
        tokens_per_image = int(arrays["targets"].size // image_count)
        for idx in range(int(arrays["targets"].size)):
            records.append(
                {
                    "source": source,
                    "image_index": int(idx // tokens_per_image),
                    "position": int(idx % tokens_per_image),
                    "target_code": int(arrays["targets"][idx]),
                    "code_shell_lrt_score": float(arrays["code_scores"][idx]),
                    "code_rejected": bool(arrays["code_rejected"][idx]),
                    "code_dimension": float(arrays["code_dimensions"][idx]),
                    "branch_ks": float(arrays["branch_ks"][idx]),
                    "branch_entropy_norm": float(arrays["branch_entropy_norm"][idx]),
                    "full_entropy_norm": float(arrays["entropy_norm"][idx]),
                    "nll": float(arrays["nll"][idx]),
                    "top1_prob": float(arrays["top1_prob"][idx]),
                }
            )

    summary = {
        "analysis": "var_shell_experiment",
        "model": {
            "repo_id": args.repo_id,
            "model_depth": int(args.model_depth),
            "var_filename": args.var_filename or str(defaults["filename"]),
            "vae_filename": args.vae_filename,
            "reported_fid": float(defaults["fid"]),
            "reported_params": str(defaults["params"]),
            "resolution": int(defaults["resolution"]),
            "patch_nums": list(patch_nums),
            "device": str(device),
            "dtype": args.model_dtype,
            "vae_path": str(vae_path),
            "var_path": str(var_path),
        },
        "codebook_shell_test": codebook_summary,
        "imagenet_positions": real_summary,
        "generated_positions": generated_summary,
        "image_records": {
            "imagenet_names": real_names,
            "imagenet_labels": real_labels,
            "generated_labels": generated_labels,
        },
        "figures": figures,
        "artifacts": {
            "summary": str(out_dir / "var_shell_summary.json"),
            "position_records": str(out_dir / "var_shell_position_records.json"),
            "codebook_arrays": str(out_dir / "var_codebook_shell_arrays.npz"),
        },
    }
    np.savez_compressed(
        out_dir / "var_codebook_shell_arrays.npz",
        **{key: value for key, value in codebook_arrays.items() if key != "distances"},
    )
    (out_dir / "var_shell_position_records.json").write_text(
        json.dumps(records, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (out_dir / "var_shell_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return summary


def build_argparser() -> argparse.ArgumentParser:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=REPO_ROOT / "runs" / "local" / "var_shell_experiment" / stamp)
    parser.add_argument("--repo-id", default="FoundationVision/var")
    parser.add_argument("--vae-filename", default="vae_ch160v4096z32.pth")
    parser.add_argument("--var-filename", default=None)
    parser.add_argument("--var-repo-path", default=None)
    parser.add_argument("--model-depth", type=int, default=16, choices=[16, 20, 24, 30, 36])
    parser.add_argument("--patch-nums", default="auto")
    parser.add_argument(
        "--dataset-records",
        default=REPO_ROOT / "runs/local/pretrained_vq_ar/llamagen_c2i_B_256_imagenet_val256_seed20260628/llamagen_c2i_dataset_records.json",
    )
    parser.add_argument(
        "--llamagen-summary",
        default=REPO_ROOT / "runs/local/vq_gpt_shell_visualizations/20260722_analytic_ad/vq_gpt_shell_summary.json",
    )
    parser.add_argument("--real-images", type=int, default=16)
    parser.add_argument("--generated-images", type=int, default=8)
    parser.add_argument("--generated-classes", default="207,281,388,530,554,731,850,980")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--neighbors", type=int, default=128)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--calibration-trials", type=int, default=50000)
    parser.add_argument("--branch-top-k", type=int, default=32)
    parser.add_argument("--permutation-reps", type=int, default=3000)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--top-k", type=int, default=900)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--model-dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Use float32 for teacher forcing; the released VAR forward path casts its inputs to float32.",
    )
    parser.add_argument("--mmap-load", action="store_true")
    return parser


def main() -> None:
    summary = run(build_argparser().parse_args())
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
