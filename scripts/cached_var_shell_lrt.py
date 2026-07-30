#!/usr/bin/env python3
"""Recompute VAR shell-LRT results from cached codebook and position artifacts."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from radial_shell_statistics import (
        calibrate_fitted_shell_deviance,
        fitted_shell_test_from_distances,
        monte_carlo_pvalue,
    )
except ImportError:  # Imported as scripts.cached_var_shell_lrt.
    from scripts.radial_shell_statistics import (
        calibrate_fitted_shell_deviance,
        fitted_shell_test_from_distances,
        monte_carlo_pvalue,
    )


def fit_radial_dimension(distances: np.ndarray, radius: float) -> float:
    values = np.asarray(distances, dtype=np.float64)
    scaled = np.clip(values / float(radius), 1e-12, 1.0)
    denominator = -float(np.log(scaled).sum())
    return float(values.size / denominator) if denominator > 0.0 else float("nan")


def nearest_neighbor_distances(codebook: np.ndarray, *, neighbors: int, chunk_size: int = 256) -> np.ndarray:
    vectors = np.asarray(codebook, dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    total = int(vectors.shape[0])
    if not 2 <= int(neighbors) < total:
        raise ValueError("neighbors must be between two and codebook size minus one")
    result = np.empty((total, int(neighbors)), dtype=np.float64)
    for start in range(0, total, int(chunk_size)):
        stop = min(start + int(chunk_size), total)
        similarities = vectors[start:stop] @ vectors.T
        squared = np.maximum(2.0 - 2.0 * similarities, 0.0)
        squared[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nearest = np.partition(squared, kth=int(neighbors) - 1, axis=1)[:, : int(neighbors)]
        nearest.sort(axis=1)
        result[start:stop] = np.sqrt(nearest)
    return result


def analyze_cached_codebook(
    arrays_path: Path,
    *,
    neighbors: int,
    bins: int,
    alpha: float,
    calibration_trials: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    packed = np.load(arrays_path)
    codebook = np.asarray(packed["normalized_codebook"], dtype=np.float32)
    pca = np.asarray(packed["pca"], dtype=np.float64)
    distances = nearest_neighbor_distances(codebook, neighbors=int(neighbors))
    tested = int(neighbors) - 1
    critical, null_statistics = calibrate_fitted_shell_deviance(
        samples=tested,
        bins=int(bins),
        alpha=float(alpha),
        trials=int(calibration_trials),
        seed=int(seed),
    )
    total = int(codebook.shape[0])
    dimensions = np.empty(total, dtype=np.float64)
    statistics = np.empty(total, dtype=np.float64)
    scores = np.empty(total, dtype=np.float64)
    pvalues = np.empty(total, dtype=np.float64)
    radii = np.empty(total, dtype=np.float64)
    counts = np.empty((total, int(bins)), dtype=np.int64)
    for code_id in range(total):
        radius = float(distances[code_id, -1])
        inner = distances[code_id, :-1]
        dimensions[code_id] = fit_radial_dimension(inner, radius)
        counts[code_id], statistics[code_id] = fitted_shell_test_from_distances(
            inner,
            radius=radius,
            bins=int(bins),
        )
        scores[code_id] = statistics[code_id] / critical
        pvalues[code_id] = monte_carlo_pvalue(statistics[code_id], null_statistics)
        radii[code_id] = radius
    rejected = pvalues <= float(alpha)
    summary = {
        "num_codes": total,
        "embedding_dimension": int(codebook.shape[1]),
        "normalization": "L2-normalized codebook vectors",
        "neighbors": int(neighbors),
        "tested_inner_radii": tested,
        "bins": int(bins),
        "alpha": float(alpha),
        "calibration_trials": int(calibration_trials),
        "shell_lrt_critical": float(critical),
        "shell_lrt_reject_count": int(rejected.sum()),
        "shell_lrt_reject_fraction": float(rejected.mean()),
        "mean_dimension_hat": float(np.mean(dimensions)),
        "median_dimension_hat": float(np.median(dimensions)),
        "median_shell_lrt_score": float(np.median(scores)),
        "shell_lrt_score_quantiles": {
            "q50": float(np.quantile(scores, 0.50)),
            "q90": float(np.quantile(scores, 0.90)),
            "q95": float(np.quantile(scores, 0.95)),
            "q99": float(np.quantile(scores, 0.99)),
        },
    }
    return summary, {
        "dimensions": dimensions,
        "statistics": statistics,
        "scores": scores,
        "pvalues": pvalues,
        "rejected": rejected,
        "radii": radii,
        "shell_counts": counts,
        "pca": pca,
        "normalized_codebook": codebook,
    }


def permutation_pvalue(
    values: np.ndarray,
    selected: np.ndarray,
    *,
    alternative: str,
    reps: int,
    seed: int,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    selected = np.asarray(selected, dtype=bool)
    observed = float(values[selected].mean() - values[~selected].mean())
    rng = np.random.default_rng(int(seed))
    extreme = 0
    for _ in range(int(reps)):
        permuted = rng.permutation(selected)
        difference = float(values[permuted].mean() - values[~permuted].mean())
        extreme += difference >= observed if alternative == "higher" else difference <= observed
    return float((extreme + 1) / (int(reps) + 1))


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
    return {
        "selected_count": int(selected.sum()),
        "rest_count": int((~selected).sum()),
        "selected_mean": float(values[selected].mean()),
        "rest_mean": float(values[~selected].mean()),
        "selected_minus_rest": float(values[selected].mean() - values[~selected].mean()),
        "alternative": alternative,
        "permutation_p": permutation_pvalue(
            values,
            selected,
            alternative=alternative,
            reps=int(reps),
            seed=int(seed),
        ),
    }


def image_sign_flip(
    values: np.ndarray,
    selected: np.ndarray,
    image_indices: np.ndarray,
    *,
    alternative: str,
) -> dict[str, Any]:
    differences = []
    for image_index in sorted(set(int(value) for value in image_indices)):
        mask = image_indices == image_index
        local_selected = selected[mask]
        if not local_selected.any() or local_selected.all():
            continue
        local_values = values[mask]
        differences.append(float(local_values[local_selected].mean() - local_values[~local_selected].mean()))
    diffs = np.asarray(differences, dtype=np.float64)
    observed = float(diffs.mean())
    combinations = 1 << int(diffs.size)
    extreme = 0
    for mask in range(combinations):
        signs = np.asarray([1.0 if mask & (1 << bit) else -1.0 for bit in range(diffs.size)])
        candidate = float(np.mean(signs * diffs))
        extreme += candidate >= observed if alternative == "higher" else candidate <= observed
    return {
        "num_images": int(diffs.size),
        "mean_image_difference": observed,
        "median_image_difference": float(np.median(diffs)),
        "wins_expected_direction": int(np.sum(diffs > 0.0) if alternative == "higher" else np.sum(diffs < 0.0)),
        "exact_sign_flip_p": float(extreme / combinations),
        "alternative": alternative,
        "image_differences": diffs.tolist(),
    }


def summarize_positions(
    records: list[dict[str, Any]],
    *,
    source: str,
    codebook_arrays: dict[str, np.ndarray],
    permutation_reps: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected_records = [row for row in records if row["source"] == source]
    targets = np.asarray([int(row["target_code"]) for row in selected_records], dtype=np.int64)
    image_indices = np.asarray([int(row["image_index"]) for row in selected_records], dtype=np.int64)
    selected = codebook_arrays["rejected"][targets]
    scores = codebook_arrays["scores"][targets]
    dimensions = codebook_arrays["dimensions"][targets]
    metric_specs = {
        "branch_ks": "lower",
        "branch_entropy_norm": "higher",
        "full_entropy_norm": "higher",
        "nll": "higher",
        "top1_prob": "lower",
    }
    comparisons: dict[str, Any] = {}
    image_comparisons: dict[str, Any] = {}
    metrics: dict[str, np.ndarray] = {}
    for offset, (metric, alternative) in enumerate(metric_specs.items()):
        values = np.asarray([float(row[metric]) for row in selected_records], dtype=np.float64)
        metrics[metric] = values
        comparisons[metric] = compare_groups(
            values,
            selected,
            alternative=alternative,
            reps=int(permutation_reps),
            seed=int(seed) + offset,
        )
        image_comparisons[metric] = image_sign_flip(
            values,
            selected,
            image_indices,
            alternative=alternative,
        )
    updated_records = []
    for row, score, is_rejected, dimension in zip(selected_records, scores, selected, dimensions):
        updated = {
            key: row[key]
            for key in [
                "source",
                "image_index",
                "position",
                "target_code",
                "branch_ks",
                "branch_entropy_norm",
                "full_entropy_norm",
                "nll",
                "top1_prob",
            ]
        }
        updated["code_shell_lrt_score"] = float(score)
        updated["code_rejected"] = bool(is_rejected)
        updated["code_dimension"] = float(dimension)
        updated_records.append(updated)
    image_count = len(set(int(value) for value in image_indices))
    summary = {
        "source": source,
        "num_images": int(image_count),
        "tokens_per_image": int(len(selected_records) // image_count),
        "num_positions": int(len(selected_records)),
        "unique_target_codes": int(np.unique(targets).size),
        "target_rejected_code_fraction": float(selected.mean()),
        "mean_target_code_shell_lrt_score": float(scores.mean()),
        "median_target_code_shell_lrt_score": float(np.median(scores)),
        "mean_target_code_dimension": float(dimensions.mean()),
        "mean_branch_ks": float(metrics["branch_ks"].mean()),
        "mean_branch_entropy_norm": float(metrics["branch_entropy_norm"].mean()),
        "mean_full_entropy_norm": float(metrics["full_entropy_norm"].mean()),
        "mean_nll": float(metrics["nll"].mean()),
        "mean_top1_prob": float(metrics["top1_prob"].mean()),
        "comparisons": comparisons,
        "image_cluster_comparisons": image_comparisons,
    }
    return summary, updated_records


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", int(size))
    except OSError:
        return ImageFont.load_default()


def write_dashboard(
    path: Path,
    *,
    codebook_summary: dict[str, Any],
    arrays: dict[str, np.ndarray],
    real_summary: dict[str, Any],
    generated_summary: dict[str, Any],
    records: list[dict[str, Any]],
    llamagen_summary_path: Path,
) -> None:
    width, height = 1500, 900
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font, panel_font, small_font = load_font(22), load_font(16), load_font(11)
    draw.text((38, 20), "VAR visual tokens: shell likelihood-ratio diagnostics", fill=(18, 28, 36), font=title_font)
    margin, gap, top = 40, 30, 72
    panel_w = (width - 2 * margin - 2 * gap) // 3
    panel_h = (height - top - margin - gap) // 2

    def panel(row: int, col: int, title: str) -> tuple[int, int, int, int]:
        x0 = margin + col * (panel_w + gap)
        y0 = top + row * (panel_h + gap)
        draw.text((x0, y0), title, fill=(18, 28, 36), font=panel_font)
        box = (x0 + 42, y0 + 34, x0 + panel_w - 14, y0 + panel_h - 38)
        draw.line((box[0], box[3], box[2], box[3]), fill=(55, 65, 72), width=2)
        draw.line((box[0], box[1], box[0], box[3]), fill=(55, 65, 72), width=2)
        return box

    scores = arrays["scores"]
    hist_box = panel(0, 0, "VAR codebook deviance scores")
    cap = max(float(np.quantile(scores, 0.995)), 1.05)
    hist, _ = np.histogram(np.clip(scores, 0.0, cap), bins=48, range=(0.0, cap))
    for idx, count in enumerate(hist):
        x0 = hist_box[0] + int(idx / hist.size * (hist_box[2] - hist_box[0]))
        x1 = hist_box[0] + int((idx + 1) / hist.size * (hist_box[2] - hist_box[0])) - 1
        y0 = hist_box[3] - int(count / max(int(hist.max()), 1) * (hist_box[3] - hist_box[1]))
        draw.rectangle((x0, y0, x1, hist_box[3]), fill=(47, 111, 143))
    threshold_x = hist_box[0] + int((hist_box[2] - hist_box[0]) / cap)
    draw.line((threshold_x, hist_box[1], threshold_x, hist_box[3]), fill=(193, 63, 63), width=3)
    draw.text((hist_box[0], hist_box[3] + 8), "T / Monte Carlo critical", fill=(70, 76, 80), font=small_font)

    pca_box = panel(0, 1, "Codebook PCA colored by mismatch")
    pca = arrays["pca"]
    x_min, x_max = float(pca[:, 0].min()), float(pca[:, 0].max())
    y_min, y_max = float(pca[:, 1].min()), float(pca[:, 1].max())
    for (x_value, y_value), score in zip(pca, scores):
        px = pca_box[0] + int((x_value - x_min) / max(x_max - x_min, 1e-9) * (pca_box[2] - pca_box[0]))
        py = pca_box[3] - int((y_value - y_min) / max(y_max - y_min, 1e-9) * (pca_box[3] - pca_box[1]))
        intensity = max(0.0, min(1.0, float(score) / 3.0))
        color = (int(44 + 190 * intensity), int(150 - 75 * intensity), int(150 - 95 * intensity))
        draw.point((px, py), fill=color)

    residual_box = panel(0, 2, "Mean shell-count residual")
    shell_counts = arrays["shell_counts"].astype(np.float64)
    residuals = shell_counts / (shell_counts.sum(axis=1, keepdims=True) / shell_counts.shape[1]) - 1.0
    mean_residual = residuals.mean(axis=0)
    q10, q90 = np.quantile(residuals, [0.10, 0.90], axis=0)
    lo, hi = min(float(q10.min()), 0.0), max(float(q90.max()), 0.0)
    span = max(hi - lo, 1e-9)
    zero_y = residual_box[3] - int((0.0 - lo) / span * (residual_box[3] - residual_box[1]))
    draw.line((residual_box[0], zero_y, residual_box[2], zero_y), fill=(55, 65, 72), width=1)
    points = []
    for idx in range(shell_counts.shape[1]):
        px = residual_box[0] + int((idx + 0.5) / shell_counts.shape[1] * (residual_box[2] - residual_box[0]))
        y_low = residual_box[3] - int((q10[idx] - lo) / span * (residual_box[3] - residual_box[1]))
        y_high = residual_box[3] - int((q90[idx] - lo) / span * (residual_box[3] - residual_box[1]))
        py = residual_box[3] - int((mean_residual[idx] - lo) / span * (residual_box[3] - residual_box[1]))
        draw.line((px, y_high, px, y_low), fill=(225, 181, 73), width=8)
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(126, 79, 0))
        draw.text((px - 3, residual_box[3] + 7), str(idx + 1), fill=(70, 76, 80), font=small_font)
        points.append((px, py))
    draw.line(points, fill=(126, 79, 0), width=3)

    comparison_box = panel(1, 0, "Cross-pipeline rejection rates")
    llama = json.loads(llamagen_summary_path.read_text(encoding="utf-8"))["codebook_shell_test"]
    names = ["ImageNet patches", "LlamaGen VQ", "VAR VQ"]
    values = [0.33035714285714285, float(llama["shell_lrt_reject_fraction"]), float(codebook_summary["shell_lrt_reject_fraction"])]
    colors = [(47, 111, 143), (240, 129, 33), (67, 160, 71)]
    bar_w = 82
    for idx, (name, value, color) in enumerate(zip(names, values, colors)):
        x0 = comparison_box[0] + 32 + idx * 125
        y0 = comparison_box[3] - int(value * (comparison_box[3] - comparison_box[1]))
        draw.rectangle((x0, y0, x0 + bar_w, comparison_box[3]), fill=color)
        draw.text((x0 + 5, y0 - 18), f"{value:.3f}", fill=(18, 28, 36), font=small_font)
        draw.text((x0, comparison_box[3] + 8), name, fill=(70, 76, 80), font=small_font)

    shifts_box = panel(1, 1, "Generator shifts at rejected codes")
    metrics = ["branch_ks", "branch_entropy_norm", "full_entropy_norm", "nll", "top1_prob"]
    short = ["branch KS", "branch H", "full H", "NLL", "top-1"]
    real = np.asarray([real_summary["comparisons"][metric]["selected_minus_rest"] for metric in metrics])
    generated = np.asarray([generated_summary["comparisons"][metric]["selected_minus_rest"] for metric in metrics])
    scale = max(float(np.max(np.abs(np.concatenate([real, generated])))), 1e-9)
    zero_y = (shifts_box[1] + shifts_box[3]) // 2
    draw.line((shifts_box[0], zero_y, shifts_box[2], zero_y), fill=(55, 65, 72), width=1)
    for idx, label in enumerate(short):
        center = shifts_box[0] + int((idx + 0.5) / len(short) * (shifts_box[2] - shifts_box[0]))
        for offset, value, color in [(-9, real[idx], (47, 111, 143)), (9, generated[idx], (211, 72, 65))]:
            end_y = zero_y - int(value / scale * (shifts_box[3] - shifts_box[1]) * 0.44)
            draw.rectangle((center + offset - 7, min(zero_y, end_y), center + offset + 7, max(zero_y, end_y)), fill=color)
        draw.text((center - 20, shifts_box[3] + 8), label, fill=(70, 76, 80), font=small_font)

    usage_box = panel(1, 2, "Code mismatch versus token usage")
    targets = np.asarray([int(row["target_code"]) for row in records], dtype=np.int64)
    usage = np.bincount(targets, minlength=scores.size)
    y_values = np.log1p(usage)
    x_cap = max(float(np.quantile(scores, 0.995)), 1.05)
    y_cap = max(float(y_values.max()), 1e-9)
    for score, usage_value, is_rejected in zip(scores, y_values, arrays["rejected"]):
        px = usage_box[0] + int(min(float(score), x_cap) / x_cap * (usage_box[2] - usage_box[0]))
        py = usage_box[3] - int(float(usage_value) / y_cap * (usage_box[3] - usage_box[1]))
        color = (211, 72, 65) if is_rejected else (91, 163, 199)
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
    threshold_x = usage_box[0] + int((usage_box[2] - usage_box[0]) / x_cap)
    draw.line((threshold_x, usage_box[1], threshold_x, usage_box[3]), fill=(193, 63, 63), width=2)
    draw.text((usage_box[0], usage_box[3] + 8), "deviance ratio; y=log(1+usage)", fill=(70, 76, 80), font=small_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = Path(args.source_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    old_summary = json.loads((source_dir / "var_shell_summary.json").read_text(encoding="utf-8"))
    records = json.loads((source_dir / "var_shell_position_records.json").read_text(encoding="utf-8"))
    codebook_summary, arrays = analyze_cached_codebook(
        source_dir / "var_codebook_shell_arrays.npz",
        neighbors=int(args.neighbors),
        bins=int(args.bins),
        alpha=float(args.alpha),
        calibration_trials=int(args.calibration_trials),
        seed=int(args.seed),
    )
    real_summary, real_records = summarize_positions(
        records,
        source="imagenet_validation",
        codebook_arrays=arrays,
        permutation_reps=int(args.permutation_reps),
        seed=int(args.seed) + 100,
    )
    generated_summary, generated_records = summarize_positions(
        records,
        source="var_generated",
        codebook_arrays=arrays,
        permutation_reps=int(args.permutation_reps),
        seed=int(args.seed) + 200,
    )
    updated_records = real_records + generated_records
    dashboard_path = out_dir / "var_shell_lrt_dashboard.png"
    write_dashboard(
        dashboard_path,
        codebook_summary=codebook_summary,
        arrays=arrays,
        real_summary=real_summary,
        generated_summary=generated_summary,
        records=updated_records,
        llamagen_summary_path=Path(args.llamagen_summary),
    )
    generated_source = source_dir / "var_generated_samples.png"
    generated_destination = out_dir / "var_generated_samples.png"
    if generated_source.exists():
        shutil.copy2(generated_source, generated_destination)
    summary = {
        "analysis": "cached_var_shell_lrt",
        "source_dir": str(source_dir),
        "model": old_summary["model"],
        "codebook_shell_test": codebook_summary,
        "imagenet_positions": real_summary,
        "generated_positions": generated_summary,
        "image_records": old_summary.get("image_records", {}),
        "figures": {
            "dashboard": str(dashboard_path),
            "generated_samples": str(generated_destination),
        },
        "artifacts": {
            "summary": str(out_dir / "var_shell_lrt_summary.json"),
            "position_records": str(out_dir / "var_shell_lrt_position_records.json"),
            "codebook_arrays": str(out_dir / "var_shell_lrt_codebook_arrays.npz"),
        },
    }
    np.savez_compressed(out_dir / "var_shell_lrt_codebook_arrays.npz", **arrays)
    (out_dir / "var_shell_lrt_position_records.json").write_text(json.dumps(updated_records, indent=2), encoding="utf-8")
    (out_dir / "var_shell_lrt_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--llamagen-summary", default="runs/local/vq_gpt_shell_visualizations/20260730_shell_lrt/vq_gpt_shell_summary.json")
    parser.add_argument("--neighbors", type=int, default=128)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--calibration-trials", type=int, default=50000)
    parser.add_argument("--permutation-reps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
