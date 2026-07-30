#!/usr/bin/env python3
"""Visualize local radial-uniformity rejections on ImageNet validation images.

This script makes the local volume null concrete by running it on patch-level
input-image features. It crops ImageNet-val images to 224x224, extracts a 14x14
grid of 16x16 patches, represents each patch by a small downsampled RGB vector,
and tests each patch anchor's transformed neighbor radii against Uniform(0, 1)
under a fitted local d-dimensional ball null.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from radial_shell_statistics import (
        calibrate_fitted_shell_deviance,
        fitted_shell_test_from_distances,
        monte_carlo_pvalue,
    )
except ImportError:  # Imported as scripts.imagenet_shell_visualizations.
    from scripts.radial_shell_statistics import (
        calibrate_fitted_shell_deviance,
        fitted_shell_test_from_distances,
        monte_carlo_pvalue,
    )


def fit_radial_dimension(distances: np.ndarray, outer_radius: float | None = None) -> float:
    """MLE for F(s)=(s/r)^d using distances within a fixed outer radius."""
    values = np.asarray(distances, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return float("nan")
    radius = float(np.max(values) if outer_radius is None else outer_radius)
    if not math.isfinite(radius) or radius <= 0.0:
        return float("nan")
    scaled = np.clip(values / radius, 1e-12, 1.0)
    denom = -float(np.sum(np.log(scaled)))
    if not math.isfinite(denom) or denom <= 1e-12:
        return float("nan")
    return float(values.size / denom)


def equal_mass_edges(dimension: float, bins: int, radius: float) -> np.ndarray:
    if dimension <= 0.0 or bins <= 0 or radius <= 0.0:
        raise ValueError("dimension, bins, and radius must be positive")
    quantiles = np.linspace(0.0, 1.0, int(bins) + 1, dtype=np.float64)
    return float(radius) * quantiles ** (1.0 / float(dimension))


def shell_counts(distances: np.ndarray, edges: np.ndarray) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    bins = edges.size - 1
    indices = np.searchsorted(edges, distances, side="right") - 1
    indices = np.clip(indices, 0, bins - 1)
    return np.bincount(indices, minlength=bins).astype(np.int64)


def kl_to_uniform(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        return float("nan")
    q = counts / total
    bins = q.size
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(q > 0.0, q * (np.log(q) + math.log(bins)), 0.0)
    return float(np.sum(terms))


def read_label_rows(labels_csv: Path, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(labels_csv).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("exists", "True")).lower() not in {"true", "1", "yes"}:
                continue
            rows.append(row)
            if len(rows) >= int(limit):
                break
    return rows


def load_image(path: Path, image_size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.fit(image, (int(image_size), int(image_size)), method=Image.Resampling.BICUBIC)


def patch_features(image: Image.Image, *, patch_size: int, feature_size: int) -> tuple[np.ndarray, list[Image.Image]]:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    grid = int(arr.shape[0] // patch_size)
    features = []
    patches = []
    for gy in range(grid):
        for gx in range(grid):
            y0 = gy * patch_size
            x0 = gx * patch_size
            patch = image.crop((x0, y0, x0 + patch_size, y0 + patch_size))
            small = patch.resize((int(feature_size), int(feature_size)), Image.Resampling.BICUBIC)
            small_arr = np.asarray(small, dtype=np.float32) / 255.0
            features.append(small_arr.reshape(-1))
            patches.append(patch)
    return np.asarray(features, dtype=np.float32), patches


def pairwise_distances(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    norms = np.sum(x * x, axis=1, keepdims=True)
    d2 = np.maximum(norms + norms.T - 2.0 * (x @ x.T), 0.0)
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(d2, out=d2)


def run_shell_tests(
    *,
    features: np.ndarray,
    neighbors: int,
    bins: int,
    alpha: float,
    calibration_trials: int,
    calibration_seed: int,
) -> list[dict[str, Any]]:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    x = (features - mean) / np.maximum(std, 1e-6)
    dists = pairwise_distances(x)
    n = int(dists.shape[0])
    k = min(int(neighbors), max(1, n - 1))
    nn_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
    nn_dists = np.take_along_axis(dists, nn_idx, axis=1)
    order = np.argsort(nn_dists, axis=1)
    nn_idx = np.take_along_axis(nn_idx, order, axis=1)
    nn_dists = np.take_along_axis(nn_dists, order, axis=1)
    tested = int(k) - 1
    critical, null_statistics = calibrate_fitted_shell_deviance(
        samples=tested,
        bins=int(bins),
        alpha=float(alpha),
        trials=int(calibration_trials),
        seed=int(calibration_seed),
    )
    records: list[dict[str, Any]] = []
    for anchor in range(n):
        local = nn_dists[anchor]
        radius = float(local[-1])
        inner = local[:-1]
        dim_hat = fit_radial_dimension(inner, radius)
        if not math.isfinite(dim_hat) or dim_hat <= 0.0:
            counts = np.zeros(int(bins), dtype=np.int64)
            stat = float("nan")
            score = float("nan")
            pvalue = float("nan")
        else:
            counts, stat = fitted_shell_test_from_distances(inner, radius=radius, bins=int(bins))
            score = float(stat / critical) if math.isfinite(stat) and critical > 0.0 else float("nan")
            pvalue = monte_carlo_pvalue(stat, null_statistics)
        records.append(
            {
                "anchor": int(anchor),
                "neighbor_indices": nn_idx[anchor].astype(int).tolist(),
                "neighbor_distances": local.astype(float).tolist(),
                "radius": radius,
                "dimension_hat": dim_hat,
                "shell_counts": counts.astype(int).tolist(),
                "shell_lrt_statistic": stat,
                "shell_lrt_critical": critical,
                "shell_lrt_score": score,
                "shell_lrt_pvalue": pvalue,
                "reject": bool(math.isfinite(pvalue) and pvalue <= float(alpha)),
                "neighbors": int(inner.size),
                "bins": int(bins),
                "calibration_trials": int(calibration_trials),
            }
        )
    return records


def score_color(value: float, cap: float) -> tuple[int, int, int, int]:
    t = 0.0 if not math.isfinite(value) else max(0.0, min(1.0, value / cap))
    if t < 0.5:
        u = t / 0.5
        r = int(55 + u * (245 - 55))
        g = int(120 + u * (208 - 120))
        b = int(190 + u * (66 - 190))
    else:
        u = (t - 0.5) / 0.5
        r = int(245 + u * (210 - 245))
        g = int(208 + u * (54 - 208))
        b = int(66 + u * (42 - 66))
    alpha = int(35 + 150 * t)
    return r, g, b, alpha


def image_scores(records: list[dict[str, Any]], *, image_index: int, patches_per_image: int) -> np.ndarray:
    start = int(image_index) * int(patches_per_image)
    stop = start + int(patches_per_image)
    values = []
    for row in records[start:stop]:
        score = float(row.get("shell_lrt_score", float("nan")))
        values.append(score if math.isfinite(score) else 0.0)
    grid = int(math.sqrt(patches_per_image))
    return np.asarray(values, dtype=np.float64).reshape(grid, grid)


def overlay_heatmap(image: Image.Image, scores: np.ndarray, *, patch_size: int, score_cap: float) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    grid = int(scores.shape[0])
    for gy in range(grid):
        for gx in range(grid):
            x0 = gx * patch_size
            y0 = gy * patch_size
            draw.rectangle(
                (x0, y0, x0 + patch_size - 1, y0 + patch_size - 1),
                fill=score_color(float(scores[gy, gx]), score_cap),
            )
    grid_draw = ImageDraw.Draw(overlay)
    for pos in range(0, image.size[0] + 1, patch_size):
        grid_draw.line((pos, 0, pos, image.size[1]), fill=(255, 255, 255, 65), width=1)
        grid_draw.line((0, pos, image.size[0], pos), fill=(255, 255, 255, 65), width=1)
    return Image.alpha_composite(base, overlay).convert("RGB")


def text_font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", int(size))
    except Exception:
        return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, *, font: ImageFont.ImageFont) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle((bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2), fill=(255, 255, 255))
    draw.text((x, y), text, fill=(20, 20, 20), font=font)


def make_heatmap_gallery(
    *,
    images: list[Image.Image],
    labels: list[str],
    records: list[dict[str, Any]],
    out_path: Path,
    patch_size: int,
    score_cap: float,
    columns: int = 4,
) -> None:
    image_size = images[0].size[0]
    panel_w = image_size * 2 + 12
    panel_h = image_size + 42
    rows = int(math.ceil(len(images) / columns))
    canvas = Image.new("RGB", (columns * panel_w, rows * panel_h), (245, 245, 245))
    font = text_font(13)
    patches_per_image = (image_size // patch_size) ** 2
    for idx, image in enumerate(images):
        row = idx // columns
        col = idx % columns
        x = col * panel_w
        y = row * panel_h
        scores = image_scores(records, image_index=idx, patches_per_image=patches_per_image)
        overlay = overlay_heatmap(image, scores, patch_size=patch_size, score_cap=score_cap)
        canvas.paste(image, (x, y + 30))
        canvas.paste(overlay, (x + image_size + 12, y + 30))
        draw = ImageDraw.Draw(canvas)
        label = labels[idx].replace("_", " ")
        reject_count = int(np.sum(scores > 1.0))
        draw_label(draw, (x + 4, y + 6), f"{idx}: {label} | rejects {reject_count}/{patches_per_image}", font=font)
        draw_label(draw, (x + 4, y + image_size + 13), "input", font=font)
        draw_label(draw, (x + image_size + 16, y + image_size + 13), "shell deviance ratio", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def draw_histogram(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], counts: list[int], expected: float) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(255, 255, 255), outline=(190, 190, 190))
    max_y = max(max(counts), expected, 1.0)
    width = x1 - x0
    height = y1 - y0
    bar_gap = 3
    bar_w = max(3, (width - 14 - bar_gap * (len(counts) - 1)) // len(counts))
    baseline = y1 - 18
    top = y0 + 8
    usable_h = baseline - top
    exp_y = baseline - int((expected / max_y) * usable_h)
    draw.line((x0 + 6, exp_y, x1 - 6, exp_y), fill=(35, 35, 35), width=1)
    for idx, count in enumerate(counts):
        bx0 = x0 + 7 + idx * (bar_w + bar_gap)
        bx1 = bx0 + bar_w
        by0 = baseline - int((count / max_y) * usable_h)
        draw.rectangle((bx0, by0, bx1, baseline), fill=(210, 54, 42))
    draw.text((x0 + 7, y1 - 15), "shell counts; line = expected", fill=(20, 20, 20), font=text_font(10))


def make_anchor_examples(
    *,
    images: list[Image.Image],
    labels: list[str],
    patches: list[Image.Image],
    records: list[dict[str, Any]],
    out_path: Path,
    patch_size: int,
    count: int,
    max_per_image: int,
) -> None:
    candidates = sorted(
        [row for row in records if math.isfinite(float(row.get("shell_lrt_score", float("nan"))))],
        key=lambda row: float(row["shell_lrt_score"]),
        reverse=True,
    )
    image_size = images[0].size[0]
    patches_per_image = (image_size // patch_size) ** 2
    image_counts: dict[int, int] = {}
    ranked: list[dict[str, Any]] = []
    for row in candidates:
        image_idx = int(row["anchor"]) // patches_per_image
        used = image_counts.get(image_idx, 0)
        if used >= int(max_per_image):
            continue
        ranked.append(row)
        image_counts[image_idx] = used + 1
        if len(ranked) >= int(count):
            break
    if len(ranked) < int(count):
        seen = {int(row["anchor"]) for row in ranked}
        for row in candidates:
            if int(row["anchor"]) in seen:
                continue
            ranked.append(row)
            if len(ranked) >= int(count):
                break
    row_h = 176
    canvas = Image.new("RGB", (1120, row_h * len(ranked)), (246, 246, 246))
    font = text_font(13)
    small_font = text_font(10)
    grid = image_size // patch_size
    for ridx, row in enumerate(ranked):
        y = ridx * row_h
        anchor = int(row["anchor"])
        image_idx = anchor // patches_per_image
        local_idx = anchor % patches_per_image
        gy = local_idx // grid
        gx = local_idx % grid
        thumb = images[image_idx].resize((144, 144), Image.Resampling.BICUBIC)
        draw_thumb = ImageDraw.Draw(thumb)
        scale = 144 / image_size
        rect = (
            int(gx * patch_size * scale),
            int(gy * patch_size * scale),
            int((gx + 1) * patch_size * scale),
            int((gy + 1) * patch_size * scale),
        )
        draw_thumb.rectangle(rect, outline=(240, 30, 30), width=3)
        canvas.paste(thumb, (10, y + 24))
        draw = ImageDraw.Draw(canvas)
        label = labels[image_idx].replace("_", " ")
        score = float(row["shell_lrt_score"])
        pvalue = float(row["shell_lrt_pvalue"])
        draw.text((10, y + 5), f"{ridx + 1}. {label} | patch ({gx},{gy}) | deviance ratio={score:.2f} | p={pvalue:.3g} | d={float(row['dimension_hat']):.2f}", fill=(20, 20, 20), font=font)
        patch = patches[anchor].resize((72, 72), Image.Resampling.NEAREST)
        canvas.paste(patch, (166, y + 50))
        draw.text((166, y + 30), "anchor crop", fill=(20, 20, 20), font=small_font)
        counts = [int(x) for x in row["shell_counts"]]
        draw_histogram(draw, (252, y + 38, 520, y + 146), counts, expected=float(row["neighbors"]) / max(1, len(counts)))
        nn = [int(x) for x in row["neighbor_indices"][:12]]
        draw.text((542, y + 30), "nearest input patches", fill=(20, 20, 20), font=small_font)
        for nidx, patch_idx in enumerate(nn):
            px = 542 + (nidx % 12) * 46
            py = y + 50
            tile = patches[patch_idx].resize((40, 40), Image.Resampling.NEAREST)
            canvas.paste(tile, (px, py))
            draw.rectangle((px, py, px + 39, py + 39), outline=(210, 210, 210))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_shell_lrt_dashboard(
    *,
    records: list[dict[str, Any]],
    labels: list[str],
    patches_per_image: int,
    out_path: Path,
) -> None:
    scores = np.asarray([float(row["shell_lrt_score"]) for row in records], dtype=np.float64)
    dimensions = np.asarray([float(row["dimension_hat"]) for row in records], dtype=np.float64)
    rejected = np.asarray([bool(row["reject"]) for row in records], dtype=bool)
    counts = np.asarray([row["shell_counts"] for row in records], dtype=np.float64)
    expected = np.asarray([float(row["neighbors"]) / counts.shape[1] for row in records])[:, None]
    residuals = counts / expected - 1.0
    image_rates = rejected.reshape(len(labels), int(patches_per_image)).mean(axis=1)

    width, height = 1500, 900
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = text_font(22)
    panel_font = text_font(16)
    small_font = text_font(11)
    draw.text((38, 20), "ImageNet patch neighborhoods: shell likelihood-ratio diagnostics", fill=(18, 28, 36), font=title_font)

    margin, gap, top = 40, 34, 72
    panel_w = (width - 2 * margin - gap) // 2
    panel_h = (height - top - margin - gap) // 2

    def panel(row: int, col: int, title: str) -> tuple[int, int, int, int]:
        x0 = margin + col * (panel_w + gap)
        y0 = top + row * (panel_h + gap)
        draw.text((x0, y0), title, fill=(18, 28, 36), font=panel_font)
        box = (x0 + 48, y0 + 34, x0 + panel_w - 18, y0 + panel_h - 42)
        draw.line((box[0], box[3], box[2], box[3]), fill=(55, 65, 72), width=2)
        draw.line((box[0], box[1], box[0], box[3]), fill=(55, 65, 72), width=2)
        return box

    hist_box = panel(0, 0, "Normalized shell deviance")
    finite_scores = scores[np.isfinite(scores)]
    score_max = max(float(np.nanquantile(finite_scores, 0.995)), 1.05)
    hist, edges = np.histogram(np.clip(finite_scores, 0.0, score_max), bins=52, range=(0.0, score_max))
    hist_max = max(int(hist.max()), 1)
    for idx, count in enumerate(hist):
        bx0 = hist_box[0] + int(idx * (hist_box[2] - hist_box[0]) / hist.size)
        bx1 = hist_box[0] + int((idx + 1) * (hist_box[2] - hist_box[0]) / hist.size) - 1
        by0 = hist_box[3] - int(count / hist_max * (hist_box[3] - hist_box[1]))
        draw.rectangle((bx0, by0, bx1, hist_box[3]), fill=(47, 111, 143))
    threshold_x = hist_box[0] + int((hist_box[2] - hist_box[0]) / score_max)
    draw.line((threshold_x, hist_box[1], threshold_x, hist_box[3]), fill=(193, 63, 63), width=3)
    draw.text((hist_box[0], hist_box[3] + 9), "T / Monte Carlo critical value", fill=(70, 76, 80), font=small_font)
    draw.text((threshold_x + 5, hist_box[1] + 5), "reject", fill=(193, 63, 63), font=small_font)

    rate_box = panel(0, 1, "Rejection fraction by image")
    order = np.argsort(image_rates)
    bar_gap = 5
    bar_w = max(8, (rate_box[2] - rate_box[0] - bar_gap * (len(labels) + 1)) // len(labels))
    for rank, image_idx in enumerate(order):
        x0 = rate_box[0] + bar_gap + rank * (bar_w + bar_gap)
        y0 = rate_box[3] - int(image_rates[image_idx] * (rate_box[3] - rate_box[1]))
        shade = int(80 + 140 * rank / max(len(labels) - 1, 1))
        draw.rectangle((x0, y0, x0 + bar_w, rate_box[3]), fill=(45, shade, 125))
        draw.text((x0, rate_box[3] + 5), str(image_idx), fill=(70, 76, 80), font=small_font)
    alpha_y = rate_box[3] - int(0.05 * (rate_box[3] - rate_box[1]))
    draw.line((rate_box[0], alpha_y, rate_box[2], alpha_y), fill=(193, 63, 63), width=2)
    legend = "  ".join(f"{idx}:{labels[idx].replace('_', ' ')[:11]}" for idx in range(len(labels)))
    draw.text((rate_box[0], rate_box[3] + 22), legend[:105], fill=(70, 76, 80), font=small_font)

    scatter_box = panel(1, 0, "Dimension versus shell mismatch")
    finite = np.isfinite(scores) & np.isfinite(dimensions)
    x_values = dimensions[finite]
    y_cap = max(float(np.nanquantile(scores[finite], 0.99)), 1.05)
    y_values = np.clip(scores[finite], 0.0, y_cap)
    x_min, x_max = float(x_values.min()), float(x_values.max())
    for x_value, y_value, is_rejected in zip(x_values, y_values, rejected[finite]):
        px = scatter_box[0] + int((x_value - x_min) / max(x_max - x_min, 1e-9) * (scatter_box[2] - scatter_box[0]))
        py = scatter_box[3] - int(y_value / y_cap * (scatter_box[3] - scatter_box[1]))
        color = (211, 72, 65) if is_rejected else (91, 163, 199)
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
    reject_y = scatter_box[3] - int(1.0 / y_cap * (scatter_box[3] - scatter_box[1]))
    draw.line((scatter_box[0], reject_y, scatter_box[2], reject_y), fill=(193, 63, 63), width=2)
    draw.text((scatter_box[0], scatter_box[3] + 9), "fitted local dimension", fill=(70, 76, 80), font=small_font)

    residual_box = panel(1, 1, "Shell-count residual profile")
    mean_residual = np.nanmean(residuals, axis=0)
    q10 = np.nanquantile(residuals, 0.10, axis=0)
    q90 = np.nanquantile(residuals, 0.90, axis=0)
    residual_min = min(float(q10.min()), 0.0)
    residual_max = max(float(q90.max()), 0.0)
    span = max(residual_max - residual_min, 1e-9)
    zero_y = residual_box[3] - int((0.0 - residual_min) / span * (residual_box[3] - residual_box[1]))
    draw.line((residual_box[0], zero_y, residual_box[2], zero_y), fill=(55, 65, 72), width=1)
    points: list[tuple[int, int]] = []
    for idx in range(counts.shape[1]):
        px = residual_box[0] + int((idx + 0.5) / counts.shape[1] * (residual_box[2] - residual_box[0]))
        y10 = residual_box[3] - int((q10[idx] - residual_min) / span * (residual_box[3] - residual_box[1]))
        y90 = residual_box[3] - int((q90[idx] - residual_min) / span * (residual_box[3] - residual_box[1]))
        py = residual_box[3] - int((mean_residual[idx] - residual_min) / span * (residual_box[3] - residual_box[1]))
        draw.line((px, y90, px, y10), fill=(225, 181, 73), width=8)
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(126, 79, 0))
        draw.text((px - 3, residual_box[3] + 7), str(idx + 1), fill=(70, 76, 80), font=small_font)
        points.append((px, py))
    if len(points) > 1:
        draw.line(points, fill=(126, 79, 0), width=3)
    draw.text((residual_box[0], residual_box[3] + 24), "equal-null-mass shell, inner to outer", fill=(70, 76, 80), font=small_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def write_report(path: Path, summary: dict[str, Any], paths: dict[str, str]) -> None:
    lines = [
        "# ImageNet Radial-Uniformity Visualization",
        "",
        "Patch-level multinomial shell likelihood-ratio tests on actual ImageNet validation images.",
        "",
        "## Configuration",
        "",
    ]
    for key in [
        "image_count",
        "image_size",
        "patch_size",
        "feature_size",
        "neighbors",
        "bins",
        "alpha",
    ]:
        lines.append(f"- {key}: `{summary[key]}`")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Patches tested: `{summary['num_patches']}`",
            f"- Rejected patches: `{summary['shell_lrt_reject_count']}`",
            f"- Rejection fraction: `{summary['shell_lrt_reject_fraction']:.4f}`",
            f"- Monte Carlo critical deviance: `{summary['shell_lrt_critical']:.4f}`",
            f"- Mean fitted radial dimension: `{summary['mean_dimension_hat']:.4f}`",
            f"- Median fitted radial dimension: `{summary['median_dimension_hat']:.4f}`",
            "",
            "## Files",
            "",
        ]
    )
    for key, value in paths.items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_label_rows(args.labels_csv, args.images)
    if not rows:
        raise ValueError("no ImageNet validation rows found")
    images: list[Image.Image] = []
    labels: list[str] = []
    feature_parts = []
    patch_tiles: list[Image.Image] = []
    for row in rows:
        image = load_image(args.image_dir / row["path"], args.image_size)
        feats, patches = patch_features(image, patch_size=args.patch_size, feature_size=args.feature_size)
        images.append(image)
        labels.append(str(row.get("class_name", row.get("label", "unknown"))))
        feature_parts.append(feats)
        patch_tiles.extend(patches)
    features = np.concatenate(feature_parts, axis=0)
    records = run_shell_tests(
        features=features,
        neighbors=args.neighbors,
        bins=args.bins,
        alpha=args.alpha,
        calibration_trials=args.calibration_trials,
        calibration_seed=args.seed,
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = out_dir / "imagenet_shell_heatmap_gallery.png"
    examples_path = out_dir / "imagenet_shell_anchor_examples.png"
    dashboard_path = out_dir / "imagenet_shell_lrt_dashboard.png"
    records_path = out_dir / "imagenet_shell_patch_records.json"
    summary_path = out_dir / "imagenet_shell_summary.json"
    report_path = out_dir / "imagenet_shell_report.md"

    make_heatmap_gallery(
        images=images[: args.gallery_images],
        labels=labels[: args.gallery_images],
        records=records,
        out_path=heatmap_path,
        patch_size=args.patch_size,
        score_cap=args.score_cap,
        columns=args.columns,
    )
    make_anchor_examples(
        images=images,
        labels=labels,
        patches=patch_tiles,
        records=records,
        out_path=examples_path,
        patch_size=args.patch_size,
        count=args.examples,
        max_per_image=args.example_max_per_image,
    )
    patches_per_image = (int(args.image_size) // int(args.patch_size)) ** 2
    make_shell_lrt_dashboard(
        records=records,
        labels=labels,
        patches_per_image=patches_per_image,
        out_path=dashboard_path,
    )
    dims = np.asarray([float(row["dimension_hat"]) for row in records], dtype=np.float64)
    statistics = np.asarray([float(row["shell_lrt_statistic"]) for row in records], dtype=np.float64)
    scores = np.asarray([float(row["shell_lrt_score"]) for row in records], dtype=np.float64)
    reject = np.asarray([bool(row["reject"]) for row in records], dtype=bool)
    summary = {
        "image_count": int(len(images)),
        "image_size": int(args.image_size),
        "patch_size": int(args.patch_size),
        "feature_size": int(args.feature_size),
        "neighbors": int(min(args.neighbors, features.shape[0] - 1)),
        "bins": int(args.bins),
        "alpha": float(args.alpha),
        "calibration_trials": int(args.calibration_trials),
        "calibration_seed": int(args.seed),
        "num_patches": int(features.shape[0]),
        "shell_lrt_reject_count": int(np.sum(reject)),
        "shell_lrt_reject_fraction": float(np.mean(reject)),
        "mean_dimension_hat": float(np.nanmean(dims)),
        "median_dimension_hat": float(np.nanmedian(dims)),
        "shell_lrt_statistic_quantiles": {
            "q50": float(np.nanquantile(statistics, 0.50)),
            "q90": float(np.nanquantile(statistics, 0.90)),
            "q95": float(np.nanquantile(statistics, 0.95)),
            "q99": float(np.nanquantile(statistics, 0.99)),
        },
        "shell_lrt_score_quantiles": {
            "q50": float(np.nanquantile(scores, 0.50)),
            "q90": float(np.nanquantile(scores, 0.90)),
            "q95": float(np.nanquantile(scores, 0.95)),
            "q99": float(np.nanquantile(scores, 0.99)),
        },
        "shell_lrt_critical": float(records[0]["shell_lrt_critical"]),
        "classes": labels,
    }
    records_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    paths = {
        "heatmap_gallery": str(heatmap_path),
        "anchor_examples": str(examples_path),
        "dashboard": str(dashboard_path),
        "records_json": str(records_path),
        "summary_json": str(summary_path),
        "report_md": str(report_path),
    }
    summary_path.write_text(json.dumps({**summary, "paths": paths}, indent=2), encoding="utf-8")
    write_report(report_path, summary, paths)
    return {**summary, "paths": paths}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = Path("C:/Users/hello/Projects/data/imagenet_val")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--image-dir", type=Path, default=default_data / "images")
    parser.add_argument("--labels-csv", type=Path, default=default_data / "imagenet_val_labels.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/local/imagenet_shell_visualizations") / stamp)
    parser.add_argument("--images", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--feature-size", type=int, default=4)
    parser.add_argument("--neighbors", type=int, default=96)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--calibration-trials", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--score-cap", type=float, default=4.0)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--gallery-images", type=int, default=6)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--example-max-per-image", type=int, default=1)
    return parser


def main() -> None:
    summary = run(build_argparser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
