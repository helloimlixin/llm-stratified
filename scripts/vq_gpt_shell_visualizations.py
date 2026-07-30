#!/usr/bin/env python3
"""VQ-VAE + GPT-style vision-generation radial-uniformity summaries.

This script connects the local volume null to cached VQ generative-model
artifacts. It runs an analytic log-radius exponentiality test on the LlamaGen VQ codebook
neighborhoods, then joins those code-level p-values to ImageNet
patch-token records from the matched VQ-tokenizer + GPT-style AR model.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]


def fit_radial_dimension(distances: np.ndarray, outer_radius: float | None = None) -> float:
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
    q = np.linspace(0.0, 1.0, int(bins) + 1, dtype=np.float64)
    return float(radius) * q ** (1.0 / float(dimension))


def shell_counts(distances: np.ndarray, edges: np.ndarray) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    bins = edges.size - 1
    idx = np.searchsorted(edges, distances, side="right") - 1
    idx = np.clip(idx, 0, bins - 1)
    return np.bincount(idx, minlength=bins).astype(np.int64)


def kl_to_uniform(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        return float("nan")
    q = counts / total
    bins = int(q.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(q > 0.0, q * (np.log(q) + math.log(bins)), 0.0)
    return float(np.sum(terms))


EXPONENTIAL_AD_CRITICALS = {
    0.15: 0.922,
    0.10: 1.078,
    0.05: 1.341,
    0.025: 1.606,
    0.01: 1.957,
}


def exponential_ad_critical(samples: int, alpha: float = 0.05) -> float:
    if int(samples) < 2:
        return float("nan")
    if float(alpha) not in EXPONENTIAL_AD_CRITICALS:
        raise ValueError(f"alpha must be one of {sorted(EXPONENTIAL_AD_CRITICALS)}")
    return float(EXPONENTIAL_AD_CRITICALS[float(alpha)] / (1.0 + 0.6 / int(samples)))


def exponential_ad_statistic(distances: np.ndarray, radius: float) -> float:
    values = np.asarray(distances, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0) & (values < float(radius))]
    if values.size < 2 or radius <= 0.0:
        return float("nan")
    log_radii = np.log(float(radius) / values)
    scale = float(np.mean(log_radii))
    if not math.isfinite(scale) or scale <= 0.0:
        return float("nan")
    standardized = np.sort(log_radii / scale)
    cdf = np.clip(1.0 - np.exp(-standardized), 1e-12, 1.0 - 1e-12)
    weights = 2.0 * np.arange(1, values.size + 1, dtype=np.float64) - 1.0
    terms = weights * (np.log(cdf) + np.log1p(-cdf[::-1]))
    return float(-values.size - np.sum(terms) / values.size)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def bool_array(records: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([bool(row.get(key, False)) for row in records], dtype=bool)


def finite_array(records: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([safe_float(row.get(key)) for row in records], dtype=np.float64)


def mean_diff(values: np.ndarray, selector: np.ndarray) -> float:
    finite = np.isfinite(values)
    a = values[finite & selector]
    b = values[finite & ~selector]
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(a.mean() - b.mean())


def permutation_p(values: np.ndarray, selector: np.ndarray, *, alternative: str, reps: int, seed: int) -> float:
    observed = mean_diff(values, selector)
    if not math.isfinite(observed):
        return float("nan")
    finite = np.isfinite(values)
    values = values[finite]
    labels = selector[finite].copy()
    rng = np.random.default_rng(int(seed))
    null = np.empty(int(reps), dtype=np.float64)
    for idx in range(int(reps)):
        rng.shuffle(labels)
        null[idx] = mean_diff(values, labels)
    if alternative == "higher":
        extreme = int(np.sum(null >= observed))
    elif alternative == "lower":
        extreme = int(np.sum(null <= observed))
    else:
        raise ValueError(alternative)
    return float((extreme + 1.0) / (int(reps) + 1.0))


def compare_groups(values: np.ndarray, selector: np.ndarray, *, alternative: str, reps: int, seed: int) -> dict[str, Any]:
    finite = np.isfinite(values)
    a = values[finite & selector]
    b = values[finite & ~selector]
    return {
        "count_selected": int((finite & selector).sum()),
        "count_rest": int((finite & ~selector).sum()),
        "mean_selected": float(np.mean(a)) if a.size else float("nan"),
        "mean_rest": float(np.mean(b)) if b.size else float("nan"),
        "median_selected": float(np.median(a)) if a.size else float("nan"),
        "median_rest": float(np.median(b)) if b.size else float("nan"),
        "diff_selected_minus_rest": mean_diff(values, selector),
        "permutation_p": permutation_p(values, selector, alternative=alternative, reps=reps, seed=seed),
        "alternative": alternative,
    }


def codebook_shell_tests(
    *,
    neighbors_npz: Path,
    codebook_records_path: Path,
    neighbors: int,
    bins: int,
    alpha: float,
    permutation_reps: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    packed = np.load(neighbors_npz)
    distances = np.asarray(packed["distances"], dtype=np.float64)
    codebook_records = read_json(codebook_records_path)
    if distances.ndim != 2:
        raise ValueError("neighbor distances must be a 2D array")
    if len(codebook_records) != int(distances.shape[0]):
        raise ValueError("codebook records and neighbor array must have the same row count")
    if int(neighbors) >= int(distances.shape[1]):
        raise ValueError("neighbors must leave room for the self-neighbor column")

    rows: list[dict[str, Any]] = []
    for code_id in range(int(distances.shape[0])):
        local = distances[code_id, 1:int(neighbors) + 1]
        local = local[np.isfinite(local) & (local > 0.0)]
        if local.size != int(neighbors):
            dim_hat = stat = score = critical = radius = float("nan")
            counts = np.zeros(int(bins), dtype=np.int64)
        else:
            radius = float(local[-1])
            inner = local[:-1]
            dim_hat = fit_radial_dimension(inner, radius)
            critical = exponential_ad_critical(inner.size, float(alpha))
            if math.isfinite(dim_hat) and dim_hat > 0.0:
                counts = shell_counts(inner, equal_mass_edges(dim_hat, int(bins), radius))
                stat = exponential_ad_statistic(inner, radius)
                score = float(stat / critical) if math.isfinite(stat) and critical > 0.0 else float("nan")
            else:
                counts = np.zeros(int(bins), dtype=np.int64)
                stat = score = float("nan")
        source = codebook_records[code_id]
        rows.append(
            {
                "code_id": int(code_id),
                "dimension_hat": float(dim_hat),
                "radius": float(radius),
                "shell_counts": counts.astype(int).tolist(),
                "ad_statistic": float(stat),
                "ad_critical": float(critical),
                "ad_score": float(score),
                "reject": bool(math.isfinite(score) and score > 1.0),
                "large_fiber_rejected": bool(source.get("large_fiber_rejected", False)),
                "large_manifold_rejected": bool(source.get("large_manifold_rejected", False)),
                "small_fiber_rejected": bool(source.get("small_fiber_rejected", False)),
                "small_manifold_rejected": bool(source.get("small_manifold_rejected", False)),
                "singular_any": bool(source.get("singular_any", False)),
                "singular_fiber_any": bool(source.get("singular_fiber_any", False)),
                "best_adjusted_pvalue": safe_float(source.get("best_adjusted_pvalue")),
            }
        )

    ad_statistics = finite_array(rows, "ad_statistic")
    ad_scores = finite_array(rows, "ad_score")
    dims = finite_array(rows, "dimension_hat")
    reject = bool_array(rows, "reject")
    shell_score = ad_scores
    large_fiber = bool_array(rows, "large_fiber_rejected")
    large_manifold = bool_array(rows, "large_manifold_rejected")
    singular_any = bool_array(rows, "singular_any")
    summary = {
        "neighbors_npz": str(neighbors_npz),
        "codebook_records": str(codebook_records_path),
        "num_codes": int(len(rows)),
        "neighbors": int(neighbors),
        "bins": int(bins),
        "alpha": float(alpha),
        "ad_critical": exponential_ad_critical(int(neighbors) - 1, float(alpha)),
        "reject_count": int(reject.sum()),
        "reject_fraction": float(reject.mean()),
        "mean_dimension_hat": float(np.nanmean(dims)),
        "median_dimension_hat": float(np.nanmedian(dims)),
        "ad_statistic_quantiles": {
            "q50": float(np.nanquantile(ad_statistics, 0.50)),
            "q90": float(np.nanquantile(ad_statistics, 0.90)),
            "q95": float(np.nanquantile(ad_statistics, 0.95)),
            "q99": float(np.nanquantile(ad_statistics, 0.99)),
        },
        "group_comparisons": {
            "large_fiber_shell_score": compare_groups(
                shell_score,
                large_fiber,
                alternative="higher",
                reps=int(permutation_reps),
                seed=int(seed) + 101,
            ),
            "large_manifold_shell_score": compare_groups(
                shell_score,
                large_manifold,
                alternative="higher",
                reps=int(permutation_reps),
                seed=int(seed) + 202,
            ),
            "singular_any_shell_score": compare_groups(
                shell_score,
                singular_any,
                alternative="higher",
                reps=int(permutation_reps),
                seed=int(seed) + 303,
            ),
        },
        "overlap": {
            "large_fiber_count": int(large_fiber.sum()),
            "large_fiber_shell_reject_count": int(np.sum(large_fiber & reject)),
            "large_fiber_shell_reject_rate": float(np.mean(reject[large_fiber])) if large_fiber.any() else float("nan"),
            "rest_shell_reject_rate": float(np.mean(reject[~large_fiber])) if (~large_fiber).any() else float("nan"),
            "singular_any_count": int(singular_any.sum()),
            "singular_any_shell_reject_count": int(np.sum(singular_any & reject)),
        },
    }
    return rows, summary


def ar_join_summary(
    *,
    ar_records_path: Path,
    codebook_shell_records: list[dict[str, Any]],
    permutation_reps: int,
    seed: int,
) -> dict[str, Any]:
    records = read_json(ar_records_path)
    by_code = {int(row["code_id"]): row for row in codebook_shell_records}
    shell_score = np.empty(len(records), dtype=np.float64)
    shell_reject = np.zeros(len(records), dtype=bool)
    shell_dim = np.empty(len(records), dtype=np.float64)
    shell_score.fill(np.nan)
    shell_dim.fill(np.nan)
    for idx, row in enumerate(records):
        code = int(row.get("target_code", -1))
        hit = by_code.get(code)
        if hit is None:
            continue
        shell_score[idx] = safe_float(hit.get("ad_score"))
        shell_dim[idx] = safe_float(hit.get("dimension_hat"))
        shell_reject[idx] = bool(hit.get("reject", False))
    large_fiber = bool_array(records, "codebook_target_large_fiber")
    metric_specs = {
        "local_ball_ks": "lower",
        "local_ball_entropy": "higher",
        "branch_ks": "lower",
        "branch_entropy": "higher",
        "ranked_ks": "lower",
    }
    large_fiber_metrics = {}
    shell_reject_metrics = {}
    for offset, (metric, alternative) in enumerate(metric_specs.items()):
        values = finite_array(records, metric)
        large_fiber_metrics[metric] = compare_groups(
            values,
            large_fiber,
            alternative=alternative,
            reps=int(permutation_reps),
            seed=int(seed) + 1000 + offset,
        )
        shell_reject_metrics[metric] = compare_groups(
            values,
            shell_reject,
            alternative=alternative,
            reps=int(permutation_reps),
            seed=int(seed) + 2000 + offset,
        )

    return {
        "ar_records": str(ar_records_path),
        "num_positions": int(len(records)),
        "num_images": int(len(set(int(row.get("sample_id", -1)) for row in records))),
        "codebook_target_large_fiber_count": int(large_fiber.sum()),
        "codebook_target_large_fiber_fraction": float(large_fiber.mean()),
        "target_shell_reject_count": int(shell_reject.sum()),
        "target_shell_reject_fraction": float(shell_reject.mean()),
        "mean_target_shell_score": float(np.nanmean(shell_score)),
        "median_target_shell_dimension": float(np.nanmedian(shell_dim)),
        "large_fiber_metric_comparisons": large_fiber_metrics,
        "shell_reject_metric_comparisons": shell_reject_metrics,
    }


def rgba_for_score(value: float, vmax: float) -> tuple[int, int, int, int]:
    t = 0.0 if not math.isfinite(value) else max(0.0, min(1.0, float(value) / max(float(vmax), 1e-9)))
    if t < 0.5:
        u = t / 0.5
        r = int(35 + u * 240)
        g = int(120 + u * 190)
        b = int(190 - u * 150)
    else:
        u = (t - 0.5) / 0.5
        r = int(240 - u * 30)
        g = int(190 - u * 135)
        b = int(40 + u * 10)
    return r, g, b, int(35 + 145 * t)


def overlay_grid(image: Image.Image, values: np.ndarray, *, vmax: float) -> Image.Image:
    base = image.convert("RGBA")
    w, h = base.size
    grid = int(values.shape[0])
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for gy in range(grid):
        for gx in range(grid):
            x0 = int(round(gx * w / grid))
            y0 = int(round(gy * h / grid))
            x1 = int(round((gx + 1) * w / grid))
            y1 = int(round((gy + 1) * h / grid))
            draw.rectangle((x0, y0, x1, y1), fill=rgba_for_score(float(values[gy, gx]), vmax))
            draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 255, 70), width=1)
    return Image.alpha_composite(base, layer).convert("RGB")


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", int(size))
    except OSError:
        return ImageFont.load_default()


def write_heatmap_gallery(
    *,
    path: Path,
    dataset_records_path: Path,
    ar_records_path: Path,
    codebook_shell_records: list[dict[str, Any]],
    image_count: int,
    image_size: int,
) -> str:
    dataset_records = read_json(dataset_records_path)
    ar_records = read_json(ar_records_path)
    by_code = {int(row["code_id"]): row for row in codebook_shell_records}
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in ar_records:
        grouped.setdefault(int(row.get("sample_id", -1)), []).append(row)
    example_count = min(int(image_count), len(dataset_records), len(grouped))
    examples_per_row = 2
    gallery_rows = int(math.ceil(example_count / examples_per_row))
    panels_per_example = 3
    label_h = 24
    panel_gap = 8
    group_gap = 24
    margin = 12
    group_width = panels_per_example * int(image_size) + (panels_per_example - 1) * panel_gap
    canvas = Image.new(
        "RGB",
        (
            margin * 2 + examples_per_row * group_width + (examples_per_row - 1) * group_gap,
            margin * 2 + gallery_rows * (int(image_size) + label_h) + (gallery_rows - 1) * group_gap,
        ),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    font = load_font(11)
    header_font = load_font(12)
    for example_idx in range(example_count):
        record = dataset_records[example_idx]
        img_path = Path(record["path"])
        image = ImageOps.fit(Image.open(img_path).convert("RGB"), (int(image_size), int(image_size)), method=Image.Resampling.BICUBIC)
        patches = sorted(grouped.get(example_idx, []), key=lambda item: int(item.get("patch_id", 0)))
        grid = int(round(math.sqrt(len(patches)))) if patches else 16
        shell_values = np.zeros((grid, grid), dtype=np.float64)
        entropy_values = np.zeros((grid, grid), dtype=np.float64)
        for item in patches[:grid * grid]:
            patch = int(item.get("patch_id", 0))
            gy, gx = divmod(patch, grid)
            code = int(item.get("target_code", -1))
            hit = by_code.get(code, {})
            shell_values[gy, gx] = min(safe_float(hit.get("ad_score")), 6.0)
            entropy_values[gy, gx] = safe_float(item.get("local_ball_entropy"))
        entropy_min = float(np.nanmin(entropy_values)) if np.isfinite(entropy_values).any() else 0.0
        entropy_span = max(float(np.nanmax(entropy_values) - entropy_min), 1e-9) if np.isfinite(entropy_values).any() else 1.0
        entropy_norm = (entropy_values - entropy_min) / entropy_span
        panels = [
            ("input", image),
            ("radial score", overlay_grid(image, shell_values, vmax=6.0)),
            ("AR entropy", overlay_grid(image, entropy_norm, vmax=1.0)),
        ]
        group_row = example_idx // examples_per_row
        group_col = example_idx % examples_per_row
        group_x = margin + group_col * (group_width + group_gap)
        y = margin + group_row * (int(image_size) + label_h + group_gap)
        for col_idx, (title, panel) in enumerate(panels):
            x = group_x + col_idx * (int(image_size) + panel_gap)
            canvas.paste(panel, (x, y + label_h))
            draw.text((x, y), title, fill=(20, 20, 20), font=header_font if group_row == 0 else font)
        class_text = f"{example_idx}: class {record.get('class_label')}"
        class_y = y + label_h + int(image_size) - 14
        class_box = draw.textbbox((group_x + 2, class_y), class_text, font=font)
        draw.rectangle((class_box[0] - 2, class_box[1] - 1, class_box[2] + 2, class_box[3] + 1), fill=(255, 255, 255))
        draw.text((group_x + 2, class_y), class_text, fill=(20, 20, 20), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return str(path)


def resize_panel(path: Path, width: int) -> Image.Image | None:
    if not path or not Path(path).exists():
        return None
    img = Image.open(path).convert("RGB")
    ratio = width / max(img.width, 1)
    return img.resize((int(width), max(1, int(round(img.height * ratio)))), Image.Resampling.LANCZOS)


def write_model_gallery(
    *,
    path: Path,
    vq_dataset_summary_path: Path,
    imagegpt_summary_path: Path,
    var_summary_path: Path,
) -> str:
    summaries = {
        "LlamaGen sources": read_json(vq_dataset_summary_path).get("figures", {}).get("sources"),
        "LlamaGen VQ reconstructions": read_json(vq_dataset_summary_path).get("figures", {}).get("reconstructions"),
        "VQ-VAE/ImageGPT inputs": read_json(imagegpt_summary_path).get("outputs", {}).get("inputs"),
        "VQ-VAE/ImageGPT samples": read_json(imagegpt_summary_path).get("outputs", {}).get("samples"),
        "Pretrained VAR samples": read_json(var_summary_path).get("outputs", {}).get("sample_grid"),
    }
    panel_w = 420
    gap = 18
    label_h = 28
    panels: list[tuple[str, Image.Image]] = []
    for title, rel in summaries.items():
        if rel is None:
            continue
        img = resize_panel(REPO_ROOT / rel if not Path(rel).is_absolute() else Path(rel), panel_w)
        if img is not None:
            panels.append((title, img))
    if not panels:
        raise ValueError("no model-gallery panels were found")
    rows = len(panels)
    height = sum(img.height + label_h for _title, img in panels) + gap * (rows - 1) + 24
    canvas = Image.new("RGB", (panel_w + 24, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = load_font(14)
    y = 12
    for title, img in panels:
        draw.text((12, y), title, fill=(20, 20, 20), font=font)
        canvas.paste(img, (12, y + label_h))
        y += label_h + img.height + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return str(path)


def write_dashboard(
    *,
    path: Path,
    codebook_records: list[dict[str, Any]],
    ar_summary: dict[str, Any],
    paired_stats_path: Path | None,
    pca_path: Path | None,
) -> str:
    ad_scores = finite_array(codebook_records, "ad_score")
    dims = finite_array(codebook_records, "dimension_hat")
    reject = bool_array(codebook_records, "reject")
    large_fiber = bool_array(codebook_records, "large_fiber_rejected")
    shell_score = ad_scores
    width, height = 1500, 900
    margin = 34
    gap = 28
    title_h = 44
    panel_w = (width - 2 * margin - 2 * gap) // 3
    panel_h = (height - margin - title_h - gap - margin) // 2
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(20)
    panel_font = load_font(15)
    small_font = load_font(11)
    draw.text((margin, 14), "VQ-VAE + GPT-style radial-uniformity results", fill=(20, 20, 20), font=title_font)

    def panel_rect(row: int, col: int) -> tuple[int, int, int, int]:
        x0 = margin + col * (panel_w + gap)
        y0 = title_h + margin + row * (panel_h + gap)
        return x0, y0, x0 + panel_w, y0 + panel_h

    def frame(rect: tuple[int, int, int, int], title: str) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = rect
        draw.text((x0, y0), title, fill=(20, 20, 20), font=panel_font)
        plot = (x0 + 46, y0 + 32, x1 - 14, y1 - 34)
        draw.rectangle(plot, outline=(210, 210, 210), width=1)
        return plot

    def draw_hist(rect: tuple[int, int, int, int], values: np.ndarray, *, bins: int, color: tuple[int, int, int], xlabel: str, vline: float | None = None) -> None:
        x0, y0, x1, y1 = rect
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return
        lo, hi = float(np.min(finite)), float(np.max(finite))
        if hi <= lo:
            hi = lo + 1.0
        counts, edges = np.histogram(finite, bins=int(bins), range=(lo, hi))
        max_count = max(int(np.max(counts)), 1)
        for idx, count in enumerate(counts):
            bx0 = x0 + int(idx * (x1 - x0) / len(counts))
            bx1 = x0 + int((idx + 1) * (x1 - x0) / len(counts)) - 1
            by1 = y1
            by0 = y1 - int((count / max_count) * (y1 - y0))
            draw.rectangle((bx0, by0, bx1, by1), fill=color)
        if vline is not None and lo <= float(vline) <= hi:
            vx = x0 + int((float(vline) - lo) / (hi - lo) * (x1 - x0))
            draw.line((vx, y0, vx, y1), fill=(220, 60, 60), width=2)
        draw.text((x0, y1 + 8), xlabel, fill=(70, 70, 70), font=small_font)
        draw.text((x0, y0 - 14), f"n={finite.size}", fill=(70, 70, 70), font=small_font)

    def draw_bars(rect: tuple[int, int, int, int], labels: list[str], values: list[float], *, colors: list[tuple[int, int, int]], ylabel: str, zero_line: bool = True) -> None:
        x0, y0, x1, y1 = rect
        finite_vals = [v for v in values if math.isfinite(v)]
        if not finite_vals:
            return
        if zero_line:
            lo = min(0.0, min(finite_vals))
            hi = max(0.0, max(finite_vals))
            pad = max((hi - lo) * 0.14, 1e-6)
            lo -= pad
            hi += pad
        else:
            lo = 0.0
            hi = max(finite_vals)
            hi += max(hi * 0.14, 1e-6)
        span = max(hi - lo, 1e-9)
        zero_y = y1 - int((0.0 - lo) / span * (y1 - y0))
        if zero_line:
            draw.line((x0, zero_y, x1, zero_y), fill=(40, 40, 40), width=1)
        bar_gap = 10
        bar_w = max(8, int((x1 - x0 - bar_gap * (len(values) + 1)) / max(len(values), 1)))
        for idx, value in enumerate(values):
            if not math.isfinite(value):
                continue
            bx0 = x0 + bar_gap + idx * (bar_w + bar_gap)
            bx1 = bx0 + bar_w
            by = y1 - int((value - lo) / span * (y1 - y0))
            draw.rectangle((bx0, min(by, zero_y), bx1, max(by, zero_y)), fill=colors[idx % len(colors)])
            draw.text((bx0, y1 + 5), labels[idx][:13], fill=(70, 70, 70), font=small_font)
            draw.text((bx0, min(by, zero_y) - 14), f"{value:.3g}", fill=(20, 20, 20), font=small_font)
        draw.text((x0, y0 - 14), ylabel, fill=(70, 70, 70), font=small_font)

    def draw_pca(rect: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = rect
        if not pca_path or not Path(pca_path).exists():
            draw.text((x0 + 8, y0 + 8), "PCA file missing", fill=(80, 80, 80), font=small_font)
            return
        coords = np.load(pca_path)
        xs, ys = coords[:, 0], coords[:, 1]
        xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
        ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
        xspan = max(xmax - xmin, 1e-9)
        yspan = max(ymax - ymin, 1e-9)
        for idx in range(coords.shape[0]):
            px = x0 + int((xs[idx] - xmin) / xspan * (x1 - x0 - 1))
            py = y1 - int((ys[idx] - ymin) / yspan * (y1 - y0 - 1))
            r, g, b, _a = rgba_for_score(float(shell_score[idx]), 6.0)
            draw.rectangle((px, py, px + 1, py + 1), fill=(r, g, b))
        draw.text((x0, y1 + 8), "color: AD statistic / 5% critical value", fill=(70, 70, 70), font=small_font)

    draw_hist(frame(panel_rect(0, 0), "Codebook Anderson-Darling scores"), ad_scores, bins=50, color=(76, 120, 168), xlabel="AD statistic / 5% critical value", vline=1.0)
    draw_pca(frame(panel_rect(0, 1), "VQ codebook PCA by radial score"))
    rates = [
        float(np.mean(reject[large_fiber])) if large_fiber.any() else float("nan"),
        float(np.mean(reject[~large_fiber])) if (~large_fiber).any() else float("nan"),
        float(np.mean(reject)),
    ]
    draw_bars(
        frame(panel_rect(0, 2), "Radial-null rejection rate"),
        ["large fiber", "rest", "all"],
        rates,
        colors=[(245, 133, 24), (158, 202, 233), (84, 162, 75)],
        ylabel="fraction rejected",
        zero_line=False,
    )
    draw_hist(frame(panel_rect(1, 0), "Fitted radial dimensions"), dims, bins=50, color=(114, 183, 178), xlabel="dimension estimate")
    metrics = ar_summary.get("large_fiber_metric_comparisons", {})
    names = ["local_ball_ks", "local_ball_entropy", "branch_ks", "branch_entropy", "ranked_ks"]
    metric_values = [float(metrics.get(name, {}).get("diff_selected_minus_rest", np.nan)) for name in names]
    draw_bars(
        frame(panel_rect(1, 1), "AR metric shifts for target large-fiber codes"),
        ["local KS", "local H", "branch KS", "branch H", "ranked KS"],
        metric_values,
        colors=[(76, 120, 168) if v < 0 else (245, 133, 24) for v in metric_values],
        ylabel="selected minus rest",
        zero_line=True,
    )
    paired_values: list[float] = []
    paired_labels: list[str] = []
    if paired_stats_path and Path(paired_stats_path).exists():
        paired = read_json(Path(paired_stats_path)).get("results", [])
        for row in paired:
            paired_labels.append(str(row.get("name", "")))
            paired_values.append(safe_float(row.get("mean_diff")))
    draw_bars(
        frame(panel_rect(1, 2), "Decoded branch diversity"),
        paired_labels or ["missing"],
        paired_values or [float("nan")],
        colors=[(245, 133, 24), (76, 120, 168)],
        ylabel="mean paired crop-L2 diff",
        zero_line=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return str(path)


def write_report(path: Path, summary: dict[str, Any]) -> str:
    codebook = summary["codebook_shell_test"]
    ar = summary["ar_join"]
    paired = summary.get("paired_branch_stats", {})
    lines = [
        "# VQ-VAE + GPT Radial-Uniformity Results",
        "",
        f"- VQ codebook codes tested: `{codebook['num_codes']}`",
        f"- Anderson-Darling rejection fraction: `{codebook['reject_fraction']:.4f}`",
        f"- Mean/median fitted radial dimension: `{codebook['mean_dimension_hat']:.4f}` / `{codebook['median_dimension_hat']:.4f}`",
        f"- Large-fiber code radial rejection rate: `{codebook['overlap']['large_fiber_shell_reject_rate']:.4f}`",
        f"- Rest code radial rejection rate: `{codebook['overlap']['rest_shell_reject_rate']:.4f}`",
        f"- ImageNet AR positions joined: `{ar['num_positions']}` over `{ar['num_images']}` images",
        f"- Target positions using radial-null-rejected VQ codes: `{ar['target_shell_reject_fraction']:.4f}`",
    ]
    results = paired.get("results", [])
    for row in results:
        lines.append(
            f"- Paired branch diversity `{row.get('name')}`: mean diff `{safe_float(row.get('mean_diff')):.6f}`, "
            f"wins `{row.get('wins')}/{row.get('n_pairs')}`, sign-flip p `{safe_float(row.get('paired_sign_flip_p_one_sided')):.4g}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    codebook_records, codebook_summary = codebook_shell_tests(
        neighbors_npz=Path(args.neighbors_npz).resolve(),
        codebook_records_path=Path(args.codebook_records).resolve(),
        neighbors=int(args.neighbors),
        bins=int(args.bins),
        alpha=float(args.alpha),
        permutation_reps=int(args.permutation_reps),
        seed=int(args.seed),
    )
    ar_summary = ar_join_summary(
        ar_records_path=Path(args.ar_records).resolve(),
        codebook_shell_records=codebook_records,
        permutation_reps=int(args.permutation_reps),
        seed=int(args.seed) + 5000,
    )
    paired_stats = read_json(Path(args.paired_stats)) if args.paired_stats and Path(args.paired_stats).exists() else {}
    figures = {
        "dashboard": write_dashboard(
            path=out_dir / "vq_gpt_shell_dashboard.png",
            codebook_records=codebook_records,
            ar_summary=ar_summary,
            paired_stats_path=Path(args.paired_stats).resolve() if args.paired_stats else None,
            pca_path=Path(args.codebook_pca).resolve() if args.codebook_pca else None,
        ),
        "imagenet_heatmaps": write_heatmap_gallery(
            path=out_dir / "vq_gpt_imagenet_shell_heatmap_gallery.png",
            dataset_records_path=Path(args.dataset_records).resolve(),
            ar_records_path=Path(args.ar_records).resolve(),
            codebook_shell_records=codebook_records,
            image_count=int(args.gallery_images),
            image_size=int(args.gallery_image_size),
        ),
        "model_gallery": write_model_gallery(
            path=out_dir / "vq_gpt_model_gallery.png",
            vq_dataset_summary_path=Path(args.vq_dataset_summary).resolve(),
            imagegpt_summary_path=Path(args.imagegpt_summary).resolve(),
            var_summary_path=Path(args.var_summary).resolve(),
        ),
    }
    summary = {
        "analysis": "vq_gpt_shell_visualizations",
        "out_dir": str(out_dir),
        "codebook_shell_test": codebook_summary,
        "ar_join": ar_summary,
        "paired_branch_stats": paired_stats,
        "figures": figures,
        "artifacts": {
            "codebook_shell_records": str(out_dir / "vq_gpt_codebook_shell_records.json"),
            "summary": str(out_dir / "vq_gpt_shell_summary.json"),
            "report": str(out_dir / "vq_gpt_shell_report.md"),
        },
    }
    (out_dir / "vq_gpt_codebook_shell_records.json").write_text(json.dumps(codebook_records, indent=2), encoding="utf-8")
    (out_dir / "vq_gpt_shell_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out_dir / "vq_gpt_shell_report.md", summary)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    default_out = REPO_ROOT / "runs" / "local" / "vq_gpt_shell_visualizations" / f"{stamp}_llamagen_vq_codebook_shell"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--neighbors-npz",
        default=REPO_ROOT / "runs/local/pretrained_vq_codebook/llamagen_c2i_B_256_codebook_stratification/vq_codebook_neighbors.npz",
    )
    parser.add_argument(
        "--codebook-records",
        default=REPO_ROOT / "runs/local/pretrained_vq_codebook/llamagen_c2i_B_256_codebook_stratification/vq_codebook_records.json",
    )
    parser.add_argument(
        "--codebook-pca",
        default=REPO_ROOT / "runs/local/pretrained_vq_codebook/llamagen_c2i_B_256_codebook_stratification/vq_codebook_pca.npy",
    )
    parser.add_argument(
        "--ar-records",
        default=REPO_ROOT / "runs/local/pretrained_vq_ar_ks/llamagen_c2i_B_256_imagenet_val64_codebook_target_large_fiber_controls/vq_ar_ks_tokens.json",
    )
    parser.add_argument(
        "--dataset-records",
        default=REPO_ROOT / "runs/local/pretrained_vq_ar/llamagen_c2i_B_256_imagenet_val256_seed20260628/llamagen_c2i_dataset_records.json",
    )
    parser.add_argument(
        "--vq-dataset-summary",
        default=REPO_ROOT / "runs/local/pretrained_vq_ar/llamagen_c2i_B_256_imagenet_val256_seed20260628/llamagen_c2i_dataset_summary.json",
    )
    parser.add_argument(
        "--paired-stats",
        default=REPO_ROOT / "runs/local/pretrained_vq_ar_polysemy_branch_gallery/paired_inference_position_guardrail/vq_ar_polysemy_branch_paired_stats.json",
    )
    parser.add_argument(
        "--imagegpt-summary",
        default=REPO_ROOT / "runs/local/vqvae_imagegpt/docs_gallery_fullscale_smoke_20260627/vqvae_imagegpt_summary.json",
    )
    parser.add_argument(
        "--var-summary",
        default=REPO_ROOT / "runs/local/pretrained_var/d16_cpu_smoke_20260627/pretrained_var_summary.json",
    )
    parser.add_argument("--out-dir", default=default_out)
    parser.add_argument("--neighbors", type=int, default=128)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--permutation-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--gallery-images", type=int, default=8)
    parser.add_argument("--gallery-image-size", type=int, default=128)
    return parser


def main() -> None:
    summary = run(build_argparser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
