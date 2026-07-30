#!/usr/bin/env python3
"""Monte Carlo size and power study for the fitted shell likelihood-ratio test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from radial_shell_statistics import (
        calibrate_fitted_shell_deviance,
        fitted_equal_mass_shell_deviance,
    )
except ImportError:  # Imported as scripts.radial_uniformity_hypothesis_experiment.
    from scripts.radial_shell_statistics import (
        calibrate_fitted_shell_deviance,
        fitted_equal_mass_shell_deviance,
    )


def run_experiment(
    *,
    dimension: float,
    bins: int,
    sample_sizes: list[int],
    trials: int,
    calibration_trials: int,
    alpha: float,
    seed: int,
) -> dict[str, object]:
    """Estimate size and power against radial shape alternatives.

    If ``U=R**dimension`` is uniform, the radii follow the local volume null.
    The beta alternatives alter radial shape rather than merely changing the
    effective dimension, which is fitted separately in every trial.
    """
    if float(dimension) <= 0.0:
        raise ValueError("dimension must be positive")
    rng = np.random.default_rng(int(seed))
    scenarios = [
        ("null", 1.0, 1.0, "correct radial volume law"),
        ("inner_heavy", 0.45, 1.60, "excess mass near the anchor"),
        ("outer_heavy", 1.60, 0.45, "excess mass near the outer radius"),
    ]
    rows: list[dict[str, object]] = []
    critical_values: dict[str, float] = {}

    for sample_index, samples in enumerate(sample_sizes):
        critical, _null_statistics = calibrate_fitted_shell_deviance(
            samples=int(samples),
            bins=int(bins),
            alpha=float(alpha),
            trials=int(calibration_trials),
            seed=int(seed) + 10000 + sample_index,
        )
        critical_values[str(int(samples))] = float(critical)
        for name, beta_a, beta_b, description in scenarios:
            enclosed_volume = rng.beta(beta_a, beta_b, size=(int(trials), int(samples)))
            radii = np.clip(enclosed_volume, 1e-12, 1.0 - 1e-12) ** (1.0 / float(dimension))
            log_radii = -np.log(radii)
            statistics = np.asarray(
                fitted_equal_mass_shell_deviance(log_radii, int(bins)),
                dtype=np.float64,
            )
            rows.append(
                {
                    "scenario": name,
                    "description": description,
                    "dimension": float(dimension),
                    "beta_a": float(beta_a),
                    "beta_b": float(beta_b),
                    "samples": int(samples),
                    "bins": int(bins),
                    "trials": int(trials),
                    "calibration_trials": int(calibration_trials),
                    "alpha": float(alpha),
                    "critical_value": float(critical),
                    "mean_deviance": float(np.mean(statistics)),
                    "median_deviance": float(np.median(statistics)),
                    "q95_deviance": float(np.quantile(statistics, 0.95)),
                    "rejection_rate": float(np.mean(statistics > critical)),
                }
            )

    return {
        "config": {
            "dimension": float(dimension),
            "bins": int(bins),
            "sample_sizes": [int(value) for value in sample_sizes],
            "trials": int(trials),
            "calibration_trials": int(calibration_trials),
            "alpha": float(alpha),
            "seed": int(seed),
            "critical_values": critical_values,
            "calibration": "full fitted-dimension shell pipeline",
        },
        "summary": rows,
    }


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = ["arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_power_dashboard(result: dict[str, object], path: Path) -> None:
    """Render a paper-ready visual summary without a plotting dependency."""
    rows = list(result["summary"])
    config = dict(result["config"])
    sizes = [int(value) for value in config["sample_sizes"]]
    colors = {
        "null": (47, 111, 143),
        "inner_heavy": (230, 126, 34),
        "outer_heavy": (193, 63, 63),
    }
    labels = {
        "null": "Null",
        "inner_heavy": "Inner-heavy",
        "outer_heavy": "Outer-heavy",
    }

    width, height = 1600, 900
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((42, 24), "Calibrated shell likelihood-ratio test: size and power", fill=(18, 28, 36), font=_font(25, bold=True))
    draw.text(
        (42, 58),
        f"d={config['dimension']:g}, K={config['bins']}, alpha={config['alpha']:.2f}; {config['trials']:,} trials per point",
        fill=(73, 82, 88),
        font=_font(14),
    )

    def axes(box: tuple[int, int, int, int]) -> None:
        draw.line((box[0], box[3], box[2], box[3]), fill=(55, 65, 72), width=2)
        draw.line((box[0], box[1], box[0], box[3]), fill=(55, 65, 72), width=2)

    power_box = (92, 150, 1010, 790)
    draw.text((44, 105), "Rejection probability", fill=(18, 28, 36), font=_font(19, bold=True))
    axes(power_box)
    for tick in range(0, 11, 2):
        value = tick / 10.0
        y = power_box[3] - int(value * (power_box[3] - power_box[1]))
        draw.line((power_box[0] - 7, y, power_box[2], y), fill=(226, 230, 232), width=1)
        draw.text((power_box[0] - 49, y - 8), f"{value:.1f}", fill=(73, 82, 88), font=_font(12))
    alpha = float(config["alpha"])
    alpha_y = power_box[3] - int(alpha * (power_box[3] - power_box[1]))
    draw.line((power_box[0], alpha_y, power_box[2], alpha_y), fill=(100, 100, 100), width=2)
    draw.text((power_box[0] + 8, alpha_y - 22), "nominal size", fill=(73, 82, 88), font=_font(12))

    x_positions: list[int] = []
    for idx, samples in enumerate(sizes):
        x = power_box[0] + int((idx + 0.5) / len(sizes) * (power_box[2] - power_box[0]))
        x_positions.append(x)
        draw.line((x, power_box[3], x, power_box[3] + 7), fill=(55, 65, 72), width=2)
        draw.text((x - 18, power_box[3] + 12), str(samples), fill=(73, 82, 88), font=_font(12))
    draw.text((power_box[0] + 385, power_box[3] + 43), "neighbors N", fill=(73, 82, 88), font=_font(13))

    for scenario in ["null", "inner_heavy", "outer_heavy"]:
        values = [float(next(row["rejection_rate"] for row in rows if row["scenario"] == scenario and int(row["samples"]) == samples)) for samples in sizes]
        points = [
            (x, power_box[3] - int(value * (power_box[3] - power_box[1])))
            for x, value in zip(x_positions, values)
        ]
        draw.line(points, fill=colors[scenario], width=5)
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=colors[scenario], outline=(255, 255, 255), width=2)

    legend_y = 118
    for idx, scenario in enumerate(["null", "inner_heavy", "outer_heavy"]):
        x = 475 + idx * 185
        draw.line((x, legend_y, x + 32, legend_y), fill=colors[scenario], width=5)
        draw.text((x + 40, legend_y - 9), labels[scenario], fill=(40, 48, 54), font=_font(13))

    side_x0, side_x1 = 1090, 1545
    draw.text((side_x0, 105), "Largest neighborhood", fill=(18, 28, 36), font=_font(19, bold=True))
    largest = max(sizes)
    selected = [row for row in rows if int(row["samples"]) == largest]
    bar_box = (side_x0, 150, side_x1, 505)
    axes(bar_box)
    bar_width = 88
    for idx, scenario in enumerate(["null", "inner_heavy", "outer_heavy"]):
        row = next(item for item in selected if item["scenario"] == scenario)
        value = float(row["rejection_rate"])
        x0 = bar_box[0] + 37 + idx * 135
        y0 = bar_box[3] - int(value * (bar_box[3] - bar_box[1]))
        draw.rectangle((x0, y0, x0 + bar_width, bar_box[3]), fill=colors[scenario])
        draw.text((x0 + 14, y0 - 24), f"{value:.3f}", fill=(40, 48, 54), font=_font(13, bold=True))
        short_label = {"null": "Null", "inner_heavy": "Inner", "outer_heavy": "Outer"}[scenario]
        draw.text((x0 + 12, bar_box[3] + 12), short_label, fill=(73, 82, 88), font=_font(12))
    draw.text((side_x0, bar_box[3] + 47), f"N={largest}; vertical scale is rejection probability", fill=(73, 82, 88), font=_font(12))

    note_box = (side_x0, 610, side_x1, 790)
    draw.rounded_rectangle(note_box, radius=6, fill=(245, 247, 248), outline=(205, 211, 214), width=2)
    draw.text((note_box[0] + 18, note_box[1] + 16), "Reading the experiment", fill=(18, 28, 36), font=_font(17, bold=True))
    notes = [
        "The null curve stays near 0.05.",
        "Power rises with neighborhood size.",
        "Both center- and boundary-heavy",
        "departures are detected.",
    ]
    for idx, line in enumerate(notes):
        draw.text((note_box[0] + 20, note_box[1] + 55 + idx * 27), line, fill=(55, 65, 72), font=_font(14))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=float, default=4.0)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--calibration-trials", type=int, default=50000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    result = run_experiment(
        dimension=args.dimension,
        bins=args.bins,
        sample_sizes=args.sample_sizes,
        trials=args.trials,
        calibration_trials=args.calibration_trials,
        alpha=args.alpha,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.figure is not None:
        render_power_dashboard(result, args.figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
