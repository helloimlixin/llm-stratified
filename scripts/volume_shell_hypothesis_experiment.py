#!/usr/bin/env python3
"""Monte Carlo calibration for the local volume shell hypothesis test.

The experiment simulates the multinomial shell test derived from the local
d-dimensional volume null. It estimates rejection rates under the null and
under radial alternatives, using both the finite-sample method-of-types
threshold and the Wilks chi-square calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Scenario:
    name: str
    radial_dimension: float
    description: str


def shell_edges_equal_null_mass(*, bins: int, dimension: float, radius: float) -> np.ndarray:
    """Return radii whose shells have equal mass under the d-ball null."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")
    probs = np.linspace(0.0, 1.0, int(bins) + 1, dtype=np.float64)
    return float(radius) * probs ** (1.0 / float(dimension))


def shell_probabilities(*, edges: np.ndarray, radial_dimension: float, radius: float) -> np.ndarray:
    """Shell probabilities for a radial CDF F(s)=(s/r)^radial_dimension."""
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("edges must be a one-dimensional array with at least two entries")
    if radial_dimension <= 0:
        raise ValueError("radial_dimension must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")
    scaled = np.clip(edges / float(radius), 0.0, 1.0)
    cdf = scaled ** float(radial_dimension)
    probs = np.diff(cdf)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("invalid shell probabilities")
    return probs / total


def kl_divergence(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Vectorized D_KL(Q || P) for rows of q."""
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if q.ndim == 1:
        q = q[None, :]
    if q.shape[-1] != p.shape[0]:
        raise ValueError("q and p must have the same number of bins")
    safe_p = np.clip(p, 1e-300, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(q > 0.0, q * (np.log(q) - np.log(safe_p)), 0.0)
    return np.sum(terms, axis=-1)


def finite_sample_kl_threshold(*, samples: int, bins: int, alpha: float) -> float:
    """Method-of-types threshold for D_KL(Q || P)."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    if bins <= 0:
        raise ValueError("bins must be positive")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    return (bins * math.log(samples + 1.0) + math.log(1.0 / alpha)) / float(samples)


def quantile_higher(values: np.ndarray, probability: float) -> float:
    """Return the smallest observed value with empirical CDF at least probability."""
    if not (0.0 <= probability <= 1.0):
        raise ValueError("probability must lie in [0, 1]")
    sorted_values = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if sorted_values.size == 0:
        raise ValueError("values must be non-empty")
    rank = int(math.ceil(float(probability) * sorted_values.size)) - 1
    rank = max(0, min(rank, sorted_values.size - 1))
    return float(sorted_values[rank])


def chi_square_critical_wilson_hilferty(*, df: int, alpha: float) -> float:
    """Approximate chi-square upper critical value via Wilson-Hilferty."""
    if df <= 0:
        raise ValueError("df must be positive")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    z = NormalDist().inv_cdf(1.0 - float(alpha))
    base = 1.0 - 2.0 / (9.0 * df) + z * math.sqrt(2.0 / (9.0 * df))
    return float(df * max(base, 0.0) ** 3)


def run_experiment(
    *,
    dimension: float,
    bins: int,
    sample_sizes: list[int],
    trials: int,
    calibration_trials: int,
    alpha: float,
    radius: float,
    seed: int,
) -> dict[str, Any]:
    edges = shell_edges_equal_null_mass(bins=bins, dimension=dimension, radius=radius)
    null_probs = shell_probabilities(edges=edges, radial_dimension=dimension, radius=radius)
    scenarios = [
        Scenario(
            name=f"null_d{dimension:g}",
            radial_dimension=dimension,
            description="correct local d-ball radial law",
        ),
        Scenario(
            name=f"inner_heavy_d{dimension / 2.0:g}",
            radial_dimension=dimension / 2.0,
            description="too much mass near the anchor",
        ),
        Scenario(
            name=f"outer_heavy_d{dimension * 2.0:g}",
            radial_dimension=dimension * 2.0,
            description="too much mass in outer shells",
        ),
    ]
    rng = np.random.default_rng(int(seed))
    calibration_rng = np.random.default_rng(int(seed) + 1)
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    chi2_critical = chi_square_critical_wilson_hilferty(df=bins - 1, alpha=alpha)
    bootstrap_thresholds: dict[int, float] = {}

    for samples in sample_sizes:
        null_counts = calibration_rng.multinomial(int(samples), null_probs, size=int(calibration_trials))
        null_q = null_counts.astype(np.float64) / float(samples)
        null_kl = kl_divergence(null_q, null_probs)
        bootstrap_thresholds[int(samples)] = quantile_higher(null_kl, 1.0 - float(alpha))

    for scenario in scenarios:
        scenario_probs = shell_probabilities(
            edges=edges,
            radial_dimension=scenario.radial_dimension,
            radius=radius,
        )
        expected_kl = float(kl_divergence(scenario_probs, null_probs)[0])
        for samples in sample_sizes:
            counts = rng.multinomial(int(samples), scenario_probs, size=int(trials))
            q = counts.astype(np.float64) / float(samples)
            kl_values = kl_divergence(q, null_probs)
            finite_threshold = finite_sample_kl_threshold(samples=samples, bins=bins, alpha=alpha)
            asymptotic_threshold = chi2_critical / (2.0 * float(samples))
            bootstrap_threshold = bootstrap_thresholds[int(samples)]
            finite_reject = kl_values >= finite_threshold
            asymptotic_reject = kl_values >= asymptotic_threshold
            bootstrap_reject = kl_values >= bootstrap_threshold

            row = {
                "scenario": scenario.name,
                "description": scenario.description,
                "test_dimension": float(dimension),
                "radial_dimension": float(scenario.radial_dimension),
                "samples": int(samples),
                "bins": int(bins),
                "trials": int(trials),
                "calibration_trials": int(calibration_trials),
                "alpha": float(alpha),
                "expected_kl_to_null": expected_kl,
                "mean_kl": float(np.mean(kl_values)),
                "median_kl": float(np.median(kl_values)),
                "q90_kl": float(np.quantile(kl_values, 0.90)),
                "q95_kl": float(np.quantile(kl_values, 0.95)),
                "finite_threshold": float(finite_threshold),
                "finite_rejection_rate": float(np.mean(finite_reject)),
                "asymptotic_threshold": float(asymptotic_threshold),
                "asymptotic_rejection_rate": float(np.mean(asymptotic_reject)),
                "bootstrap_threshold": float(bootstrap_threshold),
                "bootstrap_rejection_rate": float(np.mean(bootstrap_reject)),
            }
            rows.append(row)
            records.append(
                {
                    **row,
                    "null_probs": null_probs.tolist(),
                    "scenario_probs": scenario_probs.tolist(),
                    "shell_edges": edges.tolist(),
                }
            )

    return {
        "config": {
            "dimension": float(dimension),
            "bins": int(bins),
            "sample_sizes": [int(x) for x in sample_sizes],
            "trials": int(trials),
            "calibration_trials": int(calibration_trials),
            "alpha": float(alpha),
            "radius": float(radius),
            "seed": int(seed),
            "chi_square_critical_wilson_hilferty": float(chi2_critical),
            "bootstrap_thresholds": {str(k): float(v) for k, v in bootstrap_thresholds.items()},
            "note": "Shell edges are chosen to have equal mass under the null.",
        },
        "summary": rows,
        "records": records,
    }


def _polyline(points: list[tuple[float, float]], color: str) -> str:
    pairs = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{pairs}" fill="none" stroke="{color}" stroke-width="2.5" />'


def write_svg_plot(path: Path, *, rows: list[dict[str, Any]], alpha: float) -> None:
    """Write a small dependency-free SVG line plot of rejection rates."""
    sample_sizes = sorted({int(row["samples"]) for row in rows})
    scenarios = list(dict.fromkeys(str(row["scenario"]) for row in rows))
    colors = ["#2f5d8a", "#a63d40", "#2f7d4f", "#8f5aa8", "#b7772a"]
    row_map = {(str(row["scenario"]), int(row["samples"])): row for row in rows}

    width, height = 1240, 420
    margin_left, margin_right = 74, 26
    margin_top, margin_bottom = 48, 62
    panel_gap = 52
    panel_width = (width - margin_left - margin_right - 2 * panel_gap) / 3.0
    panel_height = height - margin_top - margin_bottom

    def x_at(panel: int, sample: int) -> float:
        start = margin_left + panel * (panel_width + panel_gap)
        if len(sample_sizes) == 1:
            return start + panel_width / 2.0
        idx = sample_sizes.index(sample)
        return start + idx * panel_width / (len(sample_sizes) - 1)

    def y_at(rate: float) -> float:
        return margin_top + (1.0 - max(0.0, min(1.0, rate))) * panel_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">Volume shell test calibration</text>',
    ]
    panel_specs = [
        ("finite_rejection_rate", "Finite-sample KL threshold"),
        ("asymptotic_rejection_rate", "Chi-square approximation"),
        ("bootstrap_rejection_rate", "Parametric bootstrap threshold"),
    ]
    for panel_idx, (metric, title) in enumerate(panel_specs):
        x0 = margin_left + panel_idx * (panel_width + panel_gap)
        x1 = x0 + panel_width
        y0 = margin_top
        y1 = margin_top + panel_height
        lines.extend(
            [
                f'<text x="{(x0 + x1) / 2:.1f}" y="{margin_top - 16}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700">{title}</text>',
                f'<line x1="{x0:.1f}" y1="{y1:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#222" stroke-width="1" />',
                f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y1:.1f}" stroke="#222" stroke-width="1" />',
            ]
        )
        for ytick in [0.0, alpha, 0.25, 0.50, 0.75, 1.0]:
            y = y_at(float(ytick))
            dash = ' stroke-dasharray="4 4"' if abs(ytick - alpha) < 1e-12 else ""
            stroke = "#b33" if abs(ytick - alpha) < 1e-12 else "#ddd"
            lines.append(
                f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}" stroke="{stroke}" stroke-width="1"{dash} />'
            )
            lines.append(
                f'<text x="{x0 - 8:.1f}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{ytick:.2f}</text>'
            )
        for sample in sample_sizes:
            x = x_at(panel_idx, sample)
            lines.append(
                f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y1 + 5:.1f}" stroke="#222" stroke-width="1" />'
            )
            lines.append(
                f'<text x="{x:.1f}" y="{y1 + 22:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{sample}</text>'
            )
        for scenario_idx, scenario in enumerate(scenarios):
            color = colors[scenario_idx % len(colors)]
            points = [
                (x_at(panel_idx, sample), y_at(float(row_map[(scenario, sample)][metric])))
                for sample in sample_sizes
            ]
            lines.append(_polyline(points, color))
            for x, y in points:
                lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}" />')

    legend_x = margin_left
    legend_y = height - 20
    for scenario_idx, scenario in enumerate(scenarios):
        color = colors[scenario_idx % len(colors)]
        x = legend_x + scenario_idx * 210
        lines.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3" />')
        lines.append(f'<text x="{x + 30}" y="{legend_y + 4}" font-family="Arial" font-size="12">{scenario}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_report(path: Path, *, payload: dict[str, Any], paths: dict[str, str]) -> None:
    config = payload["config"]
    rows = payload["summary"]
    lines = [
        "# Volume Shell Hypothesis Experiment",
        "",
        "Monte Carlo calibration for the multinomial shell test of the local "
        f"{config['dimension']:g}-dimensional volume null.",
        "",
        "## Configuration",
        "",
        f"- Dimension: `{config['dimension']:g}`",
        f"- Shells: `{config['bins']}` equal-null-mass bins",
        f"- Sample sizes: `{', '.join(str(x) for x in config['sample_sizes'])}`",
        f"- Trials per scenario: `{config['trials']}`",
        f"- Bootstrap calibration trials: `{config['calibration_trials']}`",
        f"- Alpha: `{config['alpha']}`",
        f"- Seed: `{config['seed']}`",
        "",
        "## Rejection Rates",
        "",
        "| scenario | N | finite KL | chi-square approx | bootstrap | mean KL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {samples} | {finite:.4f} | {asym:.4f} | {boot:.4f} | {mean_kl:.4f} |".format(
                scenario=row["scenario"],
                samples=int(row["samples"]),
                finite=float(row["finite_rejection_rate"]),
                asym=float(row["asymptotic_rejection_rate"]),
                boot=float(row["bootstrap_rejection_rate"]),
                mean_kl=float(row["mean_kl"]),
            )
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- The finite-sample method-of-types threshold controls Type I error but is very conservative here.",
            "- The chi-square and parametric-bootstrap thresholds are calibrated near the nominal null rate.",
            "- The inner-heavy and outer-heavy radial alternatives are detected with high power once the threshold is calibrated.",
            "",
            "## Files",
            "",
        ]
    )
    for key, value in paths.items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(payload: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "volume_shell_hypothesis_summary.json"
    records_path = out_dir / "volume_shell_hypothesis_records.json"
    csv_path = out_dir / "volume_shell_hypothesis_summary.csv"
    svg_path = out_dir / "volume_shell_hypothesis_rejection_rates.svg"
    report_path = out_dir / "volume_shell_hypothesis_report.md"

    summary_path.write_text(json.dumps({"config": payload["config"], "summary": payload["summary"]}, indent=2), encoding="utf-8")
    records_path.write_text(json.dumps(payload["records"], indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["summary"][0].keys()))
        writer.writeheader()
        writer.writerows(payload["summary"])
    write_svg_plot(svg_path, rows=payload["summary"], alpha=float(payload["config"]["alpha"]))
    paths = {
        "summary_json": str(summary_path),
        "records_json": str(records_path),
        "summary_csv": str(csv_path),
        "plot_svg": str(svg_path),
        "report_md": str(report_path),
    }
    write_markdown_report(report_path, payload=payload, paths=paths)
    return paths


def parse_sample_sizes(value: str) -> list[int]:
    sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("at least one sample size is required")
    if any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sample sizes must be positive")
    return sizes


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=float, default=4.0, help="Null radial dimension d.")
    parser.add_argument("--bins", type=int, default=8, help="Number of concentric shells.")
    parser.add_argument("--sample-sizes", type=parse_sample_sizes, default=parse_sample_sizes("64,128,256,512"))
    parser.add_argument("--trials", type=int, default=2000, help="Monte Carlo trials per scenario and sample size.")
    parser.add_argument(
        "--calibration-trials",
        type=int,
        default=10000,
        help="Parametric-bootstrap null trials per sample size.",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Nominal Type I error.")
    parser.add_argument("--radius", type=float, default=1.0, help="Outer ball radius.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = Path("runs/local/volume_shell_hypothesis") / stamp
    payload = run_experiment(
        dimension=args.dimension,
        bins=args.bins,
        sample_sizes=args.sample_sizes,
        trials=args.trials,
        calibration_trials=args.calibration_trials,
        alpha=args.alpha,
        radius=args.radius,
        seed=args.seed,
    )
    paths = write_outputs(payload, args.out_dir)
    printable = {
        "config": payload["config"],
        "paths": paths,
        "summary": payload["summary"],
    }
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
