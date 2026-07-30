#!/usr/bin/env python3
"""Monte Carlo size and power study for the analytic log-radius test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def exponential_ad_critical(samples: int, alpha: float = 0.05) -> float:
    base = {0.15: 0.922, 0.10: 1.078, 0.05: 1.341, 0.025: 1.606, 0.01: 1.957}
    if float(alpha) not in base:
        raise ValueError(f"alpha must be one of {sorted(base)}")
    return float(base[float(alpha)] / (1.0 + 0.6 / int(samples)))


def exponential_ad_rows(log_radii: np.ndarray) -> np.ndarray:
    """Anderson-Darling exponentiality statistics with row-wise fitted scale."""
    values = np.asarray(log_radii, dtype=np.float64)
    standardized = values / np.mean(values, axis=1, keepdims=True)
    ordered = np.sort(standardized, axis=1)
    cdf = np.clip(1.0 - np.exp(-ordered), 1e-12, 1.0 - 1e-12)
    n = values.shape[1]
    weights = 2.0 * np.arange(1, n + 1, dtype=np.float64) - 1.0
    terms = weights[None, :] * (np.log(cdf) + np.log1p(-cdf[:, ::-1]))
    return -float(n) - np.sum(terms, axis=1) / float(n)


def run_experiment(
    *,
    dimension: float,
    sample_sizes: list[int],
    trials: int,
    alpha: float,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(int(seed))
    scenarios = [
        ("null", 1.0, 1.0, "correct radial volume law"),
        ("inner_heavy", 0.45, 1.60, "excess mass near the anchor"),
        ("outer_heavy", 1.60, 0.45, "excess mass near the outer radius"),
    ]
    rows: list[dict[str, object]] = []

    for samples in sample_sizes:
        critical = exponential_ad_critical(int(samples), float(alpha))
        for name, beta_a, beta_b, description in scenarios:
            enclosed_volume = rng.beta(beta_a, beta_b, size=(int(trials), int(samples)))
            radii = np.clip(enclosed_volume, 1e-12, 1.0) ** (1.0 / float(dimension))
            log_radii = -np.log(radii)
            statistics = exponential_ad_rows(log_radii)
            rows.append(
                {
                    "scenario": name,
                    "description": description,
                    "dimension": float(dimension),
                    "beta_a": float(beta_a),
                    "beta_b": float(beta_b),
                    "samples": int(samples),
                    "trials": int(trials),
                    "alpha": float(alpha),
                    "critical_value": critical,
                    "mean_ad": float(np.mean(statistics)),
                    "median_ad": float(np.median(statistics)),
                    "rejection_rate": float(np.mean(statistics > critical)),
                }
            )

    return {
        "config": {
            "dimension": float(dimension),
            "sample_sizes": [int(value) for value in sample_sizes],
            "trials": int(trials),
            "alpha": float(alpha),
            "seed": int(seed),
        },
        "summary": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=float, default=4.0)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_experiment(
        dimension=args.dimension,
        sample_sizes=args.sample_sizes,
        trials=args.trials,
        alpha=args.alpha,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
