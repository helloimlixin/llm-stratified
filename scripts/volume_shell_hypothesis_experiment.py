#!/usr/bin/env python3
"""Compatibility entry point for the calibrated shell likelihood-ratio study.

The paper and experiments use one inferential path: fit the local radial
dimension, form equal-null-mass shells, compute ``2 N D(Q || P)``, and
calibrate the complete fitted pipeline by Monte Carlo simulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from radial_uniformity_hypothesis_experiment import render_power_dashboard, run_experiment as _run_experiment
except ImportError:  # Imported as scripts.volume_shell_hypothesis_experiment.
    from scripts.radial_uniformity_hypothesis_experiment import render_power_dashboard, run_experiment as _run_experiment


def shell_edges_equal_null_mass(*, bins: int, dimension: float, radius: float) -> np.ndarray:
    """Return concentric boundaries with probability ``1 / bins`` under the null."""
    if bins <= 0 or dimension <= 0.0 or radius <= 0.0:
        raise ValueError("bins, dimension, and radius must be positive")
    enclosed_mass = np.linspace(0.0, 1.0, int(bins) + 1, dtype=np.float64)
    return float(radius) * enclosed_mass ** (1.0 / float(dimension))


def shell_probabilities(*, edges: np.ndarray, radial_dimension: float, radius: float) -> np.ndarray:
    """Evaluate the radial-volume PMF on a supplied shell partition."""
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("edges must be a one-dimensional array")
    if radial_dimension <= 0.0 or radius <= 0.0:
        raise ValueError("radial_dimension and radius must be positive")
    cdf = np.clip(edges / float(radius), 0.0, 1.0) ** float(radial_dimension)
    probabilities = np.diff(cdf)
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("shells have zero total probability")
    return probabilities / total


def kl_divergence(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Return row-wise ``D_KL(Q || P)`` with the convention ``0 log 0 = 0``."""
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if q.ndim == 1:
        q = q[None, :]
    if q.shape[-1] != p.shape[0]:
        raise ValueError("q and p must have the same number of shells")
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(q > 0.0, q * (np.log(q) - np.log(p)), 0.0)
    return np.sum(terms, axis=-1)


def run_experiment(
    *,
    dimension: float,
    bins: int,
    sample_sizes: list[int],
    trials: int,
    calibration_trials: int,
    alpha: float,
    radius: float = 1.0,
    seed: int,
) -> dict[str, object]:
    """Run the shared fitted-shell simulation; ``radius`` is scale-invariant."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    return _run_experiment(
        dimension=dimension,
        bins=bins,
        sample_sizes=sample_sizes,
        trials=trials,
        calibration_trials=calibration_trials,
        alpha=alpha,
        seed=seed,
    )


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
