"""Shared statistics for fitted-dimension concentric-shell tests."""

from __future__ import annotations

import math

import numpy as np


def shell_deviance(counts: np.ndarray, probabilities: np.ndarray | None = None) -> float:
    """Return the multinomial deviance ``2 N D(Q_hat || P)``."""
    observed = np.asarray(counts, dtype=np.float64).reshape(-1)
    if observed.size < 2 or np.any(observed < 0.0):
        raise ValueError("counts must contain at least two nonnegative entries")
    samples = float(observed.sum())
    if samples <= 0.0:
        return float("nan")
    if probabilities is None:
        expected_probabilities = np.full(observed.size, 1.0 / observed.size, dtype=np.float64)
    else:
        expected_probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if expected_probabilities.shape != observed.shape:
            raise ValueError("counts and probabilities must have the same shape")
        if np.any(expected_probabilities <= 0.0) or not np.isfinite(expected_probabilities).all():
            raise ValueError("probabilities must be finite and strictly positive")
        expected_probabilities = expected_probabilities / expected_probabilities.sum()
    positive = observed > 0.0
    return float(
        2.0
        * np.sum(
            observed[positive]
            * np.log(observed[positive] / (samples * expected_probabilities[positive]))
        )
    )


def fitted_equal_mass_shell_counts(log_radii: np.ndarray, bins: int) -> np.ndarray:
    """Count radii in equal-null-mass shells after fitting exponential scale.

    ``log_radii`` contains ``log(r_star / R_i)``. Under the local volume null,
    these values are exponential with unknown rate. Dividing by their row mean
    removes that rate, and ``exp(-Z_i / mean(Z))`` is the fitted enclosed-volume
    coordinate used to define the shells.
    """
    values = np.asarray(log_radii, dtype=np.float64)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("log_radii must be a vector or matrix with at least two samples per row")
    if int(bins) < 2:
        raise ValueError("bins must be at least two")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("log_radii must be finite and strictly positive")

    standardized = values / np.mean(values, axis=1, keepdims=True)
    fitted_volume = np.exp(-standardized)
    indices = np.floor(fitted_volume * int(bins)).astype(np.int64)
    indices = np.clip(indices, 0, int(bins) - 1)
    counts = np.stack(
        [np.sum(indices == shell, axis=1) for shell in range(int(bins))],
        axis=1,
    ).astype(np.int64)
    return counts[0] if squeeze else counts


def fitted_equal_mass_shell_deviance(log_radii: np.ndarray, bins: int) -> np.ndarray | float:
    """Compute fitted-shell deviances for one row or a matrix of log-radii."""
    counts = fitted_equal_mass_shell_counts(log_radii, bins)
    squeeze = counts.ndim == 1
    if squeeze:
        return shell_deviance(counts)
    samples = counts.sum(axis=1, keepdims=True).astype(np.float64)
    expected = samples / float(bins)
    positive = counts > 0
    terms = np.zeros_like(counts, dtype=np.float64)
    terms[positive] = counts[positive] * np.log(counts[positive] / np.broadcast_to(expected, counts.shape)[positive])
    statistics = 2.0 * terms.sum(axis=1)
    return statistics


def fitted_shell_test_from_distances(
    distances: np.ndarray,
    *,
    radius: float,
    bins: int,
) -> tuple[np.ndarray, float]:
    """Return fitted equal-mass shell counts and their likelihood-ratio statistic."""
    values = np.asarray(distances, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0) & (values < float(radius))]
    if values.size < 2 or not math.isfinite(float(radius)) or float(radius) <= 0.0:
        return np.zeros(int(bins), dtype=np.int64), float("nan")
    log_radii = np.log(float(radius) / values)
    counts = fitted_equal_mass_shell_counts(log_radii, int(bins))
    return counts, shell_deviance(counts)


def quantile_higher(values: np.ndarray, probability: float) -> float:
    """Return the smallest observed value whose empirical CDF reaches a probability."""
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if ordered.size == 0:
        raise ValueError("values must be non-empty")
    rank = int(math.ceil(float(probability) * ordered.size)) - 1
    return float(ordered[max(0, min(rank, ordered.size - 1))])


def calibrate_fitted_shell_deviance(
    *,
    samples: int,
    bins: int,
    alpha: float,
    trials: int,
    seed: int,
    batch_size: int = 4096,
) -> tuple[float, np.ndarray]:
    """Calibrate the complete fitted-shell pipeline under the exponential null.

    The standardized statistic is scale-free, so unit-rate exponential draws
    cover every positive null dimension. The returned null statistics are
    sorted to support efficient Monte Carlo p-values.
    """
    if int(samples) < 2:
        raise ValueError("samples must be at least two")
    if int(bins) < 2:
        raise ValueError("bins must be at least two")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if int(trials) < 2:
        raise ValueError("trials must be at least two")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")

    rng = np.random.default_rng(int(seed))
    statistics = np.empty(int(trials), dtype=np.float64)
    for start in range(0, int(trials), int(batch_size)):
        stop = min(start + int(batch_size), int(trials))
        log_radii = rng.exponential(scale=1.0, size=(stop - start, int(samples)))
        statistics[start:stop] = fitted_equal_mass_shell_deviance(log_radii, int(bins))
    statistics.sort()
    return quantile_higher(statistics, 1.0 - float(alpha)), statistics


def monte_carlo_pvalue(statistic: float, sorted_null_statistics: np.ndarray) -> float:
    """Return the finite-simulation upper-tail p-value with a plus-one correction."""
    if not math.isfinite(float(statistic)):
        return float("nan")
    null = np.asarray(sorted_null_statistics, dtype=np.float64).reshape(-1)
    if null.size == 0:
        raise ValueError("sorted_null_statistics must be non-empty")
    first_at_least = int(np.searchsorted(null, float(statistic), side="left"))
    exceedances = int(null.size - first_at_least)
    return float((1 + exceedances) / (null.size + 1))
