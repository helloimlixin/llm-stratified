"""Build aggregate VAR generation statistics from cached per-token JSON.

This is the lightweight path for paper results: it avoids reloading COCO or the
VAR checkpoint and recomputes only summary statistics and aggregate plots from
``epoch_000_var_generation_polysemy.json``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _mean(values: np.ndarray) -> float:
    values = _finite(values)
    return float(values.mean()) if values.size else float("nan")


def _sem(values: np.ndarray) -> float:
    values = _finite(values)
    if values.size <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(values.size))


def _rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(values)
    if int(mask.sum()) == 0:
        return out
    finite = values[mask]
    order = np.argsort(finite, kind="mergesort")
    sorted_values = finite[order]
    ranks_sorted = np.empty(sorted_values.shape, dtype=np.float64)
    start = 0
    while start < sorted_values.size:
        end = start + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks_sorted[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    ranks = np.empty(order.shape, dtype=np.float64)
    ranks[order] = ranks_sorted
    out[mask] = ranks
    return out


def _corr(x: np.ndarray, y: np.ndarray, *, spearman: bool = False) -> float:
    x = _rank(x) if spearman else np.asarray(x, dtype=np.float64)
    y = _rank(y) if spearman else np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(x.std()) <= 0.0 or float(y.std()) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _residualize(y: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    controls = [np.asarray(control, dtype=np.float64) for control in controls]
    mask = np.isfinite(y)
    for control in controls:
        mask &= np.isfinite(control)
    out = np.full(y.shape, np.nan, dtype=np.float64)
    if int(mask.sum()) < len(controls) + 3:
        return out
    design = [np.ones(int(mask.sum()), dtype=np.float64)]
    design.extend(control[mask] for control in controls)
    x = np.column_stack(design)
    beta, *_ = np.linalg.lstsq(x, y[mask], rcond=None)
    out[mask] = y[mask] - x @ beta
    return out


def _partial_spearman(x: np.ndarray, y: np.ndarray, controls: list[np.ndarray]) -> float:
    return _corr(
        _residualize(_rank(x), [_rank(control) for control in controls]),
        _residualize(_rank(y), [_rank(control) for control in controls]),
    )


def _tail_mask(values: np.ndarray, fraction: float, *, largest: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mask = np.zeros(values.shape, dtype=bool)
    finite_idx = np.flatnonzero(np.isfinite(values))
    if finite_idx.size == 0:
        return mask
    count = max(1, int(math.ceil(float(fraction) * finite_idx.size)))
    order = finite_idx[np.argsort(values[finite_idx])]
    selected = order[-count:] if largest else order[:count]
    mask[selected] = True
    return mask


def _cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = _finite(x)
    y = _finite(y)
    if x.size < 2 or y.size < 2:
        return float("nan")
    pooled = ((x.size - 1) * x.var(ddof=1) + (y.size - 1) * y.var(ddof=1)) / (x.size + y.size - 2)
    return float((x.mean() - y.mean()) / math.sqrt(float(pooled))) if pooled > 0.0 else float("nan")


def _perm_p(x: np.ndarray, y: np.ndarray, *, reps: int, seed: int) -> tuple[float, float]:
    x = _finite(x)
    y = _finite(y)
    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan")
    observed = float(x.mean() - y.mean())
    pooled = np.concatenate([x, y])
    rng = np.random.default_rng(seed)
    n_x = int(x.size)
    extreme = 0
    for _ in range(int(reps)):
        perm = rng.permutation(pooled)
        diff = float(perm[:n_x].mean() - perm[n_x:].mean())
        if abs(diff) >= abs(observed):
            extreme += 1
    return observed, float((extreme + 1.0) / (float(reps) + 1.0))


def _quantile_bins(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, -1, dtype=int)
    finite_idx = np.flatnonzero(np.isfinite(values))
    if finite_idx.size == 0:
        return out
    order = finite_idx[np.argsort(values[finite_idx])]
    for idx, split in enumerate(np.array_split(order, int(bins))):
        out[split] = idx
    return out


def _violation_bins(irregularity: np.ndarray, rejected: np.ndarray) -> tuple[np.ndarray, list[str]]:
    groups = np.zeros(irregularity.shape, dtype=int)
    labels = ["no violation", "low violation", "mid violation", "high violation"]
    rejected_idx = np.flatnonzero(rejected & np.isfinite(irregularity))
    if rejected_idx.size:
        order = rejected_idx[np.argsort(irregularity[rejected_idx])]
        for offset, split in enumerate(np.array_split(order, 3), start=1):
            groups[split] = offset
    return groups, labels


def _save_figure(fig, path: Path, *, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"), dpi=dpi, format="pdf")


def _plot_bins(
    *,
    dimension: np.ndarray,
    irregularity: np.ndarray,
    rejected: np.ndarray,
    entropy: np.ndarray,
    nll: np.ndarray,
    path: Path,
) -> None:
    dim_bins = _quantile_bins(dimension, 10)
    viol_bins, viol_labels = _violation_bins(irregularity, rejected)
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.2), squeeze=False)
    axes = axes.ravel()
    for ax, metric, ylabel, title in (
        (axes[0], entropy, "normalized entropy", "Entropy by dimension decile"),
        (axes[1], nll, "true-code NLL", "NLL by dimension decile"),
    ):
        xs = np.arange(10)
        means = [_mean(metric[dim_bins == idx]) for idx in xs]
        sems = [_sem(metric[dim_bins == idx]) for idx in xs]
        ax.errorbar(xs, means, yerr=sems, marker="o", linewidth=2.0, capsize=3, color="#355C7D")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(idx + 1) for idx in xs], fontsize=11)
        ax.set_xlabel("dimension decile (low to high)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, pad=10)
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.24, linewidth=0.7)

    ax = axes[2]
    xs = np.arange(len(viol_labels))
    entropy_means = [_mean(entropy[viol_bins == idx]) for idx in xs]
    entropy_sems = [_sem(entropy[viol_bins == idx]) for idx in xs]
    nll_means = [_mean(nll[viol_bins == idx]) for idx in xs]
    nll_sems = [_sem(nll[viol_bins == idx]) for idx in xs]
    counts = [int((viol_bins == idx).sum()) for idx in xs]
    width = 0.36
    ax.bar(xs - width / 2, entropy_means, width, yerr=entropy_sems, color="#C06C84", alpha=0.82, capsize=3)
    ax.set_ylabel("normalized entropy", fontsize=12, color="#8F3D57")
    ax.tick_params(axis="y", labelcolor="#8F3D57", labelsize=11)
    ax2 = ax.twinx()
    ax2.bar(xs + width / 2, nll_means, width, yerr=nll_sems, color="#6C5B7B", alpha=0.82, capsize=3)
    ax2.set_ylabel("true-code NLL", fontsize=12, color="#4F415E")
    ax2.tick_params(axis="y", labelcolor="#4F415E", labelsize=11)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{label}\n(n={count})" for label, count in zip(viol_labels, counts)], fontsize=10)
    ax.set_title("Generation metrics by fiber-violation strength", fontsize=13, pad=10)
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.7)

    fig.suptitle("Aggregate VAR Generation Diagnostics Across All Tokens", fontsize=18, y=0.985)
    fig.text(
        0.02,
        0.018,
        "Dimension trends use all teacher-forced visual tokens. Fiber-violation bins separate non-violating tokens "
        "from corrected slope-increase violations split into tertiles by irregularity.",
        fontsize=11,
        color="#333333",
    )
    fig.subplots_adjust(left=0.055, right=0.945, top=0.790, bottom=0.205, wspace=0.34)
    _save_figure(fig, path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        default=REPO_ROOT
        / "runs/local/coco_var_d30_sparse_fiber/20260506_232532/checkpoints/fiber_analysis/epoch_000_var_generation_polysemy.json",
    )
    parser.add_argument("--permutation-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    tokens = list(payload.get("tokens", []))
    if not tokens:
        raise ValueError(f"No tokens found in {args.json}")

    entropy = np.asarray([float(token["entropy_norm"]) for token in tokens], dtype=np.float64)
    nll = np.asarray([float(token["nll"]) for token in tokens], dtype=np.float64)
    top1 = np.asarray([float(token["top1_prob"]) for token in tokens], dtype=np.float64)
    dimension = np.asarray([float(token["dimension"]) for token in tokens], dtype=np.float64)
    irregularity = np.asarray([float(token["irregularity"]) for token in tokens], dtype=np.float64)
    rejected = np.asarray([bool(token.get("fiber_violation_reject", False)) for token in tokens], dtype=bool)
    image_ids = np.asarray([float(token["image_id"]) for token in tokens], dtype=np.float64)
    rows = np.asarray([float(token["row"]) for token in tokens], dtype=np.float64)
    cols = np.asarray([float(token["col"]) for token in tokens], dtype=np.float64)

    high_irregular = _tail_mask(irregularity, 0.1, largest=True)
    low_irregular = _tail_mask(irregularity, 0.1, largest=False)
    high_dimension = _tail_mask(dimension, 0.1, largest=True)
    low_dimension = _tail_mask(dimension, 0.1, largest=False)
    quiet = ~rejected
    reps = int(args.permutation_reps)
    seed = int(args.seed)

    entropy_rej_diff, entropy_rej_p = _perm_p(entropy[rejected], entropy[quiet], reps=reps, seed=seed)
    nll_rej_diff, nll_rej_p = _perm_p(nll[rejected], nll[quiet], reps=reps, seed=seed + 1)
    entropy_irr_diff, entropy_irr_p = _perm_p(entropy[high_irregular], entropy[low_irregular], reps=reps, seed=seed + 2)
    nll_irr_diff, nll_irr_p = _perm_p(nll[high_irregular], nll[low_irregular], reps=reps, seed=seed + 3)
    entropy_dim_diff, entropy_dim_p = _perm_p(entropy[high_dimension], entropy[low_dimension], reps=reps, seed=seed + 4)
    nll_dim_diff, nll_dim_p = _perm_p(nll[high_dimension], nll[low_dimension], reps=reps, seed=seed + 5)

    summary = dict(payload.get("summary", {}))
    summary.update(
        {
            "mean_entropy_high_dimension_decile": _mean(entropy[high_dimension]),
            "mean_entropy_low_dimension_decile": _mean(entropy[low_dimension]),
            "mean_nll_high_dimension_decile": _mean(nll[high_dimension]),
            "mean_nll_low_dimension_decile": _mean(nll[low_dimension]),
            "diff_entropy_rejected_minus_nonrejected": entropy_rej_diff,
            "perm_p_entropy_rejected_vs_nonrejected": entropy_rej_p,
            "cohen_d_entropy_rejected_vs_nonrejected": _cohen_d(entropy[rejected], entropy[quiet]),
            "diff_nll_rejected_minus_nonrejected": nll_rej_diff,
            "perm_p_nll_rejected_vs_nonrejected": nll_rej_p,
            "cohen_d_nll_rejected_vs_nonrejected": _cohen_d(nll[rejected], nll[quiet]),
            "diff_entropy_high_minus_low_irregular_decile": entropy_irr_diff,
            "perm_p_entropy_high_vs_low_irregular_decile": entropy_irr_p,
            "cohen_d_entropy_high_vs_low_irregular_decile": _cohen_d(entropy[high_irregular], entropy[low_irregular]),
            "diff_nll_high_minus_low_irregular_decile": nll_irr_diff,
            "perm_p_nll_high_vs_low_irregular_decile": nll_irr_p,
            "cohen_d_nll_high_vs_low_irregular_decile": _cohen_d(nll[high_irregular], nll[low_irregular]),
            "diff_entropy_high_minus_low_dimension_decile": entropy_dim_diff,
            "perm_p_entropy_high_vs_low_dimension_decile": entropy_dim_p,
            "cohen_d_entropy_high_vs_low_dimension_decile": _cohen_d(entropy[high_dimension], entropy[low_dimension]),
            "diff_nll_high_minus_low_dimension_decile": nll_dim_diff,
            "perm_p_nll_high_vs_low_dimension_decile": nll_dim_p,
            "cohen_d_nll_high_vs_low_dimension_decile": _cohen_d(nll[high_dimension], nll[low_dimension]),
            "corr_irregularity_entropy_spearman": _corr(irregularity, entropy, spearman=True),
            "corr_irregularity_nll_spearman": _corr(irregularity, nll, spearman=True),
            "corr_irregularity_top1_prob_spearman": _corr(irregularity, top1, spearman=True),
            "corr_dimension_entropy_spearman": _corr(dimension, entropy, spearman=True),
            "corr_dimension_nll_spearman": _corr(dimension, nll, spearman=True),
            "partial_corr_irregularity_entropy_given_dimension_position_spearman": _partial_spearman(
                irregularity, entropy, [dimension, image_ids, rows, cols]
            ),
            "partial_corr_irregularity_nll_given_dimension_position_spearman": _partial_spearman(
                irregularity, nll, [dimension, image_ids, rows, cols]
            ),
            "partial_corr_dimension_entropy_given_irregularity_position_spearman": _partial_spearman(
                dimension, entropy, [irregularity, image_ids, rows, cols]
            ),
            "partial_corr_dimension_nll_given_irregularity_position_spearman": _partial_spearman(
                dimension, nll, [irregularity, image_ids, rows, cols]
            ),
        }
    )
    payload["summary"] = summary
    prefix = args.json.with_suffix("")
    aggregate_path = prefix.parent / f"{prefix.name}_aggregate_bins.png"
    _plot_bins(
        dimension=dimension,
        irregularity=irregularity,
        rejected=rejected,
        entropy=entropy,
        nll=nll,
        path=aggregate_path,
    )
    figures = dict(payload.get("figures", {}))
    figures["aggregate_bins"] = str(aggregate_path)
    payload["figures"] = figures
    with args.json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({"summary": summary, "aggregate_bins": str(aggregate_path)}, indent=2))


if __name__ == "__main__":
    main()
