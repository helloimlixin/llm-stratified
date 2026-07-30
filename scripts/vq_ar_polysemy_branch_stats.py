"""Paired inference for VQ-AR polysemy branch galleries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_pairs(summary_path: Path, metric: str) -> tuple[np.ndarray, np.ndarray]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    singular = []
    regular = []
    for row in summary.get("anchors", []):
        value = row.get(metric)
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            value_f = float("nan")
        if row.get("kind") == "singular":
            singular.append(value_f)
        elif row.get("kind") == "regular":
            regular.append(value_f)
    n = min(len(singular), len(regular))
    if n == 0:
        raise ValueError(f"{summary_path} does not contain paired singular/regular anchors")
    singular_arr = np.asarray(singular[:n], dtype=np.float64)
    regular_arr = np.asarray(regular[:n], dtype=np.float64)
    finite = np.isfinite(singular_arr) & np.isfinite(regular_arr)
    if not finite.any():
        raise ValueError(f"{summary_path} has no finite paired values for {metric}")
    return singular_arr[finite], regular_arr[finite]


def binomial_tail_probability(wins: int, n: int, p: float = 0.5) -> float:
    if n <= 0:
        return float("nan")
    wins = int(wins)
    prob = 0.0
    for k in range(wins, n + 1):
        prob += math.comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))
    return float(prob)


def paired_sign_flip_p(diffs: np.ndarray, *, reps: int, seed: int) -> float:
    diffs = np.asarray(diffs, dtype=np.float64)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan")
    observed = float(np.mean(diffs))
    if diffs.size <= 20:
        signs = np.asarray(np.meshgrid(*[[-1.0, 1.0]] * diffs.size)).T.reshape(-1, diffs.size)
        null = signs @ diffs / diffs.size
        extreme = int(np.sum(null >= observed))
        return float((extreme + 1.0) / (null.size + 1.0))
    rng = np.random.default_rng(int(seed))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(int(reps), diffs.size))
    null = signs @ diffs / diffs.size
    extreme = int(np.sum(null >= observed))
    return float((extreme + 1.0) / (int(reps) + 1.0))


def bootstrap_ci(diffs: np.ndarray, *, reps: int, seed: int) -> dict[str, float]:
    diffs = np.asarray(diffs, dtype=np.float64)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return {"mean_ci_low": float("nan"), "mean_ci_high": float("nan")}
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, diffs.size, size=(int(reps), diffs.size))
    means = diffs[idx].mean(axis=1)
    return {
        "mean_ci_low": float(np.quantile(means, 0.025)),
        "mean_ci_high": float(np.quantile(means, 0.975)),
    }


def paired_summary(
    singular: np.ndarray,
    regular: np.ndarray,
    *,
    branches: int,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    diffs = np.asarray(singular, dtype=np.float64) - np.asarray(regular, dtype=np.float64)
    wins = int(np.sum(diffs > 0.0))
    losses = int(np.sum(diffs < 0.0))
    ties = int(np.sum(diffs == 0.0))
    n = int(diffs.size)
    ci = bootstrap_ci(diffs, reps=reps, seed=seed)
    std = float(np.std(diffs, ddof=1)) if n > 1 else float("nan")
    paired_d = float(np.mean(diffs) / std) if std > 0 and math.isfinite(std) else float("nan")
    return {
        "n_pairs": n,
        "branches": int(branches),
        "branch_variants_excluding_originals": int(2 * n * int(branches)),
        "singular_mean": float(np.mean(singular)),
        "regular_mean": float(np.mean(regular)),
        "singular_median": float(np.median(singular)),
        "regular_median": float(np.median(regular)),
        "mean_diff": float(np.mean(diffs)),
        "median_diff": float(np.median(diffs)),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": float(wins / n) if n else float("nan"),
        "paired_cohens_dz": paired_d,
        "sign_test_p_one_sided": binomial_tail_probability(wins, n),
        "paired_sign_flip_p_one_sided": paired_sign_flip_p(diffs, reps=reps, seed=seed + 17),
        **ci,
    }


def plot_stats(rows: list[dict[str, Any]], path: Path) -> str:
    import matplotlib.pyplot as plt

    labels = [row["name"] for row in rows]
    means = np.asarray([row["mean_diff"] for row in rows], dtype=float)
    lo = np.asarray([row["mean_ci_low"] for row in rows], dtype=float)
    hi = np.asarray([row["mean_ci_high"] for row in rows], dtype=float)
    yerr = np.vstack([means - lo, hi - means])
    win_rates = np.asarray([row["win_rate"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = np.arange(len(rows))
    axes[0].bar(x, means, color="#ff7f0e", width=0.62)
    axes[0].errorbar(x, means, yerr=yerr, fmt="none", color="black", capsize=4, linewidth=1)
    axes[0].axhline(0.0, color="#222222", linewidth=1)
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("mean paired crop-L2 difference")
    axes[0].set_title("Bootstrap 95% CI")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, win_rates, color="#4c78a8", width=0.62)
    axes[1].axhline(0.5, color="#222222", linewidth=1, linestyle=":")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylabel("singular wins / matched pairs")
    axes[1].set_title("Pairwise sign consistency")
    for idx, row in enumerate(rows):
        axes[1].text(idx, min(0.96, row["win_rate"] + 0.04), f"{row['wins']}/{row['n_pairs']}", ha="center", fontsize=9)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Paired tests for singular-token branch polysemy", y=1.03)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return str(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for idx, summary_text in enumerate(args.summary):
        summary_path = Path(summary_text)
        singular, regular = load_pairs(summary_path, args.metric)
        name = args.name[idx] if idx < len(args.name) else summary_path.parent.name
        branches = args.branches[idx] if idx < len(args.branches) else 0
        row = {
            "name": name,
            "summary_path": str(summary_path.resolve()),
            "metric": args.metric,
            **paired_summary(
                singular,
                regular,
                branches=branches,
                reps=args.reps,
                seed=args.seed + idx * 1009,
            ),
        }
        rows.append(row)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {"paired_stats": plot_stats(rows, out_dir / "vq_ar_polysemy_branch_paired_stats.png")}
    summary = {
        "metric": args.metric,
        "reps": int(args.reps),
        "seed": int(args.seed),
        "results": rows,
        "figures": figures,
    }
    (out_dir / "vq_ar_polysemy_branch_paired_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", required=True, help="Path to a branch-gallery summary JSON. Repeat for multiple runs.")
    parser.add_argument("--name", action="append", default=[], help="Display name for each summary, in the same order as --summary.")
    parser.add_argument("--branches", action="append", type=int, default=[], help="Branch count for each summary, used only for reporting variant counts.")
    parser.add_argument("--metric", default="crop_pairwise_l2")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260630)
    return parser


def main() -> None:
    summary = run(build_argparser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
