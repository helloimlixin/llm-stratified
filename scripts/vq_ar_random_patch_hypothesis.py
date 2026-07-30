"""Robust random-patch hypothesis tests for VQ-AR uniformity probes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_METRICS = (
    "local_ball_ks:lower",
    "local_ball_entropy:higher",
    "ranked_ks:lower",
    "branch_ks:lower",
    "branch_entropy:higher",
)


def finite_array(records: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in records:
        value = row.get(key)
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            value_f = float("nan")
        values.append(value_f)
    return np.asarray(values, dtype=np.float64)


def bool_array(records: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([bool(row.get(key, False)) for row in records], dtype=bool)


def int_array(records: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([int(row.get(key, -1)) for row in records], dtype=np.int64)


def parse_metric_spec(text: str) -> tuple[str, str]:
    if ":" in text:
        name, alternative = text.split(":", 1)
    else:
        name, alternative = text, "lower"
    alternative = alternative.strip().lower()
    if alternative not in {"lower", "higher"}:
        raise ValueError(f"metric alternative must be lower or higher, got {alternative!r}")
    return name.strip(), alternative


def mean_diff(values: np.ndarray, selector: np.ndarray) -> float:
    finite = np.isfinite(values)
    singular = finite & selector
    regular = finite & ~selector
    if int(singular.sum()) == 0 or int(regular.sum()) == 0:
        return float("nan")
    return float(values[singular].mean() - values[regular].mean())


def cohen_d(values: np.ndarray, selector: np.ndarray) -> float:
    finite = np.isfinite(values)
    a = values[finite & selector]
    b = values[finite & ~selector]
    if a.size < 2 or b.size < 2:
        return float("nan")
    var = ((a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1)) / max(a.size + b.size - 2, 1)
    if not np.isfinite(var) or var <= 0.0:
        return float("nan")
    return float((a.mean() - b.mean()) / math.sqrt(float(var)))


def one_sided_p(null: np.ndarray, observed: float, alternative: str) -> float:
    null = np.asarray(null, dtype=np.float64)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(observed):
        return float("nan")
    if alternative == "lower":
        extreme = int(np.sum(null <= observed))
    elif alternative == "higher":
        extreme = int(np.sum(null >= observed))
    else:
        raise ValueError(alternative)
    return float((extreme + 1.0) / (null.size + 1.0))


def summarize_distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def balanced_random_patch_diffs(
    values: np.ndarray,
    selector: np.ndarray,
    *,
    reps: int,
    sample_per_group: int,
    seed: int,
) -> np.ndarray:
    finite = np.isfinite(values)
    singular_idx = np.flatnonzero(finite & selector)
    regular_idx = np.flatnonzero(finite & ~selector)
    n = min(int(sample_per_group), singular_idx.size, regular_idx.size)
    if n <= 0:
        return np.full(max(1, int(reps)), np.nan)
    rng = np.random.default_rng(int(seed))
    diffs = np.empty(int(reps), dtype=np.float64)
    for i in range(int(reps)):
        a = rng.choice(singular_idx, size=n, replace=False)
        b = rng.choice(regular_idx, size=n, replace=False)
        diffs[i] = float(values[a].mean() - values[b].mean())
    return diffs


def image_block_bootstrap_diffs(
    values: np.ndarray,
    selector: np.ndarray,
    image_ids: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> np.ndarray:
    finite = np.isfinite(values)
    images = np.unique(image_ids[finite])
    if images.size == 0:
        return np.full(max(1, int(reps)), np.nan)
    grouped = {int(image): np.flatnonzero(finite & (image_ids == image)) for image in images}
    rng = np.random.default_rng(int(seed))
    diffs = np.empty(int(reps), dtype=np.float64)
    for i in range(int(reps)):
        sampled = rng.choice(images, size=images.size, replace=True)
        a_vals = []
        b_vals = []
        for image in sampled:
            idx = grouped[int(image)]
            a = idx[selector[idx]]
            b = idx[~selector[idx]]
            if a.size:
                a_vals.append(values[a])
            if b.size:
                b_vals.append(values[b])
        if not a_vals or not b_vals:
            diffs[i] = float("nan")
        else:
            diffs[i] = float(np.concatenate(a_vals).mean() - np.concatenate(b_vals).mean())
    return diffs


def within_image_permutation_null(
    values: np.ndarray,
    selector: np.ndarray,
    image_ids: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> np.ndarray:
    finite = np.isfinite(values)
    images = np.unique(image_ids[finite])
    grouped = [np.flatnonzero(finite & (image_ids == image)) for image in images]
    rng = np.random.default_rng(int(seed))
    null = np.empty(int(reps), dtype=np.float64)
    for i in range(int(reps)):
        permuted = selector.copy()
        for idx in grouped:
            labels = permuted[idx].copy()
            rng.shuffle(labels)
            permuted[idx] = labels
        null[i] = mean_diff(values, permuted)
    return null


def enrichment_stat(flat: np.ndarray, selector: np.ndarray) -> float:
    flat = np.asarray(flat, dtype=bool)
    if int(flat.sum()) == 0 or int((~flat).sum()) == 0:
        return float("nan")
    return float(selector[flat].mean() - selector[~flat].mean())


def within_image_enrichment_null(
    flat: np.ndarray,
    selector: np.ndarray,
    image_ids: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> np.ndarray:
    images = np.unique(image_ids)
    grouped = [np.flatnonzero(image_ids == image) for image in images]
    rng = np.random.default_rng(int(seed))
    null = np.empty(int(reps), dtype=np.float64)
    for i in range(int(reps)):
        permuted = selector.copy()
        for idx in grouped:
            labels = permuted[idx].copy()
            rng.shuffle(labels)
            permuted[idx] = labels
        null[i] = enrichment_stat(flat, permuted)
    return null


def sample_size_curve(
    values: np.ndarray,
    selector: np.ndarray,
    *,
    sizes: list[int],
    reps: int,
    seed: int,
) -> list[dict[str, float | int]]:
    rows = []
    for offset, size in enumerate(sizes):
        diffs = balanced_random_patch_diffs(
            values,
            selector,
            reps=reps,
            sample_per_group=int(size),
            seed=int(seed) + offset * 1009,
        )
        summary = summarize_distribution(diffs)
        rows.append({"sample_per_group": int(size), **summary})
    return rows


def plot_histogram(path: Path, values: np.ndarray, *, observed: float, title: str, xlabel: str) -> str:
    import matplotlib.pyplot as plt

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if finite.size:
        ax.hist(
            finite,
            bins=40,
            weights=np.full(finite.shape, 1.0 / finite.size),
            color="#4C78A8",
            alpha=0.75,
            edgecolor="white",
            linewidth=0.4,
        )
    ax.axvline(0.0, color="#222222", linewidth=1, linestyle=":")
    ax.axvline(float(observed), color="#F58518", linewidth=2, linestyle="--", label="observed")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of random samples")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return str(path)


def plot_sample_size(path: Path, curves: dict[str, list[dict[str, float | int]]]) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = {"local_ball_ks": "#4C78A8", "local_ball_entropy": "#F58518", "ranked_ks": "#54A24B"}
    for metric, rows in curves.items():
        x = np.asarray([int(row["sample_per_group"]) for row in rows], dtype=np.float64)
        y = np.asarray([float(row["mean"]) for row in rows], dtype=np.float64)
        lo = np.asarray([float(row["ci_low"]) for row in rows], dtype=np.float64)
        hi = np.asarray([float(row["ci_high"]) for row in rows], dtype=np.float64)
        color = colors.get(metric, None)
        ax.plot(x, y, marker="o", label=metric, color=color)
        ax.fill_between(x, lo, hi, alpha=0.16, color=color)
    ax.axhline(0.0, color="#222222", linewidth=1, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("random patch embeddings per group")
    ax.set_ylabel("singular minus regular mean")
    ax.set_title("Balanced random patch sampling stability")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return str(path)


def plot_control_bars(path: Path, rows: list[dict[str, Any]], metric: str, *, ylabel: str) -> str:
    import matplotlib.pyplot as plt

    names = [str(row["selector"]) for row in rows]
    values = [float(row["metrics"][metric]["observed_diff"]) for row in rows]
    colors = ["#F58518" if name == "codebook_target_large_fiber" else "#9ecae9" for name in names]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.axhline(0.0, color="#222222", linewidth=1)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Selector controls: {metric}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return str(path)


def selector_analysis(
    *,
    records: list[dict[str, Any]],
    selector_name: str,
    metric_specs: list[tuple[str, str]],
    image_ids: np.ndarray,
    reps: int,
    balanced_sample: int,
    seed: int,
) -> dict[str, Any]:
    selector = bool_array(records, selector_name)
    out: dict[str, Any] = {
        "selector": selector_name,
        "count": int(selector.sum()),
        "fraction": float(selector.mean()),
        "metrics": {},
    }
    for offset, (metric, alternative) in enumerate(metric_specs):
        values = finite_array(records, metric)
        observed = mean_diff(values, selector)
        balanced = balanced_random_patch_diffs(
            values,
            selector,
            reps=reps,
            sample_per_group=balanced_sample,
            seed=seed + offset * 1000 + 11,
        )
        block = image_block_bootstrap_diffs(
            values,
            selector,
            image_ids,
            reps=reps,
            seed=seed + offset * 1000 + 23,
        )
        perm = within_image_permutation_null(
            values,
            selector,
            image_ids,
            reps=reps,
            seed=seed + offset * 1000 + 37,
        )
        out["metrics"][metric] = {
            "alternative": alternative,
            "observed_diff": observed,
            "cohen_d": cohen_d(values, selector),
            "mean_singular": float(np.nanmean(values[selector])),
            "mean_regular": float(np.nanmean(values[~selector])),
            "balanced_random_patch": summarize_distribution(balanced),
            "balanced_random_patch_sign_support": float(np.mean(balanced < 0.0 if alternative == "lower" else balanced > 0.0)),
            "image_block_bootstrap": summarize_distribution(block),
            "image_block_sign_support": float(np.mean(block < 0.0 if alternative == "lower" else block > 0.0)),
            "within_image_permutation_p": one_sided_p(perm, observed, alternative),
            "within_image_permutation_null": summarize_distribution(perm),
        }
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    records_path = Path(args.records).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(records_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("--records must point to a non-empty vq_ar_ks_tokens.json list")
    image_ids = int_array(records, "sample_id")
    metric_specs = [parse_metric_spec(item) for item in args.metrics]
    selector_names = [args.selector]
    if args.include_controls:
        all_keys = set(records[0].keys())
        control_keys = sorted(
            key
            for key in all_keys
            if key.startswith("codebook_target_random_large_fiber_")
            or key.startswith("codebook_target_freqmatched_large_fiber_")
        )
        selector_names.extend(control_keys)

    selector_rows = [
        selector_analysis(
            records=records,
            selector_name=selector_name,
            metric_specs=metric_specs,
            image_ids=image_ids,
            reps=args.reps,
            balanced_sample=args.balanced_sample,
            seed=args.seed + idx * 20000,
        )
        for idx, selector_name in enumerate(selector_names)
    ]

    primary_selector = bool_array(records, args.selector)
    figures: dict[str, str] = {}
    for metric, alternative in metric_specs:
        values = finite_array(records, metric)
        observed = selector_rows[0]["metrics"][metric]["observed_diff"]
        balanced = balanced_random_patch_diffs(
            values,
            primary_selector,
            reps=args.reps,
            sample_per_group=args.balanced_sample,
            seed=args.seed + 40000 + len(figures),
        )
        figures[f"{metric}_balanced_hist"] = plot_histogram(
            out_dir / f"{metric}_balanced_random_patch_diff.png",
            balanced,
            observed=observed,
            title=f"Balanced random patch test: {metric}",
            xlabel="singular minus regular mean",
        )
        perm = within_image_permutation_null(
            values,
            primary_selector,
            image_ids,
            reps=args.reps,
            seed=args.seed + 50000 + len(figures),
        )
        figures[f"{metric}_within_image_perm"] = plot_histogram(
            out_dir / f"{metric}_within_image_permutation_null.png",
            perm,
            observed=observed,
            title=f"Within-image permutation null: {metric}",
            xlabel="permuted singular minus regular mean",
        )

    curve_metrics = [metric for metric, _alt in metric_specs[:3]]
    max_group = min(int(primary_selector.sum()), int((~primary_selector).sum()))
    sizes = [size for size in args.sample_sizes if size <= max_group]
    if max_group not in sizes:
        sizes.append(max_group)
    curves = {
        metric: sample_size_curve(
            finite_array(records, metric),
            primary_selector,
            sizes=sizes,
            reps=max(100, args.reps // 5),
            seed=args.seed + idx * 3000 + 70000,
        )
        for idx, metric in enumerate(curve_metrics)
    }
    figures["sample_size_curve"] = plot_sample_size(out_dir / "balanced_random_patch_sample_size_curve.png", curves)
    figures["local_ball_ks_control_bars"] = plot_control_bars(
        out_dir / "selector_controls_local_ball_ks.png",
        selector_rows,
        "local_ball_ks",
        ylabel="KS diff; lower is closer to uniform",
    )
    figures["local_ball_entropy_control_bars"] = plot_control_bars(
        out_dir / "selector_controls_local_ball_entropy.png",
        selector_rows,
        "local_ball_entropy",
        ylabel="entropy diff; higher is closer to uniform",
    )

    enrichments = {}
    for metric, alternative in metric_specs:
        values = finite_array(records, metric)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        threshold = float(np.nanquantile(values, args.flat_quantile if alternative == "lower" else 1.0 - args.flat_quantile))
        flat = np.zeros(values.shape, dtype=bool)
        flat[finite] = values[finite] <= threshold if alternative == "lower" else values[finite] >= threshold
        observed = enrichment_stat(flat, primary_selector)
        null = within_image_enrichment_null(
            flat,
            primary_selector,
            image_ids,
            reps=args.reps,
            seed=args.seed + 90000 + len(enrichments),
        )
        enrichments[metric] = {
            "alternative": alternative,
            "flat_quantile": float(args.flat_quantile),
            "threshold": threshold,
            "flat_count": int(flat.sum()),
            "singular_rate_flat": float(primary_selector[flat].mean()),
            "singular_rate_rest": float(primary_selector[~flat].mean()),
            "observed_rate_diff": observed,
            "within_image_permutation_p": one_sided_p(null, observed, "higher"),
            "within_image_permutation_null": summarize_distribution(null),
        }
        figures[f"{metric}_enrichment_perm"] = plot_histogram(
            out_dir / f"{metric}_flat_decile_enrichment_permutation_null.png",
            null,
            observed=observed,
            title=f"Flat-tail enrichment null: {metric}",
            xlabel="singular rate in flat tail minus rest",
        )

    summary = {
        "records": str(records_path),
        "out_dir": str(out_dir),
        "num_records": len(records),
        "num_images": int(np.unique(image_ids).size),
        "selector": args.selector,
        "balanced_sample_per_group": int(args.balanced_sample),
        "reps": int(args.reps),
        "flat_quantile": float(args.flat_quantile),
        "selector_analyses": selector_rows,
        "flat_tail_enrichment": enrichments,
        "sample_size_curves": curves,
        "figures": figures,
    }
    summary_path = out_dir / "random_patch_hypothesis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=[tag.strip() for tag in str(args.wandb_tags).split(",") if tag.strip()],
            config={
                "records": str(records_path),
                "num_records": len(records),
                "num_images": int(np.unique(image_ids).size),
                "selector": args.selector,
                "balanced_sample_per_group": int(args.balanced_sample),
                "reps": int(args.reps),
                "flat_quantile": float(args.flat_quantile),
            },
        )
        payload = {}
        primary = selector_rows[0]
        for metric, result in primary["metrics"].items():
            payload[f"random_patch/{metric}_observed_diff"] = result["observed_diff"]
            payload[f"random_patch/{metric}_within_image_permutation_p"] = result["within_image_permutation_p"]
            payload[f"random_patch/{metric}_balanced_ci_low"] = result["balanced_random_patch"]["ci_low"]
            payload[f"random_patch/{metric}_balanced_ci_high"] = result["balanced_random_patch"]["ci_high"]
        for metric, result in enrichments.items():
            payload[f"random_patch/{metric}_flat_enrichment_diff"] = result["observed_rate_diff"]
            payload[f"random_patch/{metric}_flat_enrichment_p"] = result["within_image_permutation_p"]
        for key, path in figures.items():
            payload[f"random_patch_figures/{key}"] = wandb.Image(path)
        wandb.log(payload)
        artifact = wandb.Artifact(f"{args.wandb_name}_outputs", type="analysis")
        artifact.add_file(str(summary_path))
        for path in figures.values():
            artifact.add_file(path)
        run.log_artifact(artifact)
        run.finish()

    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--selector", default="codebook_target_large_fiber")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--balanced-sample", type=int, default=2048)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--flat-quantile", type=float, default=0.10)
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="stratified-manifold-learning")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default="vq-ar-random-patch-hypothesis")
    parser.add_argument("--wandb-tags", default="vq-ar,random-patch,hypothesis-test")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = run(args)
    printable = {k: v for k, v in summary.items() if k not in {"selector_analyses", "sample_size_curves"}}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
