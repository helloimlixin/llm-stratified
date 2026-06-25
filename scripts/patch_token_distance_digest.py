"""Build a compact summary figure for patch-token distance diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fiber.figure_io import save_figure

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise ImportError("matplotlib is required for patch-token distance digest figures") from exc


COLORS = {
    "DINOv3-H+": "#377eb8",
    "SAM-H": "#2ca25f",
    "SigLIP2-B": "#e69f00",
    "AIMv2-L": "#7b3294",
}

METRICS = [
    ("Distance-rank agreement", "mean_rank_spearman"),
    ("Top-16 neighbor overlap", "mean_neighbor_overlap"),
    ("Matrix-rank agreement", "mean_matrix_spearman"),
]


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _parse_summary_spec(spec: str) -> tuple[str | None, Path]:
    if "=" not in spec:
        return None, _resolve(spec)
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    return (label or None), _resolve(raw_path.strip())


def _read_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _load_run(spec: str) -> dict[str, Any]:
    label_override, summary_path = _parse_summary_spec(spec)
    with summary_path.open() as fp:
        summary = json.load(fp)
    label = label_override or str(summary["label"])
    csv_path = _resolve(summary["csv_path"])
    rows: list[dict[str, float]] = []
    with csv_path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for raw in reader:
            rows.append(
                {
                    "image_id": _read_float(raw, "image_id"),
                    "rank": _read_float(raw, "mean_rank_spearman"),
                    "overlap": _read_float(raw, "mean_neighbor_overlap"),
                    "matrix": _read_float(raw, "matrix_spearman"),
                }
            )
    return {
        "label": label,
        "summary": summary,
        "rows": rows,
        "color": COLORS.get(label, "#4d4d4d"),
    }


def _jitter(count: int, width: float = 0.10) -> np.ndarray:
    if count <= 1:
        return np.zeros(count)
    return np.linspace(-width, width, count)


def _style_axis(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def _draw_metric_bars(ax: Any, runs: list[dict[str, Any]]) -> None:
    x = np.arange(len(METRICS), dtype=float)
    width = 0.18
    offsets = (np.arange(len(runs), dtype=float) - (len(runs) - 1) / 2.0) * width
    for offset, run in zip(offsets, runs):
        heights = [float(run["summary"][metric_key]) for _name, metric_key in METRICS]
        bars = ax.bar(
            x + offset,
            heights,
            width=width * 0.92,
            label=run["label"],
            color=run["color"],
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, value in zip(bars, heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.018,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                rotation=90,
            )
    ax.set_title("A. One-score summary", loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("agreement with raw-patch geometry")
    ax.set_ylim(0.0, 0.72)
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _key in METRICS], rotation=12, ha="right")
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper right")
    _style_axis(ax)


def _draw_strip(ax: Any, runs: list[dict[str, Any]], *, value_key: str, title: str, ylabel: str) -> None:
    for idx, run in enumerate(runs):
        values = np.asarray([row[value_key] for row in run["rows"]], dtype=float)
        values = values[np.isfinite(values)]
        xs = idx + _jitter(len(values), width=0.13)
        ax.scatter(
            xs,
            values,
            s=28,
            color=run["color"],
            alpha=0.72,
            edgecolors="white",
            linewidths=0.5,
        )
        mean = float(np.nanmean(values))
        ax.plot([idx - 0.23, idx + 0.23], [mean, mean], color="#222222", linewidth=2.0)
        ax.text(idx, mean + 0.025, f"{mean:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(runs)))
    ax.set_xticklabels([run["label"] for run in runs], rotation=20, ha="right")
    _style_axis(ax)


def _draw_relationship(ax: Any, runs: list[dict[str, Any]]) -> None:
    for run in runs:
        rank = np.asarray([row["rank"] for row in run["rows"]], dtype=float)
        overlap = np.asarray([row["overlap"] for row in run["rows"]], dtype=float)
        mask = np.isfinite(rank) & np.isfinite(overlap)
        ax.scatter(
            rank[mask],
            overlap[mask],
            s=36,
            color=run["color"],
            label=run["label"],
            alpha=0.75,
            edgecolors="white",
            linewidths=0.5,
        )
        ax.scatter(
            [float(np.nanmean(rank))],
            [float(np.nanmean(overlap))],
            s=120,
            color=run["color"],
            edgecolors="#111111",
            linewidths=1.0,
            marker="D",
        )
    ax.set_title("D. Rank vs. nearest neighbors", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("distance-rank agreement")
    ax.set_ylabel("top-16 neighbor overlap")
    ax.set_xlim(0.0, 0.72)
    ax.set_ylim(0.0, 0.52)
    ax.annotate(
        "more raw-like",
        xy=(0.66, 0.46),
        xytext=(0.35, 0.47),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": "#444444"},
        fontsize=9,
        color="#333333",
    )
    _style_axis(ax)


def build_figure(runs: list[dict[str, Any]], out_path: Path) -> Path:
    fig = plt.figure(figsize=(13.4, 8.9), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.0, 1.0, 1.10],
        left=0.075,
        right=0.985,
        bottom=0.09,
        top=0.90,
        wspace=0.24,
        hspace=0.36,
    )
    ax_a = fig.add_subplot(gs[0, :3])
    ax_d = fig.add_subplot(gs[0, 3])
    ax_b = fig.add_subplot(gs[1, :2])
    ax_c = fig.add_subplot(gs[1, 2:])

    _draw_metric_bars(ax_a, runs)
    _draw_relationship(ax_d, runs)
    _draw_strip(
        ax_b,
        runs,
        value_key="rank",
        title="B. Per-image distance-rank agreement",
        ylabel="mean Spearman over anchor patches",
    )
    ax_b.set_ylim(0.0, 0.78)
    _draw_strip(
        ax_c,
        runs,
        value_key="overlap",
        title="C. Per-image top-16 neighbor overlap",
        ylabel="fraction of raw neighbors recovered",
    )
    ax_c.set_ylim(0.0, 0.54)

    fig.suptitle(
        "Patch-Token Distance Diagnostic: Raw-Geometry Retention",
        fontsize=18,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.075,
        0.025,
        "Higher values mean token distances preserve more raw RGB patch-distance structure; lower values can reflect semantic/contextual reorganization.",
        fontsize=9.2,
        color="#333333",
    )
    path = save_figure(fig, out_path, dpi=240)
    plt.close(fig)
    return path


def run(args: argparse.Namespace) -> None:
    runs = [_load_run(spec) for spec in args.summary]
    if len(runs) < 2:
        raise ValueError("Provide at least two --summary entries")
    path = build_figure(runs, args.out)
    print(f"[patch_token_distance_digest] wrote {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Summary JSON path, optionally as Label=path. Repeat once per model.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
