"""Sweep OMP residual targets from saved dense patch-token artifacts.

This script reuses a saved ``embeddings/epoch_000.pt`` artifact and the matching
``fiber_epoch_000.json`` results, then recomputes the local sparse dictionary
probe over the same fixed-k token neighborhoods for a list of residual targets.
It is intentionally analysis-only: no model forward pass or classifier training
is run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fiber.figure_io import save_figure  # noqa: E402
from fiber.sparse_probe import (  # noqa: E402
    _build_sparse_probe_heatmaps,
    _build_sparse_probe_plot,
    _finite_corr,
    _first_dim,
    _min_pvalue,
    _standardize_patch_matrix,
    extract_patch_vectors,
    fit_pca_dictionary,
    min_change_pvalue,
    select_probe_tokens,
)

try:  # noqa: E402
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _pairwise_distances(embeddings: torch.Tensor) -> np.ndarray:
    x = embeddings.detach().float().cpu().numpy().astype(np.float64, copy=False)
    sq = np.sum(x * x, axis=1, keepdims=True)
    d2 = sq + sq.T - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    dists = np.sqrt(d2, out=d2)
    np.fill_diagonal(dists, 0.0)
    return dists


def _dictionary_is_orthonormal(dictionary: np.ndarray, *, atol: float = 1e-6) -> bool:
    atoms = np.asarray(dictionary, dtype=np.float64)
    if atoms.ndim != 2 or atoms.shape[0] == 0:
        return False
    gram = atoms @ atoms.T
    eye = np.eye(int(atoms.shape[0]), dtype=np.float64)
    return bool(np.allclose(gram, eye, atol=atol, rtol=1e-5))


def _omp_residual_path(
    target: np.ndarray,
    dictionary: np.ndarray,
    *,
    max_sparsity: int,
    orthonormal_dictionary: bool = False,
) -> np.ndarray:
    atoms = np.asarray(dictionary, dtype=np.float64)
    x = np.asarray(target, dtype=np.float64).reshape(-1)
    if atoms.ndim != 2 or atoms.shape[0] == 0 or atoms.shape[1] != x.size:
        return np.asarray([], dtype=np.float64)
    x_norm = float(np.linalg.norm(x))
    if not math.isfinite(x_norm) or x_norm <= 1e-12:
        return np.asarray([0.0], dtype=np.float64)

    max_s = max(1, min(int(max_sparsity), int(atoms.shape[0])))
    if orthonormal_dictionary:
        coeffs = atoms @ x
        coeff_energy = np.sort(np.square(coeffs))[::-1][:max_s]
        explained = np.cumsum(coeff_energy)
        residual_sq = np.maximum((x_norm * x_norm) - explained, 0.0)
        return np.sqrt(residual_sq) / x_norm

    residual = x.copy()
    selected: list[int] = []
    residuals: list[float] = []
    for _ in range(max_s):
        corr = atoms @ residual
        if selected:
            corr[np.asarray(selected, dtype=np.int64)] = 0.0
        atom_idx = int(np.argmax(np.abs(corr)))
        if atom_idx in selected:
            break
        selected.append(atom_idx)
        basis = atoms[np.asarray(selected, dtype=np.int64)].T
        coeffs, *_ = np.linalg.lstsq(basis, x, rcond=None)
        residual = x - basis @ coeffs
        rel_residual = float(np.linalg.norm(residual) / x_norm)
        residuals.append(rel_residual)
    return np.asarray(residuals, dtype=np.float64)


def _threshold_sparse_code(paths: list[np.ndarray], threshold: float, max_sparsity: int) -> dict[str, Any] | None:
    sparsities: list[int] = []
    residuals: list[float] = []
    hits = 0
    max_s = max(1, int(max_sparsity))
    for path in paths:
        arr = np.asarray(path, dtype=np.float64)
        if arr.size == 0:
            continue
        hit_idx = np.flatnonzero(arr <= float(threshold))
        if hit_idx.size:
            idx = int(hit_idx[0])
            sparsities.append(idx + 1)
            residuals.append(float(arr[idx]))
            hits += 1
        else:
            idx = min(max_s, int(arr.size)) - 1
            sparsities.append(min(max_s, int(arr.size)))
            residuals.append(float(arr[idx]))
    if not sparsities:
        return None
    sparsity_arr = np.asarray(sparsities, dtype=np.float64)
    residual_arr = np.asarray(residuals, dtype=np.float64)
    return {
        "mean_required_sparsity": float(np.nanmean(sparsity_arr)),
        "median_required_sparsity": float(np.nanmedian(sparsity_arr)),
        "std_required_sparsity": float(np.nanstd(sparsity_arr)),
        "mean_relative_residual": float(np.nanmean(residual_arr)),
        "residual_hit_ratio": float(hits / max(1, len(sparsities))),
        "required_sparsities": [int(v) for v in sparsities],
        "relative_residuals": [float(v) for v in residuals],
    }


def _build_sweep_plot(summary_rows: list[dict[str, Any]], out_path: Path, *, title: str) -> str | None:
    if plt is None or not summary_rows:
        return None
    thresholds = np.asarray([row["residual_threshold"] for row in summary_rows], dtype=np.float64)
    mean_s = np.asarray([row["mean_required_sparsity"] for row in summary_rows], dtype=np.float64)
    median_s = np.asarray([row["median_required_sparsity"] for row in summary_rows], dtype=np.float64)
    q10 = np.asarray([row["sparsity_q10"] for row in summary_rows], dtype=np.float64)
    q90 = np.asarray([row["sparsity_q90"] for row in summary_rows], dtype=np.float64)
    cap = np.asarray([row["patch_cap_share"] for row in summary_rows], dtype=np.float64)
    hit = np.asarray([row["mean_residual_hit_ratio"] for row in summary_rows], dtype=np.float64)
    spread = np.asarray([row["sparsity_range"] for row in summary_rows], dtype=np.float64)
    max_s = float(summary_rows[0].get("max_sparsity", np.nan))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    ax0, ax1, ax2 = axes
    for ax in axes:
        ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.75)
        ax.tick_params(labelsize=12)

    ax0.fill_between(thresholds, q10, q90, color="#4477aa", alpha=0.18, label="q10-q90")
    ax0.plot(thresholds, mean_s, marker="o", linewidth=2.2, color="#4477aa", label="mean")
    ax0.plot(thresholds, median_s, marker="s", linewidth=1.8, linestyle="--", color="#228833", label="median")
    if math.isfinite(max_s):
        ax0.axhline(max_s, color="#cc6677", linestyle=":", linewidth=1.5, label="cap")
    ax0.set_xlabel("target relative residual", fontsize=13)
    ax0.set_ylabel("mean atoms required", fontsize=13)
    ax0.set_title("Sparse Complexity Relaxes Gradually", fontsize=15, pad=9)
    ax0.legend(fontsize=11, frameon=True)

    ax1.plot(thresholds, cap, marker="o", linewidth=2.1, color="#cc6677", label="patches at cap")
    ax1.plot(thresholds, hit, marker="s", linewidth=2.1, color="#117733", label="hit target")
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_xlabel("target relative residual", fontsize=13)
    ax1.set_ylabel("fraction of token neighborhoods", fontsize=13)
    ax1.set_title("Strict Targets Hide Regional Variation", fontsize=15, pad=9)
    ax1.legend(fontsize=11, frameon=True)

    ax2.plot(thresholds, spread, marker="o", linewidth=2.1, color="#aa4499")
    ax2.set_xlabel("target relative residual", fontsize=13)
    ax2.set_ylabel("range of mean atoms", fontsize=13)
    ax2.set_title("Dynamic Range Across Image Regions", fontsize=15, pad=9)

    fig.suptitle(title, fontsize=19, y=1.02)
    fig.tight_layout()
    save_figure(fig, out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _build_combined_model_plot(summary_paths: list[Path], out_path: Path) -> str | None:
    if plt is None or not summary_paths:
        return None
    records: list[tuple[str, list[dict[str, Any]]]] = []
    for path in summary_paths:
        payload = _load_json(path)
        label = str(payload.get("label") or path.parent.name)
        rows = list(payload.get("summary_rows") or [])
        if rows:
            records.append((label, rows))
    if not records:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.0))
    colors = ["#4477aa", "#cc6677", "#228833", "#aa4499", "#66ccee"]
    for ax in axes:
        ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.75)
        ax.tick_params(labelsize=12)

    for idx, (label, rows) in enumerate(records):
        c = colors[idx % len(colors)]
        thresholds = np.asarray([row["residual_threshold"] for row in rows], dtype=np.float64)
        mean_s = np.asarray([row["mean_required_sparsity"] for row in rows], dtype=np.float64)
        q10 = np.asarray([row["sparsity_q10"] for row in rows], dtype=np.float64)
        q90 = np.asarray([row["sparsity_q90"] for row in rows], dtype=np.float64)
        cap = np.asarray([row["patch_cap_share"] for row in rows], dtype=np.float64)
        spread = np.asarray([row["sparsity_range"] for row in rows], dtype=np.float64)
        axes[0].fill_between(thresholds, q10, q90, color=c, alpha=0.10)
        axes[0].plot(thresholds, mean_s, marker="o", linewidth=2.2, color=c, label=label)
        axes[1].plot(thresholds, cap, marker="o", linewidth=2.2, color=c, label=label)
        axes[2].plot(thresholds, spread, marker="o", linewidth=2.2, color=c, label=label)

    axes[0].set_title("Mean Atoms Required", fontsize=15, pad=9)
    axes[0].set_xlabel("target relative residual", fontsize=13)
    axes[0].set_ylabel("OMP atoms", fontsize=13)
    axes[0].legend(fontsize=11, frameon=True)
    axes[1].set_title("Patch Reconstructions at Cap", fontsize=15, pad=9)
    axes[1].set_xlabel("target relative residual", fontsize=13)
    axes[1].set_ylabel("fraction at cap", fontsize=13)
    axes[1].set_ylim(-0.03, 1.03)
    axes[2].set_title("Regional Dynamic Range", fontsize=15, pad=9)
    axes[2].set_xlabel("target relative residual", fontsize=13)
    axes[2].set_ylabel("range of mean atoms", fontsize=13)

    fig.suptitle("Residual-Threshold Sensitivity of Local Sparse Complexity", fontsize=19, y=1.02)
    fig.tight_layout()
    save_figure(fig, out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _compute_summary(rows: list[dict[str, Any]], *, threshold: float, max_sparsity: int, common: dict[str, Any]) -> dict[str, Any]:
    mean_s = np.asarray([row["mean_required_sparsity"] for row in rows], dtype=np.float64)
    median_s = np.asarray([row["median_required_sparsity"] for row in rows], dtype=np.float64)
    residual_hit = np.asarray([row["residual_hit_ratio"] for row in rows], dtype=np.float64)
    dims = np.asarray([row["dimension"] for row in rows], dtype=np.float64)
    irregularity = np.asarray([row["irregularity"] for row in rows], dtype=np.float64)
    finite = mean_s[np.isfinite(mean_s)]
    qs = np.nanquantile(finite, [0.10, 0.25, 0.75, 0.90]) if finite.size else np.asarray([np.nan] * 4)
    patch_sparsities = np.asarray(
        [s for row in rows for s in row.get("required_sparsities", [])],
        dtype=np.float64,
    )
    patch_cap_share = (
        float(np.nanmean(patch_sparsities >= float(max_sparsity)))
        if patch_sparsities.size
        else float("nan")
    )
    token_near_cap_share = (
        float(np.nanmean(mean_s >= 0.90 * float(max_sparsity)))
        if mean_s.size
        else float("nan")
    )
    return {
        **common,
        "residual_threshold": float(threshold),
        "evaluated_tokens": int(len(rows)),
        "mean_required_sparsity": float(np.nanmean(mean_s)) if mean_s.size else float("nan"),
        "median_required_sparsity": float(np.nanmedian(mean_s)) if median_s.size else float("nan"),
        "sparsity_std": float(np.nanstd(mean_s)) if mean_s.size else float("nan"),
        "sparsity_q10": float(qs[0]),
        "sparsity_q90": float(qs[3]),
        "sparsity_iqr": float(qs[2] - qs[1]),
        "sparsity_range": float(np.nanmax(mean_s) - np.nanmin(mean_s)) if finite.size else float("nan"),
        "patch_cap_share": patch_cap_share,
        "token_near_cap_share": token_near_cap_share,
        "mean_residual_hit_ratio": float(np.nanmean(residual_hit)) if residual_hit.size else float("nan"),
        "corr_sparsity_dimension": _finite_corr(mean_s, dims),
        "corr_sparsity_irregularity": _finite_corr(mean_s, irregularity),
    }


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = torch.load(args.embeddings, map_location="cpu")
    embeddings = artifact["embeddings"].detach().cpu()
    images = artifact["images"].detach().cpu()
    image_ids = artifact["image_ids"].detach().cpu()
    bboxes = artifact["bboxes"].detach().cpu()
    fiber_results = _load_json(Path(args.fiber_results))
    thresholds = [float(v) for v in args.thresholds]
    thresholds = sorted(set(thresholds))

    print(f"[residual_sweep] {args.label}: computing pairwise distances for {embeddings.shape[0]} tokens", flush=True)
    dists = _pairwise_distances(embeddings)
    probe_tokens = select_probe_tokens(embeddings, max_tokens=args.max_anchors, image_ids=image_ids)
    fixed_k = int(args.neighbor_k)

    prepared: list[dict[str, Any]] = []
    skipped = 0
    for item_idx, token_idx_raw in enumerate(probe_tokens.tolist(), start=1):
        token_idx = int(token_idx_raw)
        finite = np.flatnonzero(np.isfinite(dists[token_idx]))
        if finite.size == 0:
            skipped += 1
            continue
        order = finite[np.argsort(dists[token_idx, finite])]
        neigh = order[: max(2, min(fixed_k, int(finite.size)))]
        if neigh.size < int(args.min_patches):
            skipped += 1
            continue
        patches = extract_patch_vectors(
            images=images,
            image_ids=image_ids,
            bboxes=bboxes,
            token_indices=neigh,
            patch_size=args.patch_size,
        )
        if patches.shape[0] < 2:
            skipped += 1
            continue
        x_std = _standardize_patch_matrix(patches)
        dictionary, mean = fit_pca_dictionary(patches, dictionary_size=args.dictionary_size)
        if dictionary.shape[0] == 0:
            skipped += 1
            continue
        orthonormal_dictionary = _dictionary_is_orthonormal(dictionary)
        paths = [
            _omp_residual_path(
                row - mean,
                dictionary,
                max_sparsity=args.max_sparsity,
                orthonormal_dictionary=orthonormal_dictionary,
            )
            for row in x_std
        ]
        res = fiber_results[token_idx] if token_idx < len(fiber_results) else {}
        min_p = _min_pvalue(res)
        irregularity = -math.log10(min_p + 1e-12) if math.isfinite(min_p) else 0.0
        prepared.append(
            {
                "token_index": token_idx,
                "anchor": token_idx,
                "patch_count": int(neigh.size),
                "neighborhood_token_indices": [int(v) for v in neigh.tolist()],
                "dimension": _first_dim(res),
                "min_pvalue": min_p,
                "min_change_pvalue": min_change_pvalue(res),
                "min_fiber_violation_pvalue": min_p,
                "irregularity": irregularity,
                "dictionary_atoms": int(dictionary.shape[0]),
                "paths": paths,
            }
        )
        if item_idx % 256 == 0:
            print(f"[residual_sweep] {args.label}: prepared {item_idx}/{len(probe_tokens)} anchors", flush=True)

    common = {
        "label": args.label,
        "dictionary_mode": "local",
        "neighborhood_mode": "knn",
        "neighbor_k": fixed_k,
        "candidate_tokens": int(embeddings.shape[0]),
        "requested_tokens": None if args.max_anchors is None else int(args.max_anchors),
        "skipped_small_neighborhoods": int(skipped),
        "min_patches": int(args.min_patches),
        "dictionary_size": int(args.dictionary_size),
        "max_sparsity": int(args.max_sparsity),
        "coding_algorithm": "omp",
        "heatmap_max_images": int(args.heatmap_images),
        "source_embeddings": str(Path(args.embeddings)),
        "source_fiber_results": str(Path(args.fiber_results)),
    }

    by_threshold: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        rows: list[dict[str, Any]] = []
        for item in prepared:
            coded = _threshold_sparse_code(item["paths"], threshold, args.max_sparsity)
            if coded is None:
                continue
            row = {
                key: value
                for key, value in item.items()
                if key != "paths"
            }
            row.update({"coding_algorithm": "omp", **coded})
            rows.append(row)

        summary = _compute_summary(rows, threshold=threshold, max_sparsity=args.max_sparsity, common=common)
        summary["interpretation"] = (
            f"{args.label}: residual target {threshold:.2f}; mean atoms {summary['mean_required_sparsity']:.2f}, "
            f"q10-q90 {summary['sparsity_q10']:.2f}-{summary['sparsity_q90']:.2f}, "
            f"{summary['patch_cap_share']:.0%} of patch reconstructions at cap."
        )
        tag = f"tau_{threshold:.2f}".replace(".", "p")
        threshold_dir = out_dir / tag
        threshold_dir.mkdir(parents=True, exist_ok=True)
        plot_path = _build_sparse_probe_plot(
            tokens=rows,
            out_path=threshold_dir / f"{args.slug}_{tag}_sparse_dictionary_probe.png",
            residual_threshold=threshold,
            max_sparsity=args.max_sparsity,
            caption=summary["interpretation"],
        )
        heatmap_path = _build_sparse_probe_heatmaps(
            tokens=rows,
            images=images,
            image_ids=image_ids,
            bboxes=bboxes,
            out_path=threshold_dir / f"{args.slug}_{tag}_sparse_dictionary_heatmaps.png",
            max_images=args.heatmap_images,
            caption=summary["interpretation"],
        )
        summary["plot_path"] = plot_path
        summary["heatmap_path"] = heatmap_path
        payload = {"summary": summary, "tokens": rows, "anchors": rows}
        json_path = threshold_dir / f"{args.slug}_{tag}_sparse_dictionary_probe.json"
        json_path.write_text(json.dumps(_to_serializable(payload), indent=2), encoding="utf-8")
        summary["json_path"] = str(json_path)
        by_threshold[f"{threshold:.2f}"] = payload
        summary_rows.append(summary)
        print(
            f"[residual_sweep] {args.label}: tau={threshold:.2f} "
            f"mean={summary['mean_required_sparsity']:.2f} patch_cap={summary['patch_cap_share']:.1%}",
            flush=True,
        )

    sweep_plot = _build_sweep_plot(
        summary_rows,
        out_dir / f"{args.slug}_residual_sweep.png",
        title=f"{args.label} Sparse Residual Sweep",
    )
    csv_path = out_dir / f"{args.slug}_residual_sweep_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        fieldnames = [
            "label",
            "residual_threshold",
            "evaluated_tokens",
            "mean_required_sparsity",
            "median_required_sparsity",
            "sparsity_q10",
            "sparsity_q90",
            "sparsity_range",
            "patch_cap_share",
            "token_near_cap_share",
            "mean_residual_hit_ratio",
            "corr_sparsity_dimension",
            "corr_sparsity_irregularity",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    payload = {
        "label": args.label,
        "slug": args.slug,
        "summary_rows": summary_rows,
        "sweep_plot": sweep_plot,
        "summary_csv": str(csv_path),
        "thresholds": thresholds,
    }
    summary_path = out_dir / f"{args.slug}_residual_sweep_summary.json"
    summary_path.write_text(json.dumps(_to_serializable(payload), indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, help="Saved epoch_000.pt artifact.")
    parser.add_argument("--fiber-results", type=Path, help="Matching fiber_epoch_000.json.")
    parser.add_argument("--out-dir", type=Path, help="Directory for sweep outputs.")
    parser.add_argument("--label", default="model")
    parser.add_argument("--slug", default="model")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.15, 0.20, 0.25, 0.30, 0.40])
    parser.add_argument("--neighbor-k", type=int, default=32)
    parser.add_argument("--min-patches", type=int, default=12)
    parser.add_argument("--max-anchors", type=int, default=None)
    parser.add_argument("--dictionary-size", type=int, default=64)
    parser.add_argument("--max-sparsity", type=int, default=24)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--heatmap-images", type=int, default=16)
    parser.add_argument(
        "--combine",
        type=Path,
        nargs="*",
        default=None,
        help="Instead of running a model sweep, combine existing residual_sweep_summary.json files.",
    )
    parser.add_argument("--combined-out", type=Path, default=None)
    args = parser.parse_args()
    if args.combine is not None:
        if not args.combine:
            parser.error("--combine needs at least one summary JSON")
        if args.combined_out is None:
            parser.error("--combined-out is required with --combine")
        return args
    for required in ("embeddings", "fiber_results", "out_dir"):
        if getattr(args, required) is None:
            parser.error(f"--{required.replace('_', '-')} is required")
    return args


def main() -> None:
    args = parse_args()
    if args.combine is not None:
        out_path = _build_combined_model_plot([Path(p) for p in args.combine], Path(args.combined_out))
        print(f"[residual_sweep] combined plot: {out_path}", flush=True)
        return
    payload = run_sweep(args)
    print(f"[residual_sweep] wrote {payload['slug']} summary to {Path(payload['summary_csv']).parent}", flush=True)


if __name__ == "__main__":
    main()
