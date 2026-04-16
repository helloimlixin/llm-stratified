#!/usr/bin/env python3
"""Aggregate volume-probe sweep results and render a markdown report.

Supports both:
- pixel-space sweeps from ``scripts/run_pixel_stratification_sweep.sh``
- pretrained-vs-pixel sweeps from ``scripts/run_pretrained_pixel_sweep.sh``
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SUMMARY_CANDIDATES = (
    Path("volume_probe") / "volume_summary.json",
    Path("volume_summary.json"),
)


def _load_summary(run_dir: Path) -> dict[str, Any] | None:
    for rel_path in SUMMARY_CANDIDATES:
        path = run_dir / rel_path
        if path.exists():
            with open(path) as handle:
                return json.load(handle)
    return None


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _normalize_layers(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            parsed = _safe_int(item)
            if parsed is not None:
                out.append(parsed)
        return tuple(out)
    parsed = _safe_int(value)
    return (parsed,) if parsed is not None else ()


def _parse_tag(tag: str) -> dict[str, Any]:
    patterns = [
        (
            re.compile(r"^(?P<dataset>[a-z0-9]+)_ps(?P<patch_size>\d+)_stride(?P<stride>\d+)$"),
            lambda m: {
                "dataset": m.group("dataset").upper(),
                "patch_size": int(m.group("patch_size")),
                "pixel_stride": int(m.group("stride")),
            },
        ),
        (
            re.compile(r"^(?P<dataset>[a-z0-9]+)_untrained_ps(?P<patch_size>\d+)$"),
            lambda m: {
                "dataset": m.group("dataset").upper(),
                "patch_size": int(m.group("patch_size")),
                "pixel_stride": int(m.group("patch_size")),
                "variant_hint": "untrained_tinyvit",
            },
        ),
        (
            re.compile(r"^(?P<dataset>[a-z0-9]+)_dinov2_(?P<layer_tag>[a-z0-9]+)_(?P<stride_tag>[a-z0-9]+)$"),
            lambda m: {
                "dataset": m.group("dataset").upper(),
                "variant_hint": f"dinov2_{m.group('layer_tag')}",
                "stride_hint": m.group("stride_tag"),
            },
        ),
    ]
    for pattern, handler in patterns:
        match = pattern.match(tag)
        if match:
            return handler(match)
    return {"dataset": tag.upper()}


def _layer_variant_name(layers: tuple[int, ...]) -> str:
    if not layers:
        return "dinov2_last"
    if layers == (3, 6, -1):
        return "dinov2_multilayer"
    joined = ",".join(str(v) for v in layers)
    return f"dinov2_layers_{joined}"


def _classify_run(run_dir: Path, summary: dict[str, Any], sweep_dir: Path) -> dict[str, Any]:
    tag = run_dir.name
    parsed = _parse_tag(tag)
    config = summary.get("config", {})

    dataset = str(config.get("dataset") or parsed.get("dataset") or "?").upper()
    patch_size = _safe_int(config.get("patch_size"))
    if patch_size is None:
        patch_size = parsed.get("patch_size")

    pixel_stride = config.get("pixel_patch_stride")
    if pixel_stride is None:
        pixel_stride = parsed.get("pixel_stride")
    pixel_stride = patch_size if pixel_stride is None and patch_size is not None else _safe_int(pixel_stride)

    feature_backbone = str(config.get("feature_backbone") or "").strip().lower()
    if not feature_backbone:
        feature_backbone = "dinov2" if "dinov2" in tag else "model"

    layers = _normalize_layers(config.get("dinov2_layers"))
    if feature_backbone == "dinov2":
        sweep_kind = "pretrained"
        variant = _layer_variant_name(layers)
    elif "untrained" in tag or parsed.get("variant_hint") == "untrained_tinyvit":
        sweep_kind = "pretrained"
        variant = "untrained_tinyvit"
    else:
        sweep_kind = "pixel"
        variant = "pixel_tinyvit"

    return {
        "tag": tag,
        "run_dir": str(run_dir),
        "sweep_dir": str(sweep_dir),
        "sweep_name": sweep_dir.name,
        "dataset": dataset,
        "patch_size": patch_size,
        "pixel_stride": pixel_stride,
        "feature_backbone": feature_backbone,
        "variant": variant,
        "dinov2_layers": list(layers) if layers else None,
    }


def _collect_rows(sweep_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not sweep_dir.is_dir():
        raise FileNotFoundError(f"sweep dir not found: {sweep_dir}")

    direct_summary = _load_summary(sweep_dir)
    if direct_summary is not None:
        row = _classify_run(sweep_dir, direct_summary, sweep_dir)
        row["representations"] = {}
        for rep_name, rep_data in direct_summary.get("representations", {}).items():
            rep_summary = rep_data.get("summary", {})
            row["representations"][rep_name] = {
                "mean_dim": _safe_float(rep_summary.get("mean_dim")),
                "median_dim": _safe_float(rep_summary.get("median_dim")),
                "irregular_ratio": _safe_float(rep_summary.get("irregular_ratio")),
                "mean_irregularity": _safe_float(rep_summary.get("mean_irregularity")),
                "num_tokens": _safe_int(rep_summary.get("num_tokens")) or 0,
            }
        return [row], []

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for child in sorted(sweep_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name in {"checkpoints", "volume_probe", "embeddings", "fiber_analysis", "wandb"}:
            continue
        summary = _load_summary(child)
        if summary is None:
            missing.append(str(child))
            continue
        row = _classify_run(child, summary, sweep_dir)
        row["representations"] = {}
        for rep_name, rep_data in summary.get("representations", {}).items():
            rep_summary = rep_data.get("summary", {})
            row["representations"][rep_name] = {
                "mean_dim": _safe_float(rep_summary.get("mean_dim")),
                "median_dim": _safe_float(rep_summary.get("median_dim")),
                "irregular_ratio": _safe_float(rep_summary.get("irregular_ratio")),
                "mean_irregularity": _safe_float(rep_summary.get("mean_irregularity")),
                "num_tokens": _safe_int(rep_summary.get("num_tokens")) or 0,
            }
        rows.append(row)
    return rows, missing


def _primary_pixel_rep_name(row: dict[str, Any]) -> str | None:
    representations = row.get("representations", {})
    stride = row.get("pixel_stride")
    patch_size = row.get("patch_size")
    if stride is None or patch_size is None or stride == patch_size:
        return "patch_pixels" if "patch_pixels" in representations else None
    candidate = f"patch_pixels_stride_{stride}"
    if candidate in representations:
        return candidate
    return "patch_pixels" if "patch_pixels" in representations else None


def _primary_token_rep_name(row: dict[str, Any]) -> str | None:
    representations = row.get("representations", {})
    for name in ("tokens", "tokens_layer_last", "tokens_layer_last1"):
        if name in representations:
            return name
    last_layer_names = sorted(name for name in representations if name.startswith("tokens_layer_last"))
    if last_layer_names:
        return last_layer_names[0]
    layer_names = sorted(name for name in representations if name.startswith("tokens_layer_"))
    return layer_names[-1] if layer_names else None


def _rep_metric(row: dict[str, Any], rep_name: str | None, metric: str) -> float:
    if not rep_name:
        return float("nan")
    return _safe_float(row.get("representations", {}).get(rep_name, {}).get(metric))


def _metric_cell(row: dict[str, Any], rep_name: str | None) -> str:
    mean_dim = _rep_metric(row, rep_name, "mean_dim")
    irregular_ratio = _rep_metric(row, rep_name, "irregular_ratio")
    if not np.isfinite(mean_dim) and not np.isfinite(irregular_ratio):
        return "n/a"
    dim_txt = "nan" if not np.isfinite(mean_dim) else f"{mean_dim:.2f}"
    irr_txt = "nan" if not np.isfinite(irregular_ratio) else f"{irregular_ratio:.3f}"
    return f"{dim_txt} / {irr_txt}"


def _fmt_float(value: float, digits: int = 3) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.{digits}f}"


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No completed runs found._"
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join([header_line, sep_line, body])


def _findings_pixel(rows: list[dict[str, Any]]) -> list[str]:
    findings: list[str] = []
    pixel_rows = [row for row in rows if row.get("sweep_kind") == "pixel"]
    if not pixel_rows:
        return findings

    comparisons = []
    grouped: dict[tuple[str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in pixel_rows:
        dataset = row.get("dataset")
        patch_size = row.get("patch_size")
        stride = row.get("pixel_stride")
        if dataset is None or patch_size is None or stride is None:
            continue
        grouped[(dataset, patch_size)][stride] = row

    for (dataset, patch_size), by_stride in grouped.items():
        full = by_stride.get(patch_size)
        overlap = next((row for stride, row in sorted(by_stride.items()) if stride < patch_size), None)
        if not full or not overlap:
            continue
        full_rep = _primary_pixel_rep_name(full)
        overlap_rep = _primary_pixel_rep_name(overlap)
        full_irr = _rep_metric(full, full_rep, "irregular_ratio")
        overlap_irr = _rep_metric(overlap, overlap_rep, "irregular_ratio")
        if np.isfinite(full_irr) and np.isfinite(overlap_irr):
            comparisons.append((dataset, patch_size, full_irr, overlap_irr, overlap_irr - full_irr))

    if comparisons:
        overlap_worse = sum(delta > 0 for *_head, delta in comparisons)
        biggest = max(comparisons, key=lambda item: item[-1])
        findings.append(
            f"Overlapping raw patches increased irregularity in {overlap_worse}/{len(comparisons)} "
            f"matched dataset/patch pairs; the largest jump was {biggest[0]} patch {biggest[1]} "
            f"({biggest[2]:.3f} -> {biggest[3]:.3f})."
        )

    raw_rows = []
    token_deltas = []
    for row in pixel_rows:
        pixel_rep = _primary_pixel_rep_name(row)
        token_rep = _primary_token_rep_name(row)
        raw_irr = _rep_metric(row, pixel_rep, "irregular_ratio")
        token_irr = _rep_metric(row, token_rep, "irregular_ratio")
        if np.isfinite(raw_irr):
            raw_rows.append((raw_irr, row))
        if np.isfinite(raw_irr) and np.isfinite(token_irr):
            token_deltas.append(token_irr - raw_irr)

    if raw_rows:
        max_raw, max_row = max(raw_rows, key=lambda item: item[0])
        min_raw, min_row = min(raw_rows, key=lambda item: item[0])
        findings.append(
            f"The most stratified raw-pixel configuration was {max_row['dataset']} patch {max_row['patch_size']} "
            f"stride {max_row['pixel_stride']} (irregular_ratio={max_raw:.3f}), while the smoothest completed raw-pixel "
            f"run was {min_row['dataset']} patch {min_row['patch_size']} stride {min_row['pixel_stride']} "
            f"(irregular_ratio={min_raw:.3f})."
        )

    if token_deltas:
        findings.append(
            f"Across the pure pixel sweep, token irregularity was {_fmt_float(float(np.mean(token_deltas)), 3)} "
            f"higher than the matched raw-pixel probe on average (mean token-minus-pixel delta)."
        )

    return findings


def _findings_pretrained(rows: list[dict[str, Any]]) -> list[str]:
    findings: list[str] = []
    pretrained_rows = [row for row in rows if row.get("sweep_kind") == "pretrained"]
    if not pretrained_rows:
        return findings

    dino_rows = [row for row in pretrained_rows if row.get("feature_backbone") == "dinov2"]
    untrained_rows = [row for row in pretrained_rows if row.get("variant") == "untrained_tinyvit"]

    dino_improves = []
    for row in dino_rows:
        pixel_rep = _primary_pixel_rep_name(row)
        token_rep = _primary_token_rep_name(row)
        raw_irr = _rep_metric(row, pixel_rep, "irregular_ratio")
        token_irr = _rep_metric(row, token_rep, "irregular_ratio")
        if np.isfinite(raw_irr) and np.isfinite(token_irr):
            dino_improves.append((row, raw_irr, token_irr))
    if dino_improves:
        better = sum(token_irr < raw_irr for _row, raw_irr, token_irr in dino_improves)
        findings.append(
            f"DINO token representations were less irregular than their matched raw-pixel probes in "
            f"{better}/{len(dino_improves)} completed DINO runs "
            f"(mean raw={_fmt_float(float(np.mean([raw for _row, raw, _tok in dino_improves])), 3)}, "
            f"mean token={_fmt_float(float(np.mean([tok for _row, _raw, tok in dino_improves])), 3)})."
        )

    layer_triples = []
    for row in dino_rows:
        rep03 = _rep_metric(row, "tokens_layer_03", "irregular_ratio")
        rep06 = _rep_metric(row, "tokens_layer_06", "irregular_ratio")
        replast = _rep_metric(row, "tokens_layer_last", "irregular_ratio")
        if np.isfinite(rep03) and np.isfinite(rep06) and np.isfinite(replast):
            layer_triples.append((row, rep03, rep06, replast))
    if layer_triples:
        monotone = sum(rep03 >= rep06 >= replast for _row, rep03, rep06, replast in layer_triples)
        findings.append(
            f"In the multilayer DINO runs, irregularity decreased monotonically from layer 03 to 06 to last in "
            f"{monotone}/{len(layer_triples)} cases."
        )

    if dino_rows and untrained_rows:
        dino_by_dataset = {
            row["dataset"]: row for row in dino_rows if row.get("variant") == "dinov2_last" and row.get("pixel_stride") == row.get("patch_size")
        }
        compare = []
        for row in untrained_rows:
            dino_row = dino_by_dataset.get(row["dataset"])
            if not dino_row:
                continue
            dino_irr = _rep_metric(dino_row, _primary_token_rep_name(dino_row), "irregular_ratio")
            untrained_irr = _rep_metric(row, _primary_token_rep_name(row), "irregular_ratio")
            if np.isfinite(dino_irr) and np.isfinite(untrained_irr):
                compare.append((row["dataset"], dino_irr, untrained_irr))
        if compare:
            better = sum(dino_irr < untrained_irr for _dataset, dino_irr, untrained_irr in compare)
            findings.append(
                f"Against the included untrained TinyViT baseline, DINO last-layer token irregularity was lower on "
                f"{better}/{len(compare)} shared datasets."
            )

    return findings


def _build_markdown_report(
    *,
    rows: list[dict[str, Any]],
    missing: list[str],
    sweep_dirs: list[Path],
    title: str,
    output_json: Path,
) -> str:
    for row in rows:
        # Cache classification for finding helpers.
        if row.get("feature_backbone") == "dinov2" or row.get("variant") in {"untrained_tinyvit"}:
            row["sweep_kind"] = "pretrained"
        elif row.get("variant") == "pixel_tinyvit":
            row["sweep_kind"] = "pixel"

    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    sections = [f"# {title}", ""]
    sections.append(f"Generated: {generated}")
    sections.append("")
    sections.append("## Inputs")
    for sweep_dir in sweep_dirs:
        sections.append(f"- `{sweep_dir}`")
    sections.append(f"- Aggregated JSON: `{output_json}`")
    sections.append("")

    sections.append("## Coverage")
    coverage_rows = []
    for sweep_dir in sweep_dirs:
        complete = sum(1 for row in rows if row.get("sweep_dir") == str(sweep_dir))
        incomplete = sum(1 for item in missing if str(sweep_dir) in item)
        coverage_rows.append([sweep_dir.name, str(complete), str(incomplete)])
    sections.append(_render_table(["sweep", "completed_runs", "incomplete_dirs"], coverage_rows))
    if missing:
        sections.append("")
        sections.append("Incomplete run directories:")
        for item in missing:
            sections.append(f"- `{item}`")
    sections.append("")

    sections.append("## Key Findings")
    findings = _findings_pixel(rows) + _findings_pretrained(rows)
    if findings:
        for finding in findings:
            sections.append(f"- {finding}")
    else:
        sections.append("- No completed runs were available to summarize.")
    sections.append("")

    pixel_rows = sorted(
        (row for row in rows if row.get("sweep_kind") == "pixel"),
        key=lambda row: (row.get("dataset", ""), row.get("patch_size") or 0, row.get("pixel_stride") or 0),
    )
    sections.append("## Pixel Sweep")
    sections.append("Metric cells use `mean_dim / irregular_ratio`.")
    pixel_table = []
    for row in pixel_rows:
        pixel_table.append(
            [
                row.get("dataset", "?"),
                str(row.get("patch_size", "?")),
                str(row.get("pixel_stride", "?")),
                _metric_cell(row, _primary_pixel_rep_name(row)),
                _metric_cell(row, _primary_token_rep_name(row)),
                _metric_cell(row, "patch_embeddings"),
            ]
        )
    sections.append(
        _render_table(
            ["dataset", "patch", "stride", "raw_pixels", "tokens", "patch_embeddings"],
            pixel_table,
        )
    )
    sections.append("")

    pretrained_rows = sorted(
        (row for row in rows if row.get("sweep_kind") == "pretrained"),
        key=lambda row: (row.get("dataset", ""), row.get("variant", ""), row.get("pixel_stride") or 0),
    )
    sections.append("## Pretrained Sweep")
    sections.append("Metric cells use `mean_dim / irregular_ratio`.")
    pretrained_table = []
    for row in pretrained_rows:
        pretrained_table.append(
            [
                row.get("dataset", "?"),
                row.get("variant", "?"),
                str(row.get("patch_size", "?")),
                str(row.get("pixel_stride", "?")),
                _metric_cell(row, _primary_pixel_rep_name(row)),
                _metric_cell(row, "patch_embeddings"),
                _metric_cell(row, _primary_token_rep_name(row)),
                _metric_cell(row, "tokens_layer_03"),
                _metric_cell(row, "tokens_layer_06"),
            ]
        )
    sections.append(
        _render_table(
            [
                "dataset",
                "variant",
                "patch",
                "stride",
                "raw_pixels",
                "patch_embeddings",
                "tokens_or_last",
                "layer_03",
                "layer_06",
            ],
            pretrained_table,
        )
    )
    sections.append("")

    sections.append("## Notes")
    sections.append("- `raw_pixels` refers to the primary raw-pixel probe for that run: `patch_pixels` for non-overlapping patches or `patch_pixels_stride_<k>` when overlap was enabled.")
    sections.append("- `tokens_or_last` maps to `tokens` for TinyViT runs and to the last-layer token representation for DINO runs.")
    sections.append("- Comparisons across the DINO and untrained TinyViT rows are informative but not perfectly apples-to-apples because their patch grids differ.")
    sections.append("")
    return "\n".join(sections).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze volume-probe sweep outputs and render markdown.")
    parser.add_argument("--sweep-dir", dest="sweep_dirs", action="append", required=True, type=Path)
    parser.add_argument("--output", default=None, type=Path, help="Save aggregated JSON rows.")
    parser.add_argument("--report", default=None, type=Path, help="Write markdown report to this path.")
    parser.add_argument("--title", default="Volume Probe Sweep Report")
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for sweep_dir in args.sweep_dirs:
        rows, missing_dirs = _collect_rows(sweep_dir)
        all_rows.extend(rows)
        missing.extend(missing_dirs)

    if not all_rows:
        print("No completed results found.", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        default_name = "combined_sweep_analysis.json" if len(args.sweep_dirs) > 1 else "sweep_analysis.json"
        output_path = args.sweep_dirs[0].parent / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(all_rows, handle, indent=2)

    markdown = _build_markdown_report(
        rows=all_rows,
        missing=missing,
        sweep_dirs=args.sweep_dirs,
        title=args.title,
        output_json=output_path,
    )

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(markdown)
        print(f"Saved markdown report -> {args.report}")
    else:
        print(markdown)
    print(f"Saved aggregated results -> {output_path}")


if __name__ == "__main__":
    main()
