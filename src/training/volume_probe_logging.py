from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from training.wandb_utils import finish_wandb_run, init_wandb_run


SUMMARY_METRICS = (
    ("num_tokens", "num_points"),
    ("tokens_with_strata", "points_with_strata"),
    ("mean_dim", "mean_dim"),
    ("median_dim", "median_dim"),
    ("min_pvalue", "min_pvalue"),
    ("max_pvalue", "max_pvalue"),
    ("mean_irregularity", "mean_irregularity"),
    ("max_irregularity", "max_irregularity"),
    ("irregular_ratio", "irregular_ratio"),
)


def resolve_volume_probe_tags(tags: Any) -> list[str]:
    try:
        resolved = list(tags) if tags is not None else []
    except Exception:
        resolved = []
    if "volume-probe" not in resolved:
        resolved.append("volume-probe")
    return resolved


def _representations(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reps = (results or {}).get("representations", {}) or {}
    return reps if isinstance(reps, dict) else {}


def _representation_items(results: dict[str, Any]):
    for rep_name, rep in _representations(results).items():
        yield str(rep_name), rep if isinstance(rep, dict) else {}


def _viz_entries(results: dict[str, Any]):
    viz = (results or {}).get("viz", {}) or {}
    return viz.items() if isinstance(viz, dict) else ()


def _existing_output_path(output_dir: Path, filename: Any) -> Path | None:
    if not isinstance(filename, str) or not filename:
        return None
    path = output_dir / filename
    return path if path.exists() else None


def extract_volume_probe_curves(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    curves: dict[str, dict[str, Any]] = {}
    for rep_name, rep in _representation_items(results):
        knn = rep.get("knn_curve")
        if not (isinstance(knn, dict) and knn.get("k_values") and knn.get("radii")):
            continue
        try:
            ks = [int(k) for k in list(knn.get("k_values"))]
            radii = {}
            for quantile_name, values in dict(knn.get("radii") or {}).items():
                if isinstance(values, list) and ks and values and len(values) == len(ks):
                    radii[str(quantile_name)] = [float(x) for x in values]
            if ks and radii:
                curves[rep_name] = {
                    "ks": ks,
                    "radii": radii,
                    "k_min": int(knn.get("k_min", ks[0])),
                    "k_max": int(knn.get("k_max", ks[-1])),
                }
        except Exception:
            continue
    return curves


def _numeric_metric(value: Any) -> int | float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _fmt_metric(value: Any, *, digits: int = 2, default: str = "n/a") -> str:
    numeric = _numeric_metric(value)
    if numeric is None:
        return default
    return f"{float(numeric):.{digits}f}"


def _wandb_image_with_caption(wandb, path: Path, caption: str):
    try:
        return wandb.Image(str(path), caption=caption)
    except TypeError:
        return wandb.Image(str(path))


def _summary_for_rep(results: dict[str, Any], rep_name: str) -> dict[str, Any]:
    rep = _representations(results).get(rep_name)
    return (rep or {}).get("summary", {}) if isinstance(rep, dict) else {}


def _volume_probe_viz_caption(results: dict[str, Any], key: str) -> str:
    rep_name = ""
    plot_kind = key
    for prefix in ("detail_", "scaling_", "nn_irregular_", "nn_"):
        if key.startswith(prefix):
            rep_name = key[len(prefix):]
            plot_kind = prefix.rstrip("_")
            break
    summary = _summary_for_rep(results, rep_name) if rep_name else {}
    mean_dim = _fmt_metric(summary.get("mean_dim"))
    irregular_ratio = _fmt_metric(summary.get("irregular_ratio"), digits=3)
    num_tokens = _fmt_metric(summary.get("num_tokens"), digits=0)

    if plot_kind == "detail":
        conclusion = (
            "Use this dashboard to decide whether local dimension and irregularity are broadly distributed or concentrated in specific projected regions. "
            f"Conclusion: representation {rep_name or 'n/a'} has mean dimension {mean_dim}, irregular ratio {irregular_ratio}, and {num_tokens} probed points; coherent colored regions suggest structured geometry, while speckled regions suggest abrupt local variation."
        )
    elif plot_kind == "scaling":
        conclusion = (
            "Use these scaling curves to inspect whether log neighbor count grows linearly with log radius and where change points split local strata. "
            f"Conclusion: representation {rep_name or 'n/a'} should be read as more manifold-like when curves are smooth with stable slopes, and more stratified when dashed change-point lines split visibly different slopes; its mean dimension is {mean_dim} with irregular ratio {irregular_ratio}."
        )
    elif plot_kind == "nn_irregular":
        conclusion = (
            "Use this nearest-neighbor grid to audit the most irregular anchors visually. "
            f"Conclusion: if neighbors share object parts or textures, irregularity may reflect real image structure; if they are visually unrelated, the representation neighborhood is semantically mixed. Representation {rep_name or 'n/a'} has irregular ratio {irregular_ratio}."
        )
    elif plot_kind == "nn":
        conclusion = (
            "Use this nearest-neighbor grid as a qualitative locality check. "
            f"Conclusion: visually coherent rows mean the representation preserves patch-level similarity; mixed rows indicate that geometric neighbors are not visually local. Representation {rep_name or 'n/a'} has mean dimension {mean_dim}."
        )
    elif key.startswith("example_"):
        conclusion = "Example patch grid. Conclusion: use this as a scale and content sanity check before interpreting geometric statistics."
    elif key == "example_images":
        conclusion = "Example image grid. Conclusion: use this to verify the sampled dataset, preprocessing scale, and image content for the probe run."
    else:
        conclusion = "Volume probe visualization. Conclusion: use this plot together with the numeric summary table to decide whether local geometry is smooth, stratified, or dominated by preprocessing artifacts."
    return conclusion


def _summary_payload(summary: dict[str, Any]) -> dict[str, int | float]:
    payload: dict[str, int | float] = {}
    for source_name, metric_name in SUMMARY_METRICS:
        numeric = _numeric_metric(summary.get(source_name))
        if numeric is not None:
            payload[metric_name] = numeric
    return payload


def _load_dims(output_dir: Path, filename: Any) -> np.ndarray | None:
    path = _existing_output_path(output_dir, filename)
    if path is None:
        return None
    try:
        values = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    except Exception:
        return None
    values = values[np.isfinite(values)]
    return values if values.size else None


def build_volume_probe_summary_rows(results: dict[str, Any]) -> list[dict[str, int | float | str]]:
    curves = extract_volume_probe_curves(results)
    rows: list[dict[str, int | float | str]] = []
    for rep_name, rep in _representation_items(results):
        row: dict[str, int | float | str] = {"representation": rep_name}
        row.update(_summary_payload(rep.get("summary", {}) or {}))
        curve = curves.get(rep_name)
        if curve is not None:
            row["k_min"] = curve["k_min"]
            row["k_max"] = curve["k_max"]
        rows.append(row)
    return rows


def build_volume_probe_curve_table_rows(results: dict[str, Any]) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for rep_name, curve in extract_volume_probe_curves(results).items():
        for idx, k in enumerate(curve["ks"]):
            row: dict[str, int | float | str] = {"representation": rep_name, "k": int(k)}
            for quantile_name, values in curve["radii"].items():
                row[f"radius_{quantile_name}"] = float(values[idx])
            rows.append(row)
    return rows


def _maybe_table(wandb, rows: list[dict[str, int | float | str]]):
    if not rows or not hasattr(wandb, "Table"):
        return None
    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)
    data = [[row.get(column) for column in columns] for row in rows]
    return wandb.Table(columns=columns, data=data)


def _maybe_histogram(wandb, values: np.ndarray | None):
    if values is None or not hasattr(wandb, "Histogram"):
        return None
    return wandb.Histogram(values)


def build_volume_probe_log_payload(results: dict[str, Any], output_dir: Path, wandb) -> dict[str, object]:
    payload: dict[str, object] = {}
    curves = extract_volume_probe_curves(results)

    for rep_name, rep in _representation_items(results):
        summary = rep.get("summary", {}) or {}
        for metric_name, value in _summary_payload(summary).items():
            payload[f"volume_probe/{rep_name}/{metric_name}"] = value

        curve = curves.get(rep_name)
        if curve is not None:
            payload[f"volume_probe/{rep_name}/k_min"] = curve["k_min"]
            payload[f"volume_probe/{rep_name}/k_max"] = curve["k_max"]

        dims_hist = _maybe_histogram(wandb, _load_dims(output_dir, rep.get("dims_path")))
        if dims_hist is not None:
            payload[f"volume_probe/{rep_name}/dimension_hist"] = dims_hist

    summary_table = _maybe_table(wandb, build_volume_probe_summary_rows(results))
    if summary_table is not None:
        payload["volume_probe/representation_summary"] = summary_table

    curve_table = _maybe_table(wandb, build_volume_probe_curve_table_rows(results))
    if curve_table is not None:
        payload["volume_probe/curve_table"] = curve_table

    if hasattr(wandb, "Image"):
        for key, filename in _viz_entries(results):
            path = _existing_output_path(output_dir, filename)
            if path is not None:
                caption = _volume_probe_viz_caption(results, str(key))
                payload[f"volume_probe/viz/{key}"] = _wandb_image_with_caption(wandb, path, caption)
                payload[f"volume_probe/viz/{key}_caption"] = caption

    return payload


def build_volume_probe_curve_rows(results: dict[str, Any]) -> list[tuple[int, dict[str, float]]]:
    curves = extract_volume_probe_curves(results)
    base_rep = "tokens" if "tokens" in curves else (next(iter(curves.keys())) if curves else None)
    if base_rep is None:
        return []

    base_ks = curves[base_rep]["ks"]
    idx_maps = {name: {k: i for i, k in enumerate(curve["ks"])} for name, curve in curves.items()}
    rows: list[tuple[int, dict[str, float]]] = []

    for k in base_ks:
        row = {"volume_probe/k": int(k)}
        for rep_name, curve in curves.items():
            idx = idx_maps[rep_name].get(int(k))
            if idx is None:
                continue
            for quantile_name, values in curve["radii"].items():
                row[f"volume_probe/{rep_name}/radius_{quantile_name}"] = float(values[idx])
        rows.append((int(k), row))
    return rows


def collect_volume_probe_artifact_paths(results: dict[str, Any], output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    summary_path = _existing_output_path(output_dir, "volume_summary.json")
    if summary_path is not None:
        paths.append(summary_path)

    for _key, filename in _viz_entries(results):
        path = _existing_output_path(output_dir, filename)
        if path is not None:
            paths.append(path)

    for _rep_name, rep in _representation_items(results):
        for key in ("dims_path", "results_path"):
            path = _existing_output_path(output_dir, rep.get(key))
            if path is not None:
                paths.append(path)

    return paths


def log_volume_probe_to_wandb(
    *,
    enabled: bool,
    project: str,
    name: str,
    tags: Any,
    results: dict[str, Any],
    output_dir: Path,
) -> None:
    wandb = init_wandb_run(
        enabled=enabled,
        project=project,
        name=name,
        tags=resolve_volume_probe_tags(tags),
        config=(results or {}).get("config", {}),
        missing_message="[wandb] ERROR: volume-probe logging disabled; wandb is not installed",
        show_url=True,
    )
    if wandb is None:
        return

    try:
        payload = build_volume_probe_log_payload(results, output_dir, wandb)
        if payload:
            wandb.log(payload, step=0)

        for step, row in build_volume_probe_curve_rows(results):
            wandb.log(row, step=step)

        artifact_paths = collect_volume_probe_artifact_paths(results, output_dir)
        if artifact_paths:
            artifact_name = f"{wandb.run.name}_volume_probe" if wandb.run else "volume_probe"
            artifact = wandb.Artifact(artifact_name, type="volume_probe")
            for path in artifact_paths:
                artifact.add_file(str(path))
            wandb.log_artifact(artifact)
    except Exception as exc:
        print(f"[wandb] ERROR: {exc}")
    finally:
        finish_wandb_run(wandb)
