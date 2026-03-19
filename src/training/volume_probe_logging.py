from __future__ import annotations

from pathlib import Path
from typing import Any

from training.wandb_utils import finish_wandb_run, init_wandb_run


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


def extract_volume_probe_curves(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    curves: dict[str, dict[str, Any]] = {}
    for rep_name, rep in _representations(results).items():
        knn = (rep or {}).get("knn_curve")
        if not (isinstance(knn, dict) and knn.get("k_values") and knn.get("radii")):
            continue
        try:
            ks = [int(k) for k in list(knn.get("k_values"))]
            q50 = dict(knn.get("radii") or {}).get("q50")
            if isinstance(q50, list) and ks and q50 and len(q50) == len(ks):
                curves[rep_name] = {
                    "ks": ks,
                    "q50": [float(x) for x in q50],
                    "k_min": int(knn.get("k_min", ks[0])),
                    "k_max": int(knn.get("k_max", ks[-1])),
                }
        except Exception:
            continue
    return curves


def build_volume_probe_log_payload(results: dict[str, Any], output_dir: Path, wandb) -> dict[str, object]:
    payload: dict[str, object] = {}
    curves = extract_volume_probe_curves(results)

    for rep_name, rep in _representations(results).items():
        summary = (rep or {}).get("summary", {}) or {}
        num_points = summary.get("num_tokens")
        if num_points is not None:
            payload[f"volume_probe/{rep_name}/num_points"] = num_points

        curve = curves.get(rep_name)
        if curve is not None:
            payload[f"volume_probe/{rep_name}/k_min"] = curve["k_min"]
            payload[f"volume_probe/{rep_name}/k_max"] = curve["k_max"]

    viz = (results or {}).get("viz", {}) or {}
    if isinstance(viz, dict):
        for key, filename in viz.items():
            if not isinstance(filename, str) or not filename:
                continue
            path = output_dir / filename
            if path.exists():
                payload[f"volume_probe/viz/{key}"] = wandb.Image(str(path))

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
            row[f"volume_probe/{rep_name}/radius_q50"] = float(curve["q50"][idx])
        rows.append((int(k), row))
    return rows


def collect_volume_probe_artifact_paths(results: dict[str, Any], output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    summary_path = output_dir / "volume_summary.json"
    if summary_path.exists():
        paths.append(summary_path)

    viz = (results or {}).get("viz", {}) or {}
    if isinstance(viz, dict):
        for filename in viz.values():
            if not isinstance(filename, str) or not filename:
                continue
            path = output_dir / filename
            if path.exists():
                paths.append(path)

    for rep in _representations(results).values():
        for key in ("dims_path", "results_path"):
            filename = (rep or {}).get(key)
            if not isinstance(filename, str) or not filename:
                continue
            path = output_dir / filename
            if path.exists():
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
