#!/usr/bin/env python3
"""Upload completed volume-probe sweep outputs to Weights & Biases.

This script reuses the volume-probe W&B payload builders so finished runs can
be uploaded after the fact without rerunning the sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.volume_probe_logging import (  # noqa: E402
    build_volume_probe_curve_rows,
    build_volume_probe_log_payload,
    collect_volume_probe_artifact_paths,
    resolve_volume_probe_tags,
)
from training.wandb_utils import finish_wandb_run, init_wandb_run  # noqa: E402


SUMMARY_CANDIDATES = (
    Path("volume_probe") / "volume_summary.json",
    Path("volume_summary.json"),
)


def _load_summary(run_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    for rel_path in SUMMARY_CANDIDATES:
        path = run_dir / rel_path
        if path.exists():
            with open(path) as handle:
                return path, json.load(handle)
    return None


def _iter_completed_runs(sweep_dir: Path) -> list[tuple[Path, Path, dict[str, Any]]]:
    found: list[tuple[Path, Path, dict[str, Any]]] = []
    direct = _load_summary(sweep_dir)
    if direct is not None:
        summary_path, summary = direct
        found.append((sweep_dir, summary_path.parent, summary))
        return found

    for child in sorted(sweep_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name in {"wandb", "volume_probe", "checkpoints", "embeddings", "fiber_analysis"}:
            continue
        loaded = _load_summary(child)
        if loaded is None:
            continue
        summary_path, summary = loaded
        found.append((child, summary_path.parent, summary))
    return found


def _make_wandb_dir(run_dir: Path, suffix: str) -> Path:
    target = run_dir / suffix
    target.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(target)
    return target


def _prepare_wandb_storage(run_dir: Path, *, cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    data_dir = cache_root / "data"
    artifact_dir = cache_root / "artifacts"
    tmp_dir = cache_root / "tmp"
    for path in (data_dir, artifact_dir, tmp_dir):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_CACHE_DIR"] = str(cache_root)
    os.environ["WANDB_DATA_DIR"] = str(data_dir)
    os.environ["WANDB_ARTIFACT_DIR"] = str(artifact_dir)
    os.environ.setdefault("TMPDIR", str(tmp_dir))
    tempfile.tempdir = os.environ["TMPDIR"]


def _upload_volume_probe_run(
    *,
    run_dir: Path,
    output_dir: Path,
    results: dict[str, Any],
    project: str,
    name: str,
    tags: list[str],
    cache_root: Path,
) -> dict[str, Any]:
    _prepare_wandb_storage(run_dir, cache_root=cache_root)
    _make_wandb_dir(run_dir, "wandb_posthoc")
    wandb = init_wandb_run(
        enabled=True,
        project=project,
        name=name,
        tags=resolve_volume_probe_tags(tags),
        config=(results or {}).get("config", {}),
        missing_message="[wandb] ERROR: volume-probe logging disabled; wandb is not installed",
        show_url=True,
    )
    if wandb is None:
        raise RuntimeError("failed to initialize wandb run")

    run_url = wandb.run.url if getattr(wandb, "run", None) else None
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
    finally:
        finish_wandb_run(wandb)

    return {
        "source_run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "wandb_name": name,
        "wandb_url": run_url,
    }


def _upload_sweep_report(
    *,
    run_root: Path,
    project: str,
    uploads: list[dict[str, Any]],
    cache_root: Path,
) -> dict[str, Any] | None:
    report_paths = [
        run_root / "volume_probe_sweep_report.md",
        run_root / "volume_probe_sweep_report.json",
        run_root / "submitted_jobs.txt",
    ]
    existing = [path for path in report_paths if path.exists()]
    if not existing:
        return None

    _prepare_wandb_storage(run_root, cache_root=cache_root)
    _make_wandb_dir(run_root, "wandb_posthoc_report")
    name = f"posthoc_{run_root.name}_report"
    wandb = init_wandb_run(
        enabled=True,
        project=project,
        name=name,
        tags=["posthoc-upload", "volume-probe", "sweep-report"],
        config={"run_root": str(run_root), "uploaded_runs": len(uploads)},
        show_url=True,
    )
    if wandb is None:
        raise RuntimeError("failed to initialize report wandb run")

    run_url = wandb.run.url if getattr(wandb, "run", None) else None
    try:
        wandb.log({"posthoc/uploaded_runs": int(len(uploads))}, step=0)
        if hasattr(wandb, "Table") and uploads:
            table = wandb.Table(columns=["source_run_dir", "wandb_name", "wandb_url"])
            for item in uploads:
                table.add_data(item["source_run_dir"], item["wandb_name"], item["wandb_url"])
            wandb.log({"posthoc/uploaded_runs_table": table}, step=0)

        artifact = wandb.Artifact(f"{run_root.name}_volume_probe_sweep", type="volume_probe_sweep")
        for path in existing:
            artifact.add_file(str(path))
        wandb.log_artifact(artifact)
    finally:
        finish_wandb_run(wandb)

    return {
        "source_run_dir": str(run_root),
        "wandb_name": name,
        "wandb_url": run_url,
        "artifact_files": [str(path) for path in existing],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload completed volume-probe sweep outputs to W&B.")
    parser.add_argument("--sweep-dir", dest="sweep_dirs", action="append", required=True, type=Path)
    parser.add_argument("--run-root", type=Path, default=None, help="Optional sweep root for report artifact upload.")
    parser.add_argument("--project", default="stratified-manifold-learning")
    parser.add_argument("--manifest", type=Path, default=None, help="Write uploaded run metadata to this JSON file.")
    parser.add_argument("--skip-run", dest="skip_runs", action="append", default=[], help="Run directory names to skip.")
    parser.add_argument(
        "--wandb-cache-root",
        type=Path,
        default=Path(f"/scratch/{os.environ.get('USER', 'unknown')}/.cache/wandb"),
        help="Scratch-backed W&B cache/artifact/temp root.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    uploads: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    skip_runs = set(args.skip_runs)
    for sweep_dir in args.sweep_dirs:
        completed = _iter_completed_runs(sweep_dir)
        if not completed:
            print(f"[skip] no completed volume-probe runs found under {sweep_dir}")
            continue
        print(f"[scan] {sweep_dir}: {len(completed)} completed runs")
        for run_dir, output_dir, results in completed:
            if run_dir.name in skip_runs:
                print(f"[skip] {run_dir.name}")
                continue
            name = f"posthoc_{run_dir.name}"
            tags = ["posthoc-upload", sweep_dir.name]
            if args.dry_run:
                print(f"[dry-run] would upload {run_dir} -> {name}")
                continue
            try:
                uploaded = _upload_volume_probe_run(
                    run_dir=run_dir,
                    output_dir=output_dir,
                    results=results,
                    project=args.project,
                    name=name,
                    tags=tags,
                    cache_root=args.wandb_cache_root,
                )
            except Exception as exc:
                failures.append({"source_run_dir": str(run_dir), "error": str(exc)})
                print(f"[error] {run_dir.name}: {exc}")
                continue
            uploads.append(uploaded)
            print(f"[uploaded] {run_dir.name} -> {uploaded.get('wandb_url') or 'url-unavailable'}")

    report_upload = None
    if args.run_root is not None:
        if args.dry_run:
            print(f"[dry-run] would upload sweep report from {args.run_root}")
        else:
            try:
                report_upload = _upload_sweep_report(
                    run_root=args.run_root,
                    project=args.project,
                    uploads=uploads,
                    cache_root=args.wandb_cache_root,
                )
            except Exception as exc:
                failures.append({"source_run_dir": str(args.run_root), "error": str(exc)})
                print(f"[error] sweep-report: {exc}")
            else:
                if report_upload is not None:
                    print(f"[uploaded] sweep-report -> {report_upload.get('wandb_url') or 'url-unavailable'}")

    manifest = {
        "project": args.project,
        "uploaded_runs": uploads,
        "report_upload": report_upload,
        "failures": failures,
    }
    manifest_path = args.manifest or (args.run_root / "posthoc_wandb_uploads.json" if args.run_root is not None else None)
    if manifest_path is not None and not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2)
        print(f"[saved] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
