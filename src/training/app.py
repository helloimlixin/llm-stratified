from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.config import FiberConfig, make_sam_fiber_config, make_volume_probe_config
from training.runner import run_training
from training.wandb_utils import resolve_wandb_name
from utils import resolve_ddp_settings


@dataclass(frozen=True)
class RunPaths:
    output_dir: Path
    checkpoints_dir: Path
    embeddings_dir: Path
    analysis_dir: Path
    volume_probe_dir: Path


def create_run_paths(
    paths_cfg: Any,
    output_dir: Path,
    *,
    fiber_enabled: bool,
    sam_fiber_enabled: bool = False,
) -> RunPaths:
    paths = RunPaths(
        output_dir=output_dir,
        checkpoints_dir=output_dir / paths_cfg.checkpoints,
        embeddings_dir=output_dir / paths_cfg.embeddings,
        analysis_dir=output_dir / paths_cfg.analysis,
        volume_probe_dir=output_dir / paths_cfg.volume_probe,
    )
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if fiber_enabled or sam_fiber_enabled:
        paths.embeddings_dir.mkdir(parents=True, exist_ok=True)
        paths.analysis_dir.mkdir(parents=True, exist_ok=True)
    return paths


def _is_primary_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def run_sam_fiber_job_if_enabled(cfg: Any, paths: RunPaths) -> bool:
    sam_fiber_config = make_sam_fiber_config(cfg.sam_fiber)
    if not sam_fiber_config.enabled:
        return False
    if not _is_primary_process():
        return True

    from training.sam_fiber_job import run_sam_fiber_job

    results = run_sam_fiber_job(
        dataset_name=cfg.data.name,
        root=cfg.data.root,
        img_size=cfg.data.img_size,
        batch_size_test=cfg.data.batch_size_test,
        num_workers=cfg.data.num_workers,
        subset_test=cfg.data.subset_test,
        seed=cfg.seed,
        output_dir=paths.output_dir,
        checkpoints_dir=paths.checkpoints_dir,
        embeddings_dir=paths.embeddings_dir,
        analysis_dir=paths.analysis_dir,
        sam_cfg=sam_fiber_config,
        wandb_enabled=cfg.wandb.enabled,
        wandb_project=cfg.wandb.project,
        wandb_name=resolve_wandb_name(cfg, suffix="sam_fiber"),
        wandb_tags=getattr(cfg.wandb, "tags", None),
    )
    print(f"\nSAM fiber probe complete! Results saved to: {paths.output_dir}")
    summary_path = paths.output_dir / "sam_fiber_summary.json"
    if summary_path.exists():
        print(f"SAM fiber summary: {summary_path}")
    if results:
        print(f"Collected tokens: {results.get('collection', {}).get('num_tokens', 'n/a')}")
    return True


def run_volume_probe_job_if_enabled(cfg: Any, paths: RunPaths) -> bool:
    volume_probe_config = make_volume_probe_config(cfg.volume_probe)
    if not volume_probe_config.enabled:
        return False
    if not _is_primary_process():
        return True

    from training.volume_probe_job import run_volume_probe_job
    from training.volume_probe_logging import log_volume_probe_to_wandb

    paths.volume_probe_dir.mkdir(parents=True, exist_ok=True)
    results = run_volume_probe_job(
        dataset_name=cfg.data.name,
        root=cfg.data.root,
        img_size=cfg.data.img_size,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout_rate=cfg.model.dropout,
        batch_size_train=cfg.data.batch_size,
        batch_size_test=cfg.data.batch_size_test,
        num_workers=cfg.data.num_workers,
        subset_train=cfg.data.subset_train,
        subset_test=cfg.data.subset_test,
        timm_model=cfg.model.timm_model,
        timm_pretrained=cfg.model.timm_pretrained,
        seed=cfg.seed,
        output_dir=paths.volume_probe_dir,
        volume_cfg=volume_probe_config,
    )
    log_volume_probe_to_wandb(
        enabled=cfg.wandb.enabled,
        project=cfg.wandb.project,
        name=resolve_wandb_name(cfg, suffix="volume_probe"),
        tags=getattr(cfg.wandb, "tags", None),
        results=results,
        output_dir=paths.volume_probe_dir,
    )
    print(f"\nVolume probe complete! Results saved to: {paths.volume_probe_dir}")
    return True


def run_training_job_from_config(cfg: Any, paths: RunPaths, fiber_config: FiberConfig) -> None:
    use_ddp, local_rank, world_size = resolve_ddp_settings(
        use_ddp=cfg.compute.use_ddp,
        local_rank=getattr(cfg.compute, "local_rank", 0),
        use_accelerate=cfg.compute.use_accelerate,
        require_multi_gpu=False,
    )

    run_training(
        dataset_name=cfg.data.name,
        root=cfg.data.root,
        num_runs=cfg.runs,
        num_epochs=cfg.training.epochs,
        save_interval=cfg.training.save_interval,
        lr=cfg.training.lr,
        wd=cfg.training.weight_decay,
        grad_clip=cfg.training.grad_clip,
        progress_log_interval=cfg.training.progress_log_interval,
        base_dir=str(paths.checkpoints_dir),
        seed_base=cfg.seed,
        num_workers=cfg.data.num_workers,
        img_size=cfg.data.img_size,
        patch_size=cfg.model.patch_size,
        embed_dim=cfg.model.embed_dim,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout_rate=cfg.model.dropout,
        label_smoothing=cfg.training.label_smoothing,
        warmup_epochs=cfg.training.warmup_epochs,
        cosine=cfg.training.cosine_schedule,
        compile_model=cfg.model.compile,
        wandb_on=cfg.wandb.enabled,
        wandb_project=cfg.wandb.project,
        wandb_runname=resolve_wandb_name(cfg),
        batch_size_train=cfg.data.batch_size,
        batch_size_test=cfg.data.batch_size_test,
        subset_train=cfg.data.subset_train,
        subset_test=cfg.data.subset_test,
        timm_model=cfg.model.timm_model,
        timm_pretrained=cfg.model.timm_pretrained,
        frozen_backbone=getattr(cfg.model, "frozen_backbone", None),
        frozen_backbone_model=getattr(cfg.model, "frozen_backbone_model", None),
        fiber_cfg=fiber_config,
        use_ddp=use_ddp,
        local_rank=local_rank,
        world_size=world_size,
        use_accelerate=cfg.compute.use_accelerate,
    )
    if int(cfg.training.epochs) <= 0:
        print(f"\nAnalysis-only run complete! Results saved to: {paths.output_dir}")
    else:
        print(f"\nTraining complete! Results saved to: {paths.output_dir}")
