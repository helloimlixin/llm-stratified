from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.configs import build_volume_probe_config
from training.runner import run_training
from utils import resolve_ddp_settings


@dataclass(frozen=True)
class OutputPaths:
    output_dir: Path
    checkpoints_dir: Path
    embeddings_dir: Path
    analysis_dir: Path
    volume_probe_dir: Path


def prepare_output_paths(paths_cfg: Any, output_dir: Path, *, fiber_enabled: bool) -> OutputPaths:
    paths = OutputPaths(
        output_dir=output_dir,
        checkpoints_dir=output_dir / paths_cfg.checkpoints,
        embeddings_dir=output_dir / paths_cfg.embeddings,
        analysis_dir=output_dir / paths_cfg.analysis,
        volume_probe_dir=output_dir / paths_cfg.volume_probe,
    )
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if fiber_enabled:
        paths.embeddings_dir.mkdir(parents=True, exist_ok=True)
        paths.analysis_dir.mkdir(parents=True, exist_ok=True)
    return paths


def resolve_wandb_name(cfg: Any, *, suffix: str = "") -> str:
    if cfg.wandb.name:
        return str(cfg.wandb.name)
    base_name = f"{cfg.data.name}_{cfg.model.name}"
    return f"{base_name}_{suffix}" if suffix else base_name


def maybe_run_volume_probe(cfg: Any, paths: OutputPaths) -> bool:
    volume_probe_cfg = build_volume_probe_config(cfg.volume_probe)
    if not volume_probe_cfg.enabled:
        return False

    from training.volume_probe_job import run_volume_probe_job
    from training.volume_probe_logging import log_volume_probe_to_wandb

    if int(os.environ.get("RANK", "0")) != 0:
        return True

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
        volume_cfg=volume_probe_cfg,
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


def run_training_from_cfg(cfg: Any, paths: OutputPaths, fiber_cfg: Any) -> None:
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
        fiber_cfg=fiber_cfg,
        use_ddp=use_ddp,
        local_rank=local_rank,
        world_size=world_size,
        use_accelerate=cfg.compute.use_accelerate,
    )
    print(f"\nTraining complete! Results saved to: {paths.output_dir}")
