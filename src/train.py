#!/usr/bin/env python3
"""
TinyViT Training with Hydra Configuration

This file is intentionally kept small. The implementation lives in `src/training/*`.

Examples:
    # Basic training
    python src/train.py

    # Different dataset
    python src/train.py data=stl10

    # Enable fiber analysis
    python src/train.py fiber=basic

    # Full polysemy study
    python src/train.py +experiment=polysemy_study

    # Quick test run
    python src/train.py +experiment=quick_test

    # No-training volume probe (tokens vs patch embeddings vs raw patches)
    python src/train.py +experiment=volume_probe

    # Override specific parameters
    python src/train.py training.epochs=100 training.lr=1e-3

    # Multi-GPU with DDP
    torchrun --nproc_per_node=2 src/train.py compute=ddp

    # Hyperparameter sweep
    python src/train.py --multirun training.lr=1e-3,3e-4,1e-4 data=cifar10,stl10
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from training.config import make_fiber_config
from training.loops import (
    evaluate,
    evaluate_accelerate,
    get_criterion,
    multilabel_accuracy,
    train_one_epoch,
    train_one_epoch_accelerate,
)

__all__ = [
    "run_training",
    "gather_ddp_tensor",
    "get_criterion",
    "multilabel_accuracy",
    "train_one_epoch",
    "evaluate",
    "train_one_epoch_accelerate",
    "evaluate_accelerate",
    "main",
]


# ---------------------------------------------------------------------------
# Backwards-compatible helpers (imported by tests and external callers)
# ---------------------------------------------------------------------------


def gather_ddp_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    from training.backend import gather_ddp_tensor as _gather

    return _gather(tensor, world_size)


def run_training(*args, **kwargs) -> None:
    from training.runner import run_training as _run_training

    return _run_training(*args, **kwargs)


# ---------------------------------------------------------------------------
# Hydra entrypoint
# ---------------------------------------------------------------------------


def run_from_hydra_config(cfg: Any) -> None:
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    from training.app import (
        create_run_paths,
        run_sam_fiber_job_if_enabled,
        run_training_job_from_config,
        run_volume_probe_job_if_enabled,
    )
    from training.wandb_utils import ensure_wandb_dir

    # Print resolved config
    print("=" * 60)
    print("Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Get output directory from Hydra
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    print(f"Output directory: {output_dir}")
    ensure_wandb_dir(enabled=cfg.wandb.enabled, output_dir=output_dir)

    fiber_config = make_fiber_config(cfg.fiber, patch_size=cfg.model.patch_size)
    paths = create_run_paths(
        cfg.paths,
        output_dir,
        fiber_enabled=fiber_config.enabled,
        sam_fiber_enabled=bool(getattr(cfg.sam_fiber, "enabled", False)),
    )

    if run_volume_probe_job_if_enabled(cfg, paths):
        return
    if run_sam_fiber_job_if_enabled(cfg, paths):
        return

    run_training_job_from_config(cfg, paths, fiber_config)


def run_hydra_cli() -> None:
    try:
        import hydra
        from omegaconf import DictConfig
    except Exception as exc:
        raise RuntimeError("hydra is required for src/train.py; install hydra-core") from exc

    @hydra.main(version_base=None, config_path="../configs", config_name="config")
    def _main(cfg: DictConfig) -> None:
        run_from_hydra_config(cfg)

    _main()


def main() -> None:
    run_hydra_cli()


if __name__ == "__main__":
    main()
