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

import os
from pathlib import Path
from typing import Any

import torch

from training.configs import build_fiber_config, build_volume_probe_config
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


def _run_hydra(cfg: Any) -> None:
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    from utils import resolve_ddp_settings

    # Print resolved config
    print("=" * 60)
    print("Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Get output directory from Hydra
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    print(f"Output directory: {output_dir}")

    # Create subdirectories
    checkpoints_dir = output_dir / cfg.paths.checkpoints
    embeddings_dir = output_dir / cfg.paths.embeddings
    analysis_dir = output_dir / cfg.paths.analysis
    volume_probe_dir = output_dir / cfg.paths.volume_probe

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if cfg.fiber.enabled:
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        analysis_dir.mkdir(parents=True, exist_ok=True)

    # Convert fiber/volume configs
    fiber_cfg = build_fiber_config(cfg.fiber, patch_size=cfg.model.patch_size)
    volume_probe_cfg = build_volume_probe_config(cfg.volume_probe)

    # Volume probe mode (no training)
    if volume_probe_cfg.enabled:
        from training.volume_probe_job import run_volume_probe_job

        if int(os.environ.get("RANK", "0")) != 0:
            return
        volume_probe_dir.mkdir(parents=True, exist_ok=True)
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
            output_dir=volume_probe_dir,
            volume_cfg=volume_probe_cfg,
        )
        if cfg.wandb.enabled:
            try:
                import numpy as np
                import wandb  # type: ignore
            except Exception as exc:
                print(f"[wandb] ERROR: volume-probe logging disabled ({exc})")
            else:
                wandb_name = cfg.wandb.name or f"{cfg.data.name}_{cfg.model.name}_volume_probe"
                tags = []
                try:
                    tags = list(cfg.wandb.tags) if cfg.wandb.tags is not None else []
                except Exception:
                    tags = []
                if "volume-probe" not in tags:
                    tags.append("volume-probe")
                try:
                    if os.environ.get("WANDB_MODE", "online") == "online":
                        try:
                            if wandb.api.api_key is None:
                                print("[wandb] WARNING: Not logged in, using offline mode")
                                os.environ["WANDB_MODE"] = "offline"
                        except Exception:
                            pass
                    wandb.init(project=cfg.wandb.project, name=wandb_name, tags=tags, config=results.get("config", {}))

                    log_payload: dict[str, object] = {}
                    reps = (results or {}).get("representations", {}) or {}
                    for rep_name, rep in reps.items():
                        summary = (rep or {}).get("summary", {}) or {}
                        # Keep logging focused on "volume = neighbor counts" for now.
                        num_points = summary.get("num_tokens", None)
                        if num_points is not None:
                            log_payload[f"volume_probe/{rep_name}/num_points"] = num_points

                        knn = (rep or {}).get("knn_curve")
                        if isinstance(knn, dict) and knn.get("k_values") and knn.get("radii"):
                            try:
                                ks = [int(k) for k in list(knn.get("k_values"))]
                                radii = dict(knn.get("radii") or {})
                                q50 = radii.get("q50")
                                if isinstance(q50, list) and ks and q50 and len(q50) == len(ks):
                                    k_min = int(knn.get("k_min", ks[0]))
                                    k_max = int(knn.get("k_max", ks[-1]))
                                    log_payload[f"volume_probe/{rep_name}/k_min"] = k_min
                                    log_payload[f"volume_probe/{rep_name}/k_max"] = k_max
                            except Exception as e:
                                print(f"[wandb] kNN curve log error ({rep_name}): {e}")

                    # Log visualizations produced by the volume probe (images + patches + NN grids).
                    viz = (results or {}).get("viz", {}) or {}
                    if isinstance(viz, dict):
                        for key, fn in viz.items():
                            if not isinstance(fn, str) or not fn:
                                continue
                            p = volume_probe_dir / fn
                            if p.exists():
                                log_payload[f"volume_probe/viz/{key}"] = wandb.Image(str(p))

                    # Log "static" payload at step 0 (counts + images).
                    if log_payload:
                        wandb.log(log_payload, step=0)

                    # Log per-k radius curves as a real step series (step = k).
                    try:
                        curve_reps = {}
                        for rep_name, rep in reps.items():
                            knn = (rep or {}).get("knn_curve")
                            if not (isinstance(knn, dict) and knn.get("k_values") and knn.get("radii")):
                                continue
                            ks = [int(k) for k in list(knn.get("k_values"))]
                            radii = dict(knn.get("radii") or {})
                            q50 = radii.get("q50")
                            if isinstance(q50, list) and ks and q50 and len(q50) == len(ks):
                                curve_reps[rep_name] = (ks, [float(x) for x in q50])

                        # Prefer using the tokens k-range if present, otherwise the first available rep.
                        base_rep = "tokens" if "tokens" in curve_reps else (next(iter(curve_reps.keys())) if curve_reps else None)
                        if base_rep is not None:
                            base_ks = curve_reps[base_rep][0]
                            idx_maps = {name: {k: i for i, k in enumerate(ks)} for name, (ks, _vals) in curve_reps.items()}
                            for k in base_ks:
                                row = {"volume_probe/k": int(k)}
                                for rep_name, (_ks, vals) in curve_reps.items():
                                    j = idx_maps[rep_name].get(int(k))
                                    if j is not None:
                                        row[f"volume_probe/{rep_name}/radius_q50"] = float(vals[j])
                                wandb.log(row, step=int(k))
                    except Exception as e:
                        print(f"[wandb] curve series log error: {e}")

                    # Upload the on-disk outputs as an artifact for reproducibility.
                    try:
                        art = wandb.Artifact(f"{wandb.run.name}_volume_probe", type="volume_probe")
                        summary_path = volume_probe_dir / "volume_summary.json"
                        if summary_path.exists():
                            art.add_file(str(summary_path))
                        viz = (results or {}).get("viz", {}) or {}
                        if isinstance(viz, dict):
                            for fn in viz.values():
                                if isinstance(fn, str) and fn:
                                    p = volume_probe_dir / fn
                                    if p.exists():
                                        art.add_file(str(p))
                        for rep in reps.values():
                            for key in ("dims_path", "results_path"):
                                fn = (rep or {}).get(key)
                                if isinstance(fn, str) and fn:
                                    p = volume_probe_dir / fn
                                    if p.exists():
                                        art.add_file(str(p))
                        wandb.log_artifact(art)
                    except Exception as e:
                        print(f"[wandb] artifact log error: {e}")
                except Exception as e:
                    print(f"[wandb] ERROR: {e}")
                finally:
                    try:
                        wandb.finish()
                    except Exception:
                        pass
        print(f"\nVolume probe complete! Results saved to: {volume_probe_dir}")
        return

    # Handle wandb name
    wandb_name = cfg.wandb.name
    if wandb_name is None and cfg.wandb.enabled:
        wandb_name = f"{cfg.data.name}_{cfg.model.name}"

    # Detect multi-GPU from environment
    use_ddp, local_rank, world_size = resolve_ddp_settings(
        use_ddp=cfg.compute.use_ddp,
        local_rank=getattr(cfg.compute, "local_rank", 0),
        use_accelerate=cfg.compute.use_accelerate,
        require_multi_gpu=False,
    )

    # Run training
    run_training(
        dataset_name=cfg.data.name,
        root=cfg.data.root,
        num_runs=cfg.runs,
        num_epochs=cfg.training.epochs,
        save_interval=cfg.training.save_interval,
        lr=cfg.training.lr,
        wd=cfg.training.weight_decay,
        grad_clip=cfg.training.grad_clip,
        base_dir=str(checkpoints_dir),
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
        wandb_runname=wandb_name,
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

    print(f"\nTraining complete! Results saved to: {output_dir}")


def _run_hydra_main() -> None:
    try:
        import hydra
        from omegaconf import DictConfig
    except Exception as exc:
        raise RuntimeError("hydra is required for src/train.py; install hydra-core") from exc

    @hydra.main(version_base=None, config_path="../configs", config_name="config")
    def _main(cfg: DictConfig) -> None:
        _run_hydra(cfg)

    _main()


def main() -> None:
    _run_hydra_main()


if __name__ == "__main__":
    main()

