from __future__ import annotations

import atexit
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn

from datasets import create_data_loaders, get_class_names
from models import resolve_patch_size
from utils import seed_everything

from training.backend import gather_ddp_tensor, init_backend
from training.config import FiberConfig
from training.loops import (
    evaluate,
    evaluate_accelerate,
    train_one_epoch,
    train_one_epoch_accelerate,
)
from training.model_factory import build_classifier_model
from training.wandb_utils import finish_wandb_run, init_wandb_run


def _make_grad_scaler(device: torch.device):
    if device.type != "cuda":
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            return torch.amp.GradScaler(enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def run_training(
    dataset_name: str = "CIFAR10",
    root: str = "./data",
    num_runs: int = 1,
    num_epochs: int = 10,
    save_interval: int = 2,
    lr: float = 3e-4,
    wd: float = 0.05,
    grad_clip: Optional[float] = 1.0,
    progress_log_interval: int = 0,
    base_dir: str = "./runs/tinyvit",
    seed_base: int = 1337,
    num_workers: int = 4,
    img_size: Optional[int] = None,
    patch_size: int = 4,
    embed_dim: int = 192,
    depth: int = 8,
    num_heads: int = 3,
    mlp_ratio: float = 2.0,
    dropout_rate: float = 0.1,
    label_smoothing: float = 0.0,
    warmup_epochs: Optional[int] = None,
    cosine: bool = True,
    compile_model: bool = False,
    wandb_on: bool = False,
    wandb_project: str = "tinyvit",
    wandb_runname: Optional[str] = None,
    batch_size_train: int = 128,
    batch_size_test: int = 256,
    subset_train: Optional[int] = None,
    subset_test: Optional[int] = None,
    timm_model: Optional[str] = None,
    timm_pretrained: bool = True,
    frozen_backbone: Optional[str] = None,
    frozen_backbone_model: Optional[str] = None,
    fiber_cfg: Optional[FiberConfig] = None,
    use_ddp: bool = False,
    local_rank: int = 0,
    world_size: int = 1,
    use_accelerate: bool = False,
) -> None:
    """Unified training driver supporting single-GPU, DDP, and Accelerate backends."""
    fiber_cfg = fiber_cfg or FiberConfig()
    wandb = None
    if timm_model and img_size is None:
        img_size = 224

    backend = init_backend(use_ddp=use_ddp, use_accelerate=use_accelerate, local_rank=local_rank)
    device = backend.device
    accelerator = backend.accelerator
    is_main_process = backend.is_main_process
    rank = backend.rank
    world_size = backend.world_size
    local_rank = backend.local_rank

    if use_ddp and not use_accelerate:

        def _cleanup() -> None:
            try:
                if is_main_process:
                    finish_wandb_run(wandb)
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:
                pass

        atexit.register(_cleanup)

    if is_main_process:
        os.makedirs(base_dir, exist_ok=True)
    cudnn.benchmark = device.type == "cuda"
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # Adjust batch sizes
    eff_bs_train = batch_size_train // world_size if use_ddp else batch_size_train
    eff_bs_test = batch_size_test // world_size if use_ddp else batch_size_test
    if use_accelerate:
        eff_bs_train, eff_bs_test = max(1, batch_size_train // world_size), max(1, batch_size_test // world_size)

    # Data loaders
    train_loader, test_loader, num_classes, in_chans, final_img_size, task = create_data_loaders(
        dataset_name,
        root,
        img_size,
        eff_bs_train,
        eff_bs_test,
        num_workers,
        subset_train,
        subset_test,
        device,
        distributed=use_ddp and not use_accelerate,
        rank=rank,
        world_size=world_size,
    )
    base_train_loader, base_test_loader = train_loader, test_loader
    class_names = get_class_names(test_loader.dataset, dataset_name) if fiber_cfg.enabled else None
    train_sampler = train_loader.sampler if use_ddp and not use_accelerate else None

    # Wandb init
    if wandb_on and is_main_process:
        wandb = init_wandb_run(
            enabled=True,
            project=wandb_project,
            name=wandb_runname,
            config=dict(
                dataset=dataset_name,
                img_size=final_img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                lr=lr,
                wd=wd,
                epochs=num_epochs,
                batch_size=batch_size_train,
                fiber_enabled=fiber_cfg.enabled,
                num_gpus=world_size,
                use_ddp=use_ddp,
                use_accelerate=use_accelerate,
            ),
            show_url=True,
        )
        wandb_on = wandb is not None

    for run_idx in range(num_runs):
        # Run directory setup
        if is_main_process:
            run_dir = (
                Path(base_dir) if fiber_cfg.enabled and num_runs == 1 else Path(base_dir) / f"{dataset_name}_run_{run_idx:03d}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            embed_dir = run_dir / "embeddings" if fiber_cfg.enabled else None
            analysis_dir = run_dir / "fiber_analysis" if fiber_cfg.enabled else None
            if embed_dir:
                embed_dir.mkdir(exist_ok=True)
            if analysis_dir:
                analysis_dir.mkdir(exist_ok=True)
        else:
            run_dir = embed_dir = analysis_dir = None

        seed = seed_base + run_idx + (accelerator.process_index if use_accelerate else 0)
        seed_everything(seed)

        # Model
        model, patch_size_used = build_classifier_model(
            num_classes=num_classes,
            in_chans=in_chans,
            img_size=final_img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
            timm_model=timm_model,
            timm_pretrained=timm_pretrained,
            frozen_backbone=frozen_backbone,
            frozen_backbone_model=frozen_backbone_model,
            announce=is_main_process,
        )

        if frozen_backbone and hasattr(model, "set_dataset_name"):
            model.set_dataset_name(dataset_name)

        if not use_accelerate:
            model = model.to(device)
            if use_ddp:
                model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
            model_for_saving = model.module if use_ddp else model
        else:
            model_for_saving = model

        if compile_model and hasattr(torch, "compile"):
            model = torch.compile(model)

        # Optimizer & scheduler
        effective_lr = lr * world_size if (use_ddp or use_accelerate) else lr
        optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=wd)
        warmup = warmup_epochs if warmup_epochs else max(1, min(5, int(0.1 * num_epochs)))

        def lr_lambda(e: int) -> float:
            if e < warmup:
                return (e + 1) / max(1, warmup)
            if cosine:
                t = (e - warmup) / max(1, num_epochs - warmup)
                return 0.5 * (1.0 + math.cos(math.pi * t))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        scaler = _make_grad_scaler(device) if not use_accelerate else None

        if use_accelerate:
            train_loader, test_loader = base_train_loader, base_test_loader
            model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
                model, optimizer, train_loader, test_loader, scheduler
            )

        train_history, fiber_history = [], []
        embedding_animation_snapshots: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        final_dims = final_coords_3d = final_tsne_3d = None

        if is_main_process:
            print(f"\nStarting {dataset_name} run {run_idx + 1}/{num_runs} -> {run_dir}")

        for epoch in range(num_epochs):
            # Training
            if use_accelerate:
                train_loss, train_acc = train_one_epoch_accelerate(
                    model, train_loader, optimizer, accelerator, task, grad_clip, label_smoothing, epoch
                )
                eval_loss, eval_acc = evaluate_accelerate(model, test_loader, accelerator, task, label_smoothing)
            else:
                train_loss, train_acc = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scaler,
                    device,
                    task,
                    grad_clip,
                    label_smoothing,
                    epoch,
                    train_sampler,
                    log_interval=progress_log_interval,
                    dataset_name=dataset_name,
                )
                eval_loss, eval_acc = evaluate(
                    model,
                    test_loader,
                    device,
                    task,
                    label_smoothing,
                    epoch=epoch,
                    log_interval=progress_log_interval,
                    dataset_name=dataset_name,
                )
            scheduler.step()

            # Sync metrics for DDP
            if use_ddp and not use_accelerate:
                metrics = torch.tensor([train_loss, train_acc, eval_loss, eval_acc], device=device)
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                metrics /= world_size
                train_loss, train_acc, eval_loss, eval_acc = metrics.cpu().tolist()

            lr_now = scheduler.get_last_lr()[0]
            log_row = {
                "epoch": epoch,
                "lr": lr_now,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
            }

            if is_main_process:
                print(
                    f"[{dataset_name}] Epoch {epoch:03d} | lr {lr_now:.2e} | train {train_loss:.4f}/{train_acc:.4f} | "
                    f"val {eval_loss:.4f}/{eval_acc:.4f}"
                )
                if fiber_cfg.enabled:
                    train_history.append(log_row)
                if wandb_on and wandb:
                    try:
                        wandb.log(
                            {
                                "epoch": epoch,
                                "lr": lr_now,
                                "train/loss": train_loss,
                                "train/acc": train_acc,
                                "val/loss": eval_loss,
                                "val/acc": eval_acc,
                            }
                        )
                    except Exception as e:
                        print(f"[wandb] log error: {e}")

            # Fiber analysis
            if fiber_cfg.enabled and (epoch == 0 or (epoch + 1) % fiber_cfg.embed_interval == 0 or epoch == num_epochs - 1):
                from fiber.analysis import analyze_fiber_epoch
                from fiber.patch_tokens import collect_patch_tokens

                if use_ddp and not use_accelerate and dist.is_initialized():
                    dist.barrier()
                if use_accelerate:
                    accelerator.wait_for_everyone()

                if is_main_process:
                    print(f"[fiber] Epoch {epoch:03d}: collecting patch tokens...", flush=True)
                    t0 = time.time()

                local_max = None if fiber_cfg.embed_full_val else max(1, math.ceil(fiber_cfg.max_tokens / world_size))
                model_unwrapped = (
                    accelerator.unwrap_model(model) if use_accelerate else (model.module if use_ddp else model)
                )
                patch_sz_eff = resolve_patch_size(model_unwrapped) or patch_size_used
                embeddings, labels, images, bboxes, patch_indices, img_ids, pred_labels = collect_patch_tokens(
                    model_unwrapped, test_loader, device, patch_sz_eff, local_max, show_progress=is_main_process
                )

                if is_main_process:
                    print(f"[fiber] Epoch {epoch:03d}: collected in {time.time() - t0:.1f}s", flush=True)

                # Gather across ranks
                if use_accelerate:
                    embeddings = accelerator.gather_for_metrics(embeddings.to(device))
                    labels = accelerator.gather_for_metrics(labels.to(device))
                    images = accelerator.gather_for_metrics(images.to(device))
                    bboxes = accelerator.gather_for_metrics(bboxes.to(device))
                    patch_indices = accelerator.gather_for_metrics(patch_indices.to(device))
                    img_ids = accelerator.gather_for_metrics(img_ids.to(device))
                    pred_labels = accelerator.gather_for_metrics(pred_labels.to(device))
                elif use_ddp:
                    embeddings = gather_ddp_tensor(embeddings.to(device), world_size)
                    labels = gather_ddp_tensor(labels.to(device), world_size)
                    images = gather_ddp_tensor(images.to(device), world_size)
                    bboxes = gather_ddp_tensor(bboxes.to(device), world_size)
                    patch_indices = gather_ddp_tensor(patch_indices.to(device), world_size)
                    img_ids = gather_ddp_tensor(img_ids.to(device), world_size)
                    pred_labels = gather_ddp_tensor(pred_labels.to(device), world_size)

                if is_main_process and embed_dir and analysis_dir and run_dir:
                    if not fiber_cfg.embed_full_val:
                        embeddings = embeddings[: fiber_cfg.max_tokens]
                        labels = labels[: fiber_cfg.max_tokens]
                        bboxes = bboxes[: fiber_cfg.max_tokens]
                        patch_indices = patch_indices[: fiber_cfg.max_tokens]
                        img_ids = img_ids[: fiber_cfg.max_tokens]
                        pred_labels = pred_labels[: fiber_cfg.max_tokens]
                        # images is a unique-image buffer; don't truncate by max_tokens

                    embeddings = embeddings.cpu()
                    labels = labels.cpu()
                    images = images.cpu()
                    bboxes = bboxes.cpu()
                    patch_indices = patch_indices.cpu()
                    img_ids = img_ids.cpu()
                    pred_labels = pred_labels.cpu()

                    print(f"[fiber] Epoch {epoch:03d}: running analysis...", flush=True)
                    t1 = time.time()
                    analysis = analyze_fiber_epoch(
                        epoch=epoch,
                        embeddings=embeddings,
                        labels=labels,
                        images=images,
                        bboxes=bboxes,
                        patch_indices=patch_indices,
                        image_ids=img_ids,
                        pred_labels=pred_labels,
                        num_classes=num_classes,
                        class_names=class_names,
                        dataset=dataset_name,
                        base_dir=run_dir,
                        analysis_dir=analysis_dir,
                        embed_dir=embed_dir,
                        vol_min=fiber_cfg.vol_min,
                        vol_max=fiber_cfg.vol_max,
                        ws=fiber_cfg.ws,
                        alpha=fiber_cfg.alpha,
                        nstrat=fiber_cfg.nstrat,
                        neighborhood_size=fiber_cfg.neighborhood_size or patch_sz_eff + 1,
                        polysemy=fiber_cfg.polysemy,
                        polysemy_k=fiber_cfg.polysemy_k,
                        polysemy_anchors=fiber_cfg.polysemy_anchors,
                        polysemy_grid_cols=fiber_cfg.polysemy_grid_cols,
                        vit_token_polysemy=fiber_cfg.vit_token_polysemy,
                        vit_token_polysemy_k=fiber_cfg.vit_token_polysemy_k,
                        vit_token_polysemy_topk=fiber_cfg.vit_token_polysemy_topk,
                        vit_token_polysemy_ablate=fiber_cfg.vit_token_polysemy_ablate,
                        vit_token_polysemy_ablate_batches=fiber_cfg.vit_token_polysemy_ablate_batches,
                        vit_token_polysemy_min_count=fiber_cfg.vit_token_polysemy_min_count,
                        vit_token_polysemy_ablate_reps=fiber_cfg.vit_token_polysemy_ablate_reps,
                        sparse_probe=fiber_cfg.sparse_probe,
                        sparse_probe_radius=fiber_cfg.sparse_probe_radius,
                        sparse_probe_auto_neighbor_k=fiber_cfg.sparse_probe_auto_neighbor_k,
                        sparse_probe_auto_radius_quantile=fiber_cfg.sparse_probe_auto_radius_quantile,
                        sparse_probe_min_patches=fiber_cfg.sparse_probe_min_patches,
                        sparse_probe_max_anchors=fiber_cfg.sparse_probe_max_anchors,
                        sparse_probe_dictionary_size=fiber_cfg.sparse_probe_dictionary_size,
                        sparse_probe_residual_threshold=fiber_cfg.sparse_probe_residual_threshold,
                        sparse_probe_max_sparsity=fiber_cfg.sparse_probe_max_sparsity,
                        sparse_probe_global_dictionary=fiber_cfg.sparse_probe_global_dictionary,
                        wandb_module=wandb if wandb_on else None,
                        model=model_unwrapped,
                        device=device,
                        img_size=final_img_size,
                        patch_size=patch_sz_eff,
                        val_loader=test_loader,
                    )
                    print(f"[fiber] Epoch {epoch:03d}: done in {time.time() - t1:.1f}s", flush=True)
                    fiber_history.append(analysis["fiber_summary"])
                    if fiber_cfg.embedding_animation:
                        embedding_animation_snapshots.append(
                            (epoch, embeddings.detach().cpu().clone(), labels.detach().cpu().clone())
                        )
                    final_dims, final_coords_3d, final_tsne_3d = (
                        analysis["final_dims"],
                        analysis["final_coords_3d"],
                        analysis["final_tsne_3d"],
                    )

            # Checkpoint
            if is_main_process and run_dir and (epoch == 0 or epoch % save_interval == 0 or epoch == num_epochs - 1):
                ckpt_model = accelerator.unwrap_model(model) if use_accelerate else model_for_saving
                ckpt_path = run_dir / f"epoch_{epoch:03d}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": ckpt_model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "eval_loss": eval_loss,
                        "eval_acc": eval_acc,
                        "dataset": dataset_name,
                        "task": task,
                        "img_size": final_img_size,
                        "patch_size": patch_size_used,
                        "timestamp": datetime.now().isoformat(),
                        "seed": seed,
                        "num_gpus": world_size,
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint -> {ckpt_path}")

        # Save histories
        if fiber_cfg.enabled and is_main_process and run_dir:
            from fiber.animation import build_embedding_animation_frames, generate_embedding_animation
            from fiber.plots import save_training_summary_plot

            with open(run_dir / "train_history.json", "w") as fp:
                json.dump(train_history, fp, indent=2)
            with open(run_dir / "fiber_history.json", "w") as fp:
                json.dump(fiber_history, fp, indent=2)
            if final_coords_3d is not None and final_dims is not None:
                save_training_summary_plot(
                    train_history,
                    fiber_history,
                    final_coords_3d,
                    final_dims,
                    run_dir / "fiber_bundle_summary.png",
                )
                print(f"Saved summary plot -> {run_dir / 'fiber_bundle_summary.png'}")
            if fiber_cfg.embedding_animation and embedding_animation_snapshots:
                try:
                    animation_frames = build_embedding_animation_frames(embedding_animation_snapshots)
                    if not animation_frames:
                        print("[embedding_animation] skipped: no non-empty frames")
                    else:
                        animation_path = generate_embedding_animation(
                            animation_frames,
                            title=f"{dataset_name} Embedding Space",
                            output_path=run_dir / "embedding_progression.gif",
                            fps=fiber_cfg.embedding_animation_fps,
                        )
                        print(f"Saved embedding animation -> {animation_path}")
                        if wandb_on and wandb and hasattr(wandb, "Video"):
                            wandb.log(
                                {
                                    "embeddings/progression": wandb.Video(str(animation_path), format="gif"),
                                    "embeddings/progression_frames": len(animation_frames),
                                }
                            )
                except Exception as e:
                    print(f"[embedding_animation] failed: {e}")

    # Cleanup
    if wandb_on and is_main_process:
        finish_wandb_run(wandb)
    if use_ddp and not use_accelerate and dist.is_initialized():
        dist.destroy_process_group()
