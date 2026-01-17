"""Shared utilities for TinyViT and fiber bundle analysis."""

import os
import random

import numpy as np
import torch

from data import get_norm_stats


def seed_everything(seed: int = 1337) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the default compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denormalize_images(imgs: torch.Tensor, dataset: str) -> torch.Tensor:
    """Denormalize images back to [0, 1] range.

    Args:
        imgs: Normalized image tensor [B, C, H, W]
        dataset: Dataset name for selecting normalization stats

    Returns:
        Denormalized images clamped to [0, 1]
    """
    mean, std = get_norm_stats(dataset, imgs.device, as_tensor=True)
    return (imgs * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)).clamp(0, 1)


def to_serializable(obj):
    """Convert numpy/torch types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    return obj


def resolve_ddp_settings(
    *,
    use_ddp: bool,
    local_rank: int = 0,
    use_accelerate: bool = False,
    require_multi_gpu: bool = False,
    announce: bool = True,
) -> tuple[bool, int, int]:
    """Resolve DDP settings from CLI/config inputs and torchrun environment."""
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    auto_ddp = False
    if not use_ddp and not use_accelerate and "RANK" in os.environ:
        if not require_multi_gpu or num_gpus > 1:
            use_ddp = True
            auto_ddp = True
    local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
    try:
        world_size = int(os.environ.get("WORLD_SIZE", num_gpus if use_ddp else 1))
    except ValueError:
        world_size = num_gpus if use_ddp else 1
    if use_ddp and require_multi_gpu and world_size <= 1:
        use_ddp = False
        world_size = 1
    if auto_ddp and announce:
        if num_gpus:
            print(f"[Auto] Detected {num_gpus} GPUs + torchrun, enabling DDP")
        else:
            print("[Auto] Detected torchrun, enabling DDP")
    return use_ddp, local_rank, world_size
