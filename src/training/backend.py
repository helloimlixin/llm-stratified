from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

try:
    from accelerate import Accelerator
except ImportError:  # pragma: no cover
    Accelerator = None


@dataclass
class BackendState:
    device: torch.device
    is_main_process: bool
    rank: int
    world_size: int
    accelerator: Optional[object]
    local_rank: int


def init_backend(*, use_ddp: bool, use_accelerate: bool, local_rank: int) -> BackendState:
    """Initialize DDP or Accelerate backend state (or fall back to single-process)."""
    if use_accelerate:
        if Accelerator is None:
            raise RuntimeError("accelerate required; pip install accelerate")
        accelerator = Accelerator()
        return BackendState(
            device=accelerator.device,
            is_main_process=accelerator.is_main_process,
            rank=getattr(accelerator, "process_index", 0),
            world_size=getattr(accelerator, "num_processes", 1),
            accelerator=accelerator,
            local_rank=getattr(accelerator, "local_process_index", 0),
        )

    if use_ddp:
        if not dist.is_initialized():
            if "RANK" not in os.environ:
                raise RuntimeError(
                    "DDP requires torchrun. Launch with: torchrun --nproc_per_node=<gpus> src/train.py compute=ddp"
                )
            dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return BackendState(
            device=device,
            is_main_process=rank == 0,
            rank=rank,
            world_size=world_size,
            accelerator=None,
            local_rank=local_rank,
        )

    return BackendState(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        is_main_process=True,
        rank=0,
        world_size=1,
        accelerator=None,
        local_rank=0,
    )


def gather_ddp_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size <= 1 or not dist.is_initialized():
        return tensor
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)

