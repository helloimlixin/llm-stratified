#!/usr/bin/env python
# coding: utf-8
"""
TinyViT training script (PyTorch-only) with dataset factory, CLI flags,
warmup+cosine LR, label smoothing, optional torch.compile, optional wandb,
and multi-GPU support via DistributedDataParallel (DDP).

Supported datasets (torchvision):
- CIFAR10, CIFAR100
- STL10
- Food101
- Flowers102
- SVHN
- CelebA (multi-label, 40 attributes)

Examples:
  # Single GPU
  python tinyvit.py --dataset CIFAR10 --epochs 20 --wandb --project tinyvit

  # Multi-GPU (auto-detected, uses DDP automatically)
  python tinyvit.py --dataset FOOD101 --img-size 224 --epochs 20 --wandb --project tinyvit

  # Multi-GPU with torchrun (recommended)
  torchrun --nproc_per_node=2 tinyvit.py --dataset CIFAR10 --epochs 20 --wandb --project tinyvit

  # Explicit DDP
  python tinyvit.py --dataset CIFAR10 --ddp --epochs 20 --wandb --project tinyvit
"""

import os
import math
import argparse
from datetime import datetime
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import torchvision
import torchvision.transforms as T
try:
    import wandb  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore[assignment]

# -----------------------------
# Model: TinyViT
# -----------------------------


class PatchEmbed(nn.Module):
    """Converts an image batch into a sequence of flattened patch embeddings."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Return per-patch embeddings and the spatial token count."""

        x = self.proj(x)
        B, E, H2, W2 = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H2 * W2


class MlpBlock(nn.Module):
    """Feed-forward block used inside the transformer encoder."""

    def __init__(self, embed_dim: int, mlp_ratio: float = 2.0, dropout_rate: float = 0.1) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the two-layer MLP with GELU activation and dropout."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block with pre-norm and dropout."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.drop_path1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MlpBlock(embed_dim, mlp_ratio, dropout_rate)
        self.drop_path2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention followed by the MLP block."""

        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop_path1(y)

        y2 = self.norm2(x)
        y2 = self.mlp(y2)
        x = x + self.drop_path2(y2)
        return x


class TinyViT(nn.Module):
    """Minimal Vision Transformer for small/medium-sized datasets."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 8,
        num_heads: int = 3,
        mlp_ratio: float = 2.0,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout_rate)

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout_rate) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

        def _init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute token features prior to classification head."""

        batch_size = x.shape[0]
        x, _ = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for the provided mini-batch."""

        x = self.forward_features(x)
        cls_out = x[:, 0]
        logits = self.head(cls_out)
        return logits


# -----------------------------
# Dataset factory
# -----------------------------
def build_dataset(
    name: str = "CIFAR10",
    root: str = "./data",
    img_size: Optional[int] = None,
    split_celebA: str = "train",
) -> Tuple[Dataset, Dataset, int, int, int, str]:
    """Construct a torchvision dataset pair plus metadata.

    Returns a tuple of (train_dataset, eval_dataset, num_classes, in_channels,
    resolved_image_size, task_type).
    """
    name = name.upper()
    default_img = {
        "CIFAR10": 32,
        "CIFAR100": 32,
        "STL10": 96,
        "FOOD101": 224,
        "FLOWERS102": 224,
        "CELEBA": 64,
        "SVHN": 32,
        "IMAGENET": 256,
        "FFHQ": 256,
    }
    if img_size is None:
        img_size = default_img.get(name, 32)

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2470, 0.2435, 0.2616)

    def make_aug(norm_mean, norm_std, img_size, crop_pad=4, heavy=False):
        train_tf = [
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)) if heavy else T.RandomCrop(img_size, padding=crop_pad),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(),
            T.Normalize(norm_mean, norm_std),
        ]
        test_tf = [
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(norm_mean, norm_std),
        ]
        return T.Compose(train_tf), T.Compose(test_tf)

    if name == "CIFAR10":
        train_tf, test_tf = make_aug(cifar_mean, cifar_std, img_size, crop_pad=4, heavy=False)
        train_ds = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
        test_ds = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_tf)
        num_classes, task = 10, "multiclass"

    elif name == "CIFAR100":
        train_tf, test_tf = make_aug(cifar_mean, cifar_std, img_size, crop_pad=4, heavy=False)
        train_ds = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=train_tf)
        test_ds = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=test_tf)
        num_classes, task = 100, "multiclass"

    elif name == "STL10":
        train_tf, test_tf = make_aug(imagenet_mean, imagenet_std, img_size, crop_pad=8, heavy=True)
        train_ds = torchvision.datasets.STL10(root=root, split="train", download=True, transform=train_tf)
        test_ds = torchvision.datasets.STL10(root=root, split="test", download=True, transform=test_tf)
        num_classes, task = 10, "multiclass"

    elif name == "FOOD101":
        train_tf, test_tf = make_aug(imagenet_mean, imagenet_std, img_size, crop_pad=16, heavy=True)
        train_ds = torchvision.datasets.Food101(root=root, split="train", download=True, transform=train_tf)
        test_ds = torchvision.datasets.Food101(root=root, split="test", download=True, transform=test_tf)
        num_classes, task = 101, "multiclass"

    elif name == "FLOWERS102":
        train_tf, test_tf = make_aug(imagenet_mean, imagenet_std, img_size, crop_pad=16, heavy=True)
        train_ds = torchvision.datasets.Flowers102(root=root, split="train", download=True, transform=train_tf)
        test_ds = torchvision.datasets.Flowers102(root=root, split="test", download=True, transform=test_tf)
        num_classes, task = 102, "multiclass"

    elif name == "SVHN":
        train_tf, test_tf = make_aug(cifar_mean, cifar_std, img_size, crop_pad=4, heavy=False)
        train_ds = torchvision.datasets.SVHN(root=root, split="train", download=True, transform=train_tf)
        test_ds = torchvision.datasets.SVHN(root=root, split="test", download=True, transform=test_tf)
        num_classes, task = 10, "multiclass"

    elif name == "IMAGENET":
        # Expects ImageNet-style layout: root/train, root/val with class subfolders
        train_tf = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(),
            T.Normalize(imagenet_mean, imagenet_std),
        ])
        test_tf = T.Compose([
            T.Resize(int(img_size * 1.14)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(imagenet_mean, imagenet_std),
        ])
        train_ds = torchvision.datasets.ImageNet(root=root, split="train", transform=train_tf)
        test_ds = torchvision.datasets.ImageNet(root=root, split="val",   transform=test_tf)
        num_classes, task = 1000, "multiclass"

    elif name == "FFHQ":
        # Expects ImageFolder-style layout: root/train, root/val (or both under root, adjust as needed)
        train_tf = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(),
            T.Normalize(imagenet_mean, imagenet_std),
        ])
        test_tf = T.Compose([
            T.Resize(int(img_size * 1.14)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(imagenet_mean, imagenet_std),
        ])
        train_root = os.path.join(root, "train") if os.path.isdir(os.path.join(root, "train")) else root
        val_root = os.path.join(root, "val") if os.path.isdir(os.path.join(root, "val")) else root
        train_ds = torchvision.datasets.ImageFolder(root=train_root, transform=train_tf)
        test_ds = torchvision.datasets.ImageFolder(root=val_root,   transform=test_tf)
        num_classes, task = len(train_ds.classes), "multiclass"

    elif name == "CELEBA":
        # Multi-label (40 attributes)
        def celebA_transforms(train=True):
            ops = []
            ops += [T.CenterCrop(178)]
            if train:
                ops += [T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                        T.RandomHorizontalFlip(),
                        T.ColorJitter(0.2, 0.2, 0.2)]
            else:
                ops += [T.Resize(img_size), T.CenterCrop(img_size)]
            ops += [T.ToTensor(), T.Normalize(imagenet_mean, imagenet_std)]
            return T.Compose(ops)

        train_ds = torchvision.datasets.CelebA(
            root=root,
            split="train" if split_celebA == "train" else split_celebA,
            download=True,
            transform=celebA_transforms(True),
        )
        test_ds = torchvision.datasets.CelebA(
            root=root,
            split="test",
            download=True,
            transform=celebA_transforms(False),
        )
        num_classes, task = 40, "multilabel"

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train_ds, test_ds, num_classes, 3, img_size, task


# -----------------------------
# Data loaders
# -----------------------------
def make_loaders(
    dataset_name: str = "CIFAR10",
    root: str = "./data",
    img_size: Optional[int] = None,
    batch_size_train: int = 128,
    batch_size_test: int = 256,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader, int, int, int, str]:
    """Instantiate training and evaluation data loaders."""

    train_ds, test_ds, num_classes, in_chans, img_size, task = build_dataset(dataset_name, root, img_size)

    pin = (device is not None and device.type == "cuda")
    pw = (num_workers > 0)

    # Use DistributedSampler for multi-GPU training
    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        test_sampler = DistributedSampler(
            test_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        shuffle_train = False  # sampler handles shuffling
    else:
        train_sampler = None
        test_sampler = None
        shuffle_train = True

    train_loader = DataLoader(
        train_ds, batch_size=batch_size_train, shuffle=shuffle_train,
        sampler=train_sampler, drop_last=True, num_workers=num_workers,
        pin_memory=pin, persistent_workers=pw
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size_test, shuffle=False,
        sampler=test_sampler, drop_last=False, num_workers=num_workers,
        pin_memory=pin, persistent_workers=pw
    )
    return train_loader, test_loader, num_classes, in_chans, img_size, task


# -----------------------------
# Losses & Metrics
# -----------------------------
def get_criterion(task_type: str, label_smoothing: float = 0.0) -> nn.Module:
    """Select the proper loss function for the classification regime."""

    if task_type == "multilabel":
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


@torch.no_grad()
def multilabel_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Average per-label accuracy across the batch."""

    preds = (logits > 0).to(targets.dtype)
    return (preds == targets).float().mean().item()

# -----------------------------
# Train / Eval loops
# -----------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    task_type: str = "multiclass",
    grad_clip: Optional[float] = None,
    label_smoothing: float = 0.0,
    epoch: int = 0,
    sampler: Optional[DistributedSampler] = None,
) -> Tuple[float, float]:
    """Train the model for a single epoch and return loss/accuracy."""

    model.train()
    criterion = get_criterion(task_type, label_smoothing)

    if sampler is not None:
        sampler.set_epoch(epoch)

    total_loss, total_acc, total = 0.0, 0.0, 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        if task_type == "multilabel":
            labels = labels.to(device, non_blocking=True).float()
        else:
            labels = labels.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            bs = labels.size(0)
            total_loss += loss.item() * bs
            if task_type == "multilabel":
                acc = multilabel_accuracy(logits, labels)
                total_acc += acc * bs
            else:
                preds = logits.argmax(dim=-1)
                total_acc += (preds == labels).float().mean().item() * bs
            total += bs

    return total_loss / total, total_acc / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_type: str = "multiclass",
    label_smoothing: float = 0.0,
) -> Tuple[float, float]:
    """Evaluate the model without gradient tracking."""

    model.eval()
    criterion = get_criterion(task_type, label_smoothing)
    total_loss, total_acc, total = 0.0, 0.0, 0
    with torch.inference_mode():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            if task_type == "multilabel":
                labels = labels.to(device, non_blocking=True).float()
            else:
                labels = labels.to(device, non_blocking=True).long()

            logits = model(imgs)
            loss = criterion(logits, labels)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            if task_type == "multilabel":
                acc = multilabel_accuracy(logits, labels)
                total_acc += acc * bs
            else:
                preds = logits.argmax(dim=-1)
                total_acc += (preds == labels).float().mean().item() * bs
            total += bs

    return total_loss / total, total_acc / total


# -----------------------------
# Training driver
# -----------------------------
def run_training(
    dataset_name: str = "CIFAR10",
    root: str = "./data",
    num_runs: int = 1,
    num_epochs: int = 10,
    save_interval: int = 2,
    lr: float = 3e-4,
    wd: float = 0.05,
    grad_clip: Optional[float] = 1.0,
    base_dir: str = "./tinyvit_runs",
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
    use_ddp: bool = False,
    local_rank: int = 0,
    world_size: int = 1,
) -> None:
    """High-level orchestration for single or multi-GPU TinyViT training."""

    # Setup distributed training
    if use_ddp:
        # Check if already initialized (e.g., by torchrun)
        if not dist.is_initialized():
            # Manual initialization for non-torchrun launches
            rank = int(os.environ.get("RANK", local_rank))
            world_size = int(os.environ.get("WORLD_SIZE", world_size))
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            master_port = os.environ.get("MASTER_PORT", "12355")
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_addr}:{master_port}",
                rank=rank,
                world_size=world_size,
            )
        else:
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        is_main_process = (rank == 0)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True

    if is_main_process:
        os.makedirs(base_dir, exist_ok=True)

    cudnn.benchmark = (device.type == "cuda")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # Adjust batch size per GPU for distributed training
    effective_batch_size_train = batch_size_train
    effective_batch_size_test = batch_size_test
    if use_ddp:
        effective_batch_size_train = batch_size_train // world_size
        effective_batch_size_test = batch_size_test // world_size
        if is_main_process:
            ddp_msg = (
                f"[DDP] Using {world_size} GPUs, "
                f"batch size per GPU: train={effective_batch_size_train}, "
                f"test={effective_batch_size_test}"
            )
            print(ddp_msg)

    # data/loaders
    train_loader, test_loader, num_classes, in_chans, final_img_size, task = make_loaders(
        dataset_name=dataset_name,
        root=root,
        img_size=img_size,
        batch_size_train=effective_batch_size_train,
        batch_size_test=effective_batch_size_test,
        num_workers=num_workers,
        device=device,
        distributed=use_ddp,
        rank=rank,
        world_size=world_size
    )

    # Get sampler for epoch setting
    train_sampler = train_loader.sampler if use_ddp else None

    # wandb (optional) - only init on main process
    if wandb_on and is_main_process:
        if wandb is None:
            print("[wandb] ERROR: wandb is not installed; disabling logging")
            wandb_on = False
        else:
            try:
                wandb_mode = os.environ.get("WANDB_MODE", "online")
                if wandb_mode == "online":
                    try:
                        api_key = wandb.api.api_key
                        if api_key is None:
                            print("[wandb] WARNING: Not logged in. Use 'wandb login' or set WANDB_MODE=offline")
                            print("[wandb] Continuing in offline mode...")
                            os.environ["WANDB_MODE"] = "offline"
                    except Exception:
                        print("[wandb] WARNING: Could not check login status. Continuing...")

                wandb_config = dict(
                    dataset=dataset_name,
                    img_size=final_img_size,
                    patch_size=patch_size,
                    embed_dim=embed_dim,
                    depth=depth,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_rate=dropout_rate,
                    lr=lr,
                    wd=wd,
                    label_smoothing=label_smoothing,
                    epochs=num_epochs,
                    cosine=cosine,
                    batch_size_train=batch_size_train,
                    batch_size_test=batch_size_test,
                    effective_batch_size_train=(
                        effective_batch_size_train * world_size if use_ddp else batch_size_train
                    ),
                    effective_batch_size_test=(
                        effective_batch_size_test * world_size if use_ddp else batch_size_test
                    ),
                    num_gpus=world_size,
                    use_ddp=use_ddp,
                    compile_model=compile_model,
                    grad_clip=grad_clip,
                    warmup_epochs=warmup_epochs,
                )
                wandb.init(project=wandb_project, name=wandb_runname, config=wandb_config)
                cached_run = wandb.run.name if wandb.run else wandb_runname
                print(
                    f"[wandb] Initialized successfully - Project: {wandb_project}, "
                    f"Run: {cached_run}"
                )
                print(f"[wandb] View run at: {wandb.run.url if wandb.run else 'N/A'}")
            except Exception as e:
                print(f"[wandb] ERROR: Failed to initialize wandb: {e}")
                import traceback

                traceback.print_exc()
                wandb_on = False

    for run_idx in range(num_runs):
        if is_main_process:
            run_dir = os.path.join(base_dir, f"{dataset_name}_run_{run_idx:03d}")
            os.makedirs(run_dir, exist_ok=True)
        else:
            run_dir = None

        seed = seed_base + run_idx
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        model = TinyViT(
            img_size=final_img_size, patch_size=patch_size, in_chans=in_chans,
            num_classes=num_classes, embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio, dropout_rate=dropout_rate
        ).to(device)

        # Wrap model with DDP for multi-GPU training
        if use_ddp:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False
            )
            model_for_saving = model.module  # unwrapped model for saving
        else:
            model_for_saving = model

        if compile_model and hasattr(torch, "compile"):
            model = torch.compile(model)

        # Scale learning rate by world size for DDP (linear scaling rule)
        effective_lr = lr * world_size if use_ddp else lr
        optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=wd)

        # scheduler: cosine with linear warmup
        if warmup_epochs is None:
            warmup_epochs = max(1, min(5, int(0.1 * num_epochs)))
        if cosine:
            def lr_lambda(e):
                if e < warmup_epochs:
                    return (e + 1) / max(1, warmup_epochs)
                t = (e - warmup_epochs) / max(1, (num_epochs - warmup_epochs))
                return 0.5 * (1.0 + math.cos(math.pi * t))
        else:
            def lr_lambda(e):
                if e < warmup_epochs:
                    return (e + 1) / max(1, warmup_epochs)
                return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        if is_main_process:
            print(f"\nStarting {dataset_name} run {run_idx+1}/{num_runs} → {run_dir}")
            if use_ddp:
                print(f"[DDP] Effective LR: {effective_lr:.2e} (base: {lr:.2e} × {world_size} GPUs)")

        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scaler, device,
                task_type=task, grad_clip=grad_clip, label_smoothing=label_smoothing,
                epoch=epoch, sampler=train_sampler
            )
            eval_loss, eval_acc = evaluate(
                model, test_loader, device, task_type=task, label_smoothing=label_smoothing
            )
            scheduler.step()

            # Synchronize metrics across GPUs for DDP
            if use_ddp:
                # Average metrics across all GPUs
                metrics_tensor = torch.tensor([train_loss, train_acc, eval_loss, eval_acc], device=device)
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                metrics_tensor /= world_size
                train_loss, train_acc, eval_loss, eval_acc = metrics_tensor.cpu().tolist()

            lr_now = scheduler.get_last_lr()[0]

            if is_main_process:
                msg = (f"[{dataset_name}] Epoch {epoch:03d} | lr {lr_now:.2e} | "
                       f"train {train_loss:.4f}/{train_acc:.4f} | "
                       f"val {eval_loss:.4f}/{eval_acc:.4f}")
                if use_ddp:
                    msg += f" | GPUs: {world_size}"
                print(msg)

                if wandb_on and wandb is not None:
                    try:
                        log_dict = {
                            "epoch": epoch,
                            "lr": lr_now,
                            "train/loss": train_loss,
                            "train/acc": train_acc,
                            "val/loss": eval_loss,
                            "val/acc": eval_acc,
                        }
                        wandb.log(log_dict)
                        if epoch == 0 or epoch % 10 == 0:
                            print(f"[wandb] Logged metrics for epoch {epoch}")
                    except Exception as e:
                        print(f"[wandb] ERROR: Failed to log metrics: {e}")
                        import traceback

                        traceback.print_exc()

                if epoch == 0 or (epoch % save_interval == 0) or (epoch == num_epochs - 1):
                    ckpt_path = os.path.join(run_dir, f"epoch_{epoch:03d}.pt")
                    torch.save({
                        "epoch": epoch,
                        "model_state": model_for_saving.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "eval_loss": eval_loss,
                        "eval_acc": eval_acc,
                        "dataset": dataset_name,
                        "task": task,
                        "img_size": final_img_size,
                        "patch_size": patch_size,
                        "timestamp": datetime.now().isoformat(),
                        "seed": seed,
                        "num_gpus": world_size,
                        "use_ddp": use_ddp,
                    }, ckpt_path)
                    print(f"Saved checkpoint → {ckpt_path}")

    # Cleanup
    if wandb_on and is_main_process and wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass

    if use_ddp:
        dist.destroy_process_group()


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the TinyViT script."""

    p = argparse.ArgumentParser(description="TinyViT training (PyTorch-only)")

    # data
    p.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        choices=[
            "CIFAR10",
            "CIFAR100",
            "STL10",
            "FOOD101",
            "FLOWERS102",
            "SVHN",
            "CELEBA",
            "IMAGENET",
            "FFHQ",
        ],
    )
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--batch-size-test", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)

    # model
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--num-heads", type=int, default=3)
    p.add_argument("--mlp-ratio", type=float, default=2.0)
    p.add_argument("--dropout", type=float, default=0.1)

    # train
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--save-interval", type=int, default=2)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--seed-base", type=int, default=1337)
    p.add_argument("--outdir", type=str, default="./tinyvit_runs")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--no-cosine", action="store_true", help="Disable cosine schedule after warmup")
    p.add_argument("--compile", action="store_true", help="Use torch.compile if available")

    # wandb
    p.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    p.add_argument("--project", type=str, default="tinyvit", help="Wandb project name")
    p.add_argument("--run-name", type=str, default=None, help="Wandb run name")

    # multi-GPU
    p.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel for multi-GPU training")
    p.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="Local rank for distributed training (auto-set by torchrun)",
    )

    return p.parse_args()


def main() -> None:
    """Entry point that parses arguments and dispatches training."""

    args = parse_args()

    # Auto-detect multi-GPU if not explicitly set
    use_ddp = args.ddp
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if not use_ddp and num_gpus > 1:
        # Auto-enable DDP if multiple GPUs detected
        use_ddp = True
        print(f"[Auto] Detected {num_gpus} GPUs, enabling DDP mode")

    # Get local rank from environment (set by torchrun) or use CLI arg
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    # Set world size from environment if available (torchrun sets this)
    world_size = int(os.environ.get("WORLD_SIZE", num_gpus if use_ddp else 1))

    if use_ddp and world_size == 1:
        # If DDP requested but only 1 GPU, disable DDP
        use_ddp = False
        print("[Warning] DDP requested but only 1 GPU available, disabling DDP")

    # Set environment variables for DDP if using torchrun
    if use_ddp and "RANK" not in os.environ:
        # Not launched with torchrun, will init in run_training
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

    run_training(
        dataset_name=args.dataset,
        root=args.root,
        num_runs=args.runs,
        num_epochs=args.epochs,
        save_interval=args.save_interval,
        lr=args.lr,
        wd=args.wd,
        grad_clip=args.grad_clip,
        base_dir=args.outdir,
        seed_base=args.seed_base,
        num_workers=args.num_workers,
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout_rate=args.dropout,
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
        cosine=(not args.no_cosine),
        compile_model=args.compile,
        wandb_on=args.wandb,
        wandb_project=args.project,
        wandb_runname=args.run_name,
        batch_size_train=args.batch_size,
        batch_size_test=args.batch_size_test,
        use_ddp=use_ddp,
        local_rank=local_rank,
        world_size=world_size,
    )


if __name__ == "__main__":
    main()
