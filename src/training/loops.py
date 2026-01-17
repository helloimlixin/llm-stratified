from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def get_criterion(task_type: str, label_smoothing: float = 0.0) -> nn.Module:
    return nn.BCEWithLogitsLoss() if task_type == "multilabel" else nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def _accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type == "multilabel":
        return ((logits > 0).to(labels.dtype) == labels).float().mean()
    return (logits.argmax(dim=-1) == labels).float().mean()


@torch.no_grad()
def multilabel_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return _accuracy_from_logits(logits, targets, "multilabel").item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[object],
    device: torch.device,
    task_type: str = "multiclass",
    grad_clip: Optional[float] = None,
    label_smoothing: float = 0.0,
    epoch: int = 0,
    sampler: Optional[DistributedSampler] = None,
) -> Tuple[float, float]:
    model.train()
    criterion = get_criterion(task_type, label_smoothing)
    if sampler is not None:
        sampler.set_epoch(epoch)
    total_loss, total_acc, total = 0.0, 0.0, 0
    for batch in loader:
        imgs, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        labels = labels.float() if task_type == "multilabel" else labels.long()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        with torch.no_grad():
            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_acc += _accuracy_from_logits(logits, labels, task_type).item() * bs
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
    model.eval()
    criterion = get_criterion(task_type, label_smoothing)
    total_loss, total_acc, total = 0.0, 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            imgs, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            labels = labels.float() if task_type == "multilabel" else labels.long()
            logits = model(imgs)
            loss = criterion(logits, labels)
            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_acc += _accuracy_from_logits(logits, labels, task_type).item() * bs
            total += bs
    return total_loss / total, total_acc / total


def train_one_epoch_accelerate(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accelerator,
    task_type: str = "multiclass",
    grad_clip: Optional[float] = None,
    label_smoothing: float = 0.0,
    epoch: int = 0,
    show_progress: bool = True,
) -> Tuple[float, float]:
    model.train()
    criterion = get_criterion(task_type, label_smoothing)
    if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    loss_sum = torch.tensor(0.0, device=accelerator.device)
    acc_sum = torch.tensor(0.0, device=accelerator.device)
    count_sum = torch.tensor(0.0, device=accelerator.device)
    iterator = tqdm(loader, desc=f"Train {epoch:03d}", leave=False) if show_progress and accelerator.is_main_process and tqdm else loader
    for batch in iterator:
        imgs, labels = batch[0].to(accelerator.device), batch[1].to(accelerator.device)
        labels = labels.float() if task_type == "multilabel" else labels.long()
        with accelerator.autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)
        accelerator.backward(loss)
        if grad_clip:
            accelerator.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            bs = labels.size(0)
            loss_sum += loss.detach() * bs
            acc_sum += _accuracy_from_logits(logits, labels, task_type) * bs
            count_sum += bs
    loss_sum = accelerator.reduce(loss_sum, reduction="sum")
    acc_sum = accelerator.reduce(acc_sum, reduction="sum")
    count_sum = accelerator.reduce(count_sum, reduction="sum")
    return (loss_sum / count_sum).item() if count_sum > 0 else float("nan"), (acc_sum / count_sum).item() if count_sum > 0 else float("nan")


@torch.no_grad()
def evaluate_accelerate(
    model: nn.Module,
    loader: DataLoader,
    accelerator,
    task_type: str = "multiclass",
    label_smoothing: float = 0.0,
    show_progress: bool = True,
) -> Tuple[float, float]:
    model.eval()
    criterion = get_criterion(task_type, label_smoothing)
    loss_sum = torch.tensor(0.0, device=accelerator.device)
    acc_sum = torch.tensor(0.0, device=accelerator.device)
    count_sum = torch.tensor(0.0, device=accelerator.device)
    iterator = tqdm(loader, desc="Eval", leave=False) if show_progress and accelerator.is_main_process and tqdm else loader
    for batch in iterator:
        imgs, labels = batch[0].to(accelerator.device), batch[1].to(accelerator.device)
        labels = labels.float() if task_type == "multilabel" else labels.long()
        with accelerator.autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)
        bs = labels.size(0)
        loss_sum += loss.detach() * bs
        acc_sum += _accuracy_from_logits(logits, labels, task_type) * bs
        count_sum += bs
    loss_sum = accelerator.reduce(loss_sum, reduction="sum")
    acc_sum = accelerator.reduce(acc_sum, reduction="sum")
    count_sum = accelerator.reduce(count_sum, reduction="sum")
    return (loss_sum / count_sum).item() if count_sum > 0 else float("nan"), (acc_sum / count_sum).item() if count_sum > 0 else float("nan")

