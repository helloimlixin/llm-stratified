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


def _multilabel_micro_f1_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    preds = logits > 0
    targets = labels > 0.5
    tp = torch.logical_and(preds, targets).sum(dtype=torch.float32)
    fp = torch.logical_and(preds, ~targets).sum(dtype=torch.float32)
    fn = torch.logical_and(~preds, targets).sum(dtype=torch.float32)
    denom = (2 * tp) + fp + fn
    perfect_empty = torch.ones((), device=logits.device, dtype=torch.float32)
    return torch.where(denom > 0, (2 * tp) / denom.clamp_min(1.0), perfect_empty)


def _accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type == "multilabel":
        return _multilabel_micro_f1_from_logits(logits, labels)
    return (logits.argmax(dim=-1) == labels).float().mean()


def _prepare_batch(
    batch,
    device: torch.device,
    task_type: str,
    *,
    non_blocking: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    imgs = batch[0].to(device, non_blocking=non_blocking)
    labels = batch[1].to(device, non_blocking=non_blocking)
    labels = labels.float() if task_type == "multilabel" else labels.long()
    return imgs, labels


def _finalize_epoch(total_loss: float, total_acc: float, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    return total_loss / total, total_acc / total


def _maybe_progress(loader, *, desc: str, enabled: bool):
    if enabled and tqdm is not None:
        return tqdm(loader, desc=desc, leave=False)
    return loader


def _log_interval_progress(
    *,
    phase: str,
    dataset_name: str,
    epoch: int,
    step: int,
    total_steps: Optional[int],
    total_loss: float,
    total_acc: float,
    total_items: int,
) -> None:
    if total_items <= 0:
        return
    total_steps_str = str(total_steps) if total_steps is not None else "?"
    avg_loss, avg_acc = _finalize_epoch(total_loss, total_acc, total_items)
    print(
        f"[{dataset_name}] {phase} {epoch:03d} step {step}/{total_steps_str} | "
        f"loss {avg_loss:.4f} | acc {avg_acc:.4f}",
        flush=True,
    )


@torch.no_grad()
def multilabel_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Backward-compatible name for the multilabel micro-F1 training metric."""
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
    log_interval: int = 0,
    dataset_name: str = "dataset",
) -> Tuple[float, float]:
    model.train()
    criterion = get_criterion(task_type, label_smoothing)
    if sampler is not None:
        sampler.set_epoch(epoch)
    total_loss, total_acc, total = 0.0, 0.0, 0
    total_steps = len(loader) if hasattr(loader, "__len__") else None
    for step, batch in enumerate(loader, start=1):
        imgs, labels = _prepare_batch(batch, device, task_type, non_blocking=True)
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
        if log_interval > 0 and step % log_interval == 0:
            _log_interval_progress(
                phase="Train",
                dataset_name=dataset_name,
                epoch=epoch,
                step=step,
                total_steps=total_steps,
                total_loss=total_loss,
                total_acc=total_acc,
                total_items=total,
            )
    return _finalize_epoch(total_loss, total_acc, total)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_type: str = "multiclass",
    label_smoothing: float = 0.0,
    epoch: int = 0,
    log_interval: int = 0,
    dataset_name: str = "dataset",
) -> Tuple[float, float]:
    model.eval()
    criterion = get_criterion(task_type, label_smoothing)
    total_loss, total_acc, total = 0.0, 0.0, 0
    total_steps = len(loader) if hasattr(loader, "__len__") else None
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            imgs, labels = _prepare_batch(batch, device, task_type, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_acc += _accuracy_from_logits(logits, labels, task_type).item() * bs
            total += bs
            if log_interval > 0 and step % log_interval == 0:
                _log_interval_progress(
                    phase="Eval",
                    dataset_name=dataset_name,
                    epoch=epoch,
                    step=step,
                    total_steps=total_steps,
                    total_loss=total_loss,
                    total_acc=total_acc,
                    total_items=total,
                )
    return _finalize_epoch(total_loss, total_acc, total)


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
    iterator = _maybe_progress(
        loader,
        desc=f"Train {epoch:03d}",
        enabled=show_progress and accelerator.is_main_process,
    )
    for batch in iterator:
        imgs, labels = _prepare_batch(batch, accelerator.device, task_type)
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
    iterator = _maybe_progress(loader, desc="Eval", enabled=show_progress and accelerator.is_main_process)
    for batch in iterator:
        imgs, labels = _prepare_batch(batch, accelerator.device, task_type)
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
