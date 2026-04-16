"""Embedding animation: frame building and GIF generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from fiber.visualization import (
    HAS_MATPLOTLIB,
    _require_matplotlib,
    project_embeddings_2d,
)


def _labels_to_animation_classes(labels: torch.Tensor) -> np.ndarray:
    lbl = labels.detach().cpu()
    if lbl.ndim == 0:
        return np.asarray([int(lbl.item())], dtype=np.int64)
    if lbl.ndim == 1:
        return lbl.to(torch.int64).numpy().reshape(-1)
    if lbl.ndim == 2:
        if lbl.shape[1] == 0:
            return np.full((lbl.shape[0],), -1, dtype=np.int64)
        positive = lbl > 0
        has_positive = positive.any(dim=1)
        classes = torch.argmax(lbl, dim=1).to(torch.int64)
        classes = torch.where(has_positive, classes, torch.full_like(classes, -1))
        return classes.numpy()
    flat = lbl.reshape(lbl.shape[0], -1)
    return flat[:, 0].to(torch.int64).numpy()


def build_embedding_animation_frames(
    snapshots: list[tuple[int, torch.Tensor, torch.Tensor]],
) -> list[dict[str, Any]]:
    if not snapshots:
        return []
    reference_embeddings = next(
        (
            embeddings
            for _epoch, embeddings, _labels in reversed(snapshots)
            if embeddings.ndim == 2 and embeddings.shape[0] > 0 and embeddings.shape[1] > 0
        ),
        snapshots[-1][1],
    )
    _, final_mean, final_basis = project_embeddings_2d(reference_embeddings)

    frames: list[dict[str, Any]] = []
    for epoch, embeddings, labels in snapshots:
        coords2d, _, _ = project_embeddings_2d(embeddings, mean=final_mean, basis=final_basis)
        class_ids = _labels_to_animation_classes(labels)
        n_points = min(coords2d.shape[0], class_ids.shape[0])
        if n_points == 0:
            continue
        encodings = np.column_stack((coords2d[:n_points, 0], coords2d[:n_points, 1], class_ids[:n_points]))
        frames.append({"epoch": int(epoch), "encodings": encodings})
    return frames


def _plot_embedding_animation_frame(
    ax,
    data: np.ndarray,
    frame_title: str,
    *,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
) -> None:
    plt_mod = _require_matplotlib()
    if data.size == 0:
        ax.set_title(frame_title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        return

    coords = np.asarray(data[:, :2], dtype=np.float64)
    class_ids = np.asarray(data[:, 2], dtype=np.int64)
    finite = np.isfinite(coords).all(axis=1)
    coords, class_ids = coords[finite], class_ids[finite]
    order = np.argsort(class_ids, kind="mergesort")
    coords, class_ids = coords[order], class_ids[order]

    labels = np.unique(class_ids)
    cmap = plt_mod.get_cmap("tab20", max(1, min(20, len(labels))))
    for idx, label in enumerate(labels.tolist()):
        mask = class_ids == label
        color = "#808080" if label < 0 else cmap(idx % getattr(cmap, "N", 20))
        ax.scatter(coords[mask, 0], coords[mask, 1], label=str(int(label)), alpha=0.8, s=8, c=[color], linewidths=0)

    ax.set_title(frame_title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(alpha=0.2, linewidth=0.5)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if len(labels) <= 12:
        ax.legend(loc="best", fontsize=8, markerscale=1.5, frameon=True)


def generate_embedding_animation(
    encodings: list[dict[str, Any]],
    title: str = "Embedding Space Progression",
    *,
    output_path: str | Path = "results/embedding_progression.gif",
    fps: int = 4,
) -> Path:
    plt_mod = _require_matplotlib()
    if not encodings:
        raise ValueError("encodings must contain at least one frame")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = [np.asarray(frame["encodings"], dtype=np.float64) for frame in encodings]
    nonempty = [frame for frame in arrays if frame.size]
    if nonempty:
        all_x = np.concatenate([frame[:, 0] for frame in nonempty], axis=0)
        all_y = np.concatenate([frame[:, 1] for frame in nonempty], axis=0)
        x_margin = (all_x.max() - all_x.min()) * 0.05 or 0.1
        y_margin = (all_y.max() - all_y.min()) * 0.05 or 0.1
        x_lim = (float(all_x.min() - x_margin), float(all_x.max() + x_margin))
        y_lim = (float(all_y.min() - y_margin), float(all_y.max() + y_margin))
    else:
        x_lim = y_lim = None

    fig, ax = plt_mod.subplots(figsize=(8, 6))
    from matplotlib.animation import FuncAnimation, PillowWriter

    def update(frame_idx: int) -> None:
        ax.clear()
        frame = encodings[frame_idx]
        frame_epoch = int(frame.get("epoch", frame_idx))
        _plot_embedding_animation_frame(
            ax, arrays[frame_idx], f"{title} at Epoch {frame_epoch}",
            x_lim=x_lim, y_lim=y_lim,
        )

    anim = FuncAnimation(fig, update, frames=len(encodings), interval=max(1, int(1000 / max(1, fps))))
    anim.save(path, writer=PillowWriter(fps=max(1, fps)))
    plt_mod.close(fig)
    return path
