from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from datasets import create_data_loaders, get_class_names
from fiber.animation import build_embedding_animation_frames, generate_embedding_animation
from fiber.analysis import analyze_fiber_epoch
from models import SamBackboneWrapper
from training.config import SamFiberConfig
from training.wandb_utils import finish_wandb_run, init_wandb_run
from utils import seed_everything, to_serializable


def _unwrap_base_dataset(dataset) -> Any:
    base = dataset
    while hasattr(base, "dataset"):
        base = base.dataset
    return base


def _limit_boxes(boxes: list[tuple[float, float, float, float, int]], max_boxes: int) -> list[tuple[float, float, float, float, int]]:
    if max_boxes <= 0 or len(boxes) <= max_boxes:
        return list(boxes)
    return sorted(boxes, key=lambda box: float((box[2] - box[0]) * (box[3] - box[1])), reverse=True)[:max_boxes]


def _boxes_to_patch_multihot(
    boxes: list[tuple[float, float, float, float, int]],
    *,
    grid_h: int,
    grid_w: int,
    image_h: int,
    image_w: int,
    num_classes: int,
) -> torch.Tensor:
    patches = grid_h * grid_w
    labels = torch.zeros((patches, num_classes), dtype=torch.float32)
    if grid_h <= 0 or grid_w <= 0 or image_h <= 0 or image_w <= 0 or num_classes <= 0:
        return labels
    for x0, y0, x1, y1, cat in boxes:
        cat_i = int(cat)
        if cat_i < 0 or cat_i >= num_classes:
            continue
        x0 = max(0.0, min(float(image_w), float(x0)))
        y0 = max(0.0, min(float(image_h), float(y0)))
        x1 = max(x0, min(float(image_w), float(x1)))
        y1 = max(y0, min(float(image_h), float(y1)))
        c0 = int(max(0, min(grid_w - 1, np.floor((x0 / max(1.0, image_w)) * grid_w))))
        c1 = int(max(0, min(grid_w - 1, np.floor(((x1 - 1e-6) / max(1.0, image_w)) * grid_w))))
        r0 = int(max(0, min(grid_h - 1, np.floor((y0 / max(1.0, image_h)) * grid_h))))
        r1 = int(max(0, min(grid_h - 1, np.floor(((y1 - 1e-6) / max(1.0, image_h)) * grid_h))))
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                labels[row * grid_w + col, cat_i] = 1.0
    return labels


def _grid_bboxes(*, image_h: int, image_w: int, grid_h: int, grid_w: int) -> torch.Tensor:
    y_edges = np.linspace(0, image_h, grid_h + 1)
    x_edges = np.linspace(0, image_w, grid_w + 1)
    boxes = []
    for row in range(grid_h):
        for col in range(grid_w):
            boxes.append(
                [
                    int(round(x_edges[col])),
                    int(round(y_edges[row])),
                    int(round(x_edges[col + 1])),
                    int(round(y_edges[row + 1])),
                ]
            )
    return torch.tensor(boxes, dtype=torch.int32)


def _pool_embedding_map(embedding_map: torch.Tensor, *, grid_h: int, grid_w: int) -> torch.Tensor:
    pooled = F.adaptive_avg_pool2d(embedding_map.float(), output_size=(grid_h, grid_w))
    return pooled.flatten(2).transpose(1, 2).squeeze(0).cpu()


def _image_level_pred_label(label: torch.Tensor) -> int:
    if not isinstance(label, torch.Tensor):
        return -1
    if label.dim() == 0:
        return int(label.item())
    flat = label.reshape(-1)
    if flat.numel() == 0 or float(flat.max().item()) <= 0:
        return -1
    return int(torch.argmax(flat).item())


def _fallback_patch_labels(label: torch.Tensor, patch_count: int, num_classes: int) -> torch.Tensor:
    if isinstance(label, torch.Tensor) and label.dim() == 1 and label.numel() == num_classes:
        return label.unsqueeze(0).repeat(patch_count, 1).to(dtype=torch.float32)
    labels = torch.zeros((patch_count, num_classes), dtype=torch.float32)
    pred_label = _image_level_pred_label(label if isinstance(label, torch.Tensor) else torch.tensor(-1))
    if 0 <= pred_label < num_classes:
        labels[:, pred_label] = 1.0
    return labels


def _masks_to_patch_multihot(
    masks: list[torch.Tensor],
    cats: list[int],
    *,
    grid_h: int,
    grid_w: int,
    num_classes: int,
    threshold: float,
    fallback: torch.Tensor,
) -> torch.Tensor:
    labels = torch.zeros((grid_h * grid_w, num_classes), dtype=torch.float32)
    any_valid = False
    for mask, cat in zip(masks, cats):
        cat_i = int(cat)
        if cat_i < 0 or cat_i >= num_classes:
            continue
        mask_t = torch.as_tensor(mask, dtype=torch.float32).squeeze()
        if mask_t.ndim != 2 or mask_t.numel() == 0:
            continue
        pooled = F.adaptive_avg_pool2d(mask_t.unsqueeze(0).unsqueeze(0), output_size=(grid_h, grid_w)).squeeze(0).squeeze(0)
        present = pooled >= float(threshold)
        if present.any():
            labels[present.reshape(-1), cat_i] = 1.0
            any_valid = True
    return labels if any_valid else fallback


def _mask_preview_palette() -> list[tuple[int, int, int]]:
    return [
        (230, 57, 70),
        (29, 53, 87),
        (69, 123, 157),
        (42, 157, 143),
        (244, 162, 97),
        (233, 196, 106),
        (168, 218, 220),
        (142, 202, 230),
    ]


def _overlay_masks(
    *,
    image: torch.Tensor,
    masks: list[torch.Tensor],
    boxes: list[tuple[float, float, float, float, int]],
    grid_h: int | None = None,
    grid_w: int | None = None,
) -> Image.Image | None:
    if Image is None or ImageDraw is None:
        return None
    image_np = (image.permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    base = Image.fromarray(image_np).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    palette = _mask_preview_palette()
    for idx, (mask, box) in enumerate(zip(masks, boxes)):
        mask_t = torch.as_tensor(mask, dtype=torch.float32).squeeze()
        if mask_t.ndim != 2:
            continue
        color = palette[idx % len(palette)]
        alpha = np.zeros((mask_t.shape[0], mask_t.shape[1], 4), dtype=np.uint8)
        alpha[..., 0] = color[0]
        alpha[..., 1] = color[1]
        alpha[..., 2] = color[2]
        alpha[..., 3] = (mask_t.clamp(0, 1).numpy() * 96.0).astype(np.uint8)
        mask_image = Image.fromarray(alpha, mode="RGBA").resize(base.size, resample=getattr(Image, "Resampling", Image).BILINEAR)
        overlay = Image.alpha_composite(overlay, mask_image)
    draw = ImageDraw.Draw(overlay)
    for idx, box in enumerate(boxes):
        color = palette[idx % len(palette)]
        x0, y0, x1, y1, _cat = box
        draw.rectangle((x0, y0, x1, y1), outline=color + (255,), width=2)
    if grid_h is not None and grid_w is not None and grid_h > 0 and grid_w > 0:
        width, height = base.size
        grid_color = (255, 255, 255, 90)
        for col in range(1, int(grid_w)):
            x = int(round(width * col / float(grid_w)))
            draw.line((x, 0, x, height), fill=grid_color, width=1)
        for row in range(1, int(grid_h)):
            y = int(round(height * row / float(grid_h)))
            draw.line((0, y, width, y), fill=grid_color, width=1)
    return Image.alpha_composite(base, overlay).convert("RGB")


def _make_image_grid(images: list[Image.Image], *, cols: int = 2, pad: int = 8) -> Image.Image | None:
    if Image is None or not images:
        return None
    cols = max(1, min(cols, len(images)))
    rows = int(np.ceil(len(images) / cols))
    width = max(img.width for img in images)
    height = max(img.height for img in images)
    grid = Image.new("RGB", (cols * width + (cols + 1) * pad, rows * height + (rows + 1) * pad), (14, 18, 24))
    for idx, image in enumerate(images):
        row, col = divmod(idx, cols)
        x = pad + col * (width + pad)
        y = pad + row * (height + pad)
        grid.paste(image.resize((width, height), resample=getattr(Image, "Resampling", Image).BILINEAR), (x, y))
    return grid


def _make_epoch_loader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    indices: list[int] | None,
) -> DataLoader:
    epoch_dataset = dataset if indices is None else Subset(dataset, indices)
    return DataLoader(
        epoch_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _label_name(class_id: int, class_names: list[str] | None) -> str:
    if class_names and 0 <= int(class_id) < len(class_names):
        return str(class_names[int(class_id)])
    return str(int(class_id))


def _preview_rows(
    previews: list[dict[str, Any]],
    *,
    class_names: list[str] | None,
    mask_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for preview in previews:
        categories = [int(cat) for cat in preview.get("categories", [])]
        masks = [torch.as_tensor(mask, dtype=torch.float32).squeeze() for mask in preview.get("masks", [])]
        image_tensor = torch.as_tensor(preview.get("image"), dtype=torch.float32)
        image_h = int(image_tensor.shape[-2]) if image_tensor.ndim >= 2 else 1
        image_w = int(image_tensor.shape[-1]) if image_tensor.ndim >= 1 else 1
        image_area = float(max(1, image_h * image_w))
        box_area_fraction = 0.0
        for box in preview.get("boxes", []):
            x0, y0, x1, y1, _cat = box
            box_area_fraction += max(0.0, float(x1) - float(x0)) * max(0.0, float(y1) - float(y0)) / image_area
        if masks:
            binary_masks = [(mask >= float(mask_threshold)).to(torch.float32) for mask in masks]
            area_fracs = [float(mask.mean().item()) for mask in binary_masks]
            union = torch.stack(binary_masks, dim=0).amax(dim=0)
            union_frac = float(union.mean().item())
            largest_idx = int(np.argmax(area_fracs))
            largest_frac = float(area_fracs[largest_idx])
            largest_cat = categories[largest_idx] if largest_idx < len(categories) else -1
            class_areas: dict[int, float] = {}
            for cat, frac in zip(categories, area_fracs):
                class_areas[int(cat)] = class_areas.get(int(cat), 0.0) + float(frac)
            class_area_text = ", ".join(
                f"{_label_name(cat, class_names)} {frac:.0%}"
                for cat, frac in sorted(class_areas.items(), key=lambda kv: (-kv[1], kv[0]))
            )
        else:
            area_fracs = []
            union_frac = 0.0
            largest_frac = 0.0
            largest_cat = -1
            class_area_text = "none"
        counts: dict[int, int] = {}
        for cat in categories:
            counts[int(cat)] = counts.get(int(cat), 0) + 1
        classes_text = ", ".join(f"{_label_name(cat, class_names)} x{count}" for cat, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))) or "none"
        if union_frac >= 0.5:
            coverage_regime = "dense"
        elif union_frac >= 0.2:
            coverage_regime = "moderate"
        else:
            coverage_regime = "sparse"
        description = (
            f"Image {int(preview.get('image_id', -1))}: {len(categories)} box prompts across {classes_text}. "
            f"Prompt boxes span {box_area_fraction:.0%} of the frame. "
            f"Thresholded SAM masks cover {union_frac:.0%} of the image in total, "
            f"with the largest {_label_name(largest_cat, class_names) if largest_cat >= 0 else 'n/a'} region at {largest_frac:.0%}. "
            f"Per-class mask coverage: {class_area_text}. Overall coverage looks {coverage_regime}."
        )
        rows.append(
            {
                "image_id": int(preview.get("image_id", -1)),
                "num_prompts": int(len(categories)),
                "classes": classes_text,
                "prompt_box_fraction": box_area_fraction,
                "union_mask_fraction": union_frac,
                "largest_mask_fraction": largest_frac,
                "largest_mask_class": _label_name(largest_cat, class_names) if largest_cat >= 0 else "n/a",
                "per_class_mask_coverage": class_area_text,
                "coverage_regime": coverage_regime,
                "description": description,
            }
        )
    return rows


def _preview_epoch_summary(
    previews: list[dict[str, Any]],
    *,
    preview_rows: list[dict[str, Any]],
    class_names: list[str] | None,
) -> dict[str, Any]:
    if not previews or not preview_rows:
        return {
            "num_preview_images": 0,
            "avg_prompts": 0.0,
            "avg_union_mask_fraction": 0.0,
            "avg_prompt_box_fraction": 0.0,
            "top_classes": "none",
            "description": "No segmentation previews were logged for this epoch.",
        }

    class_counts: dict[int, int] = {}
    for preview in previews:
        for cat in preview.get("categories", []):
            class_counts[int(cat)] = class_counts.get(int(cat), 0) + 1
    top_classes = ", ".join(
        f"{_label_name(cat, class_names)} x{count}"
        for cat, count in sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4]
    ) or "none"
    avg_prompts = float(np.mean([float(row["num_prompts"]) for row in preview_rows]))
    avg_union_mask_fraction = float(np.mean([float(row["union_mask_fraction"]) for row in preview_rows]))
    avg_prompt_box_fraction = float(np.mean([float(row["prompt_box_fraction"]) for row in preview_rows]))
    description = (
        f"Epoch preview summary: {len(preview_rows)} images, {avg_prompts:.1f} prompts per image on average, "
        f"{avg_prompt_box_fraction:.0%} prompt-box coverage, and {avg_union_mask_fraction:.0%} thresholded mask coverage. "
        f"Most frequent prompt classes: {top_classes}."
    )
    return {
        "num_preview_images": int(len(preview_rows)),
        "avg_prompts": avg_prompts,
        "avg_union_mask_fraction": avg_union_mask_fraction,
        "avg_prompt_box_fraction": avg_prompt_box_fraction,
        "top_classes": top_classes,
        "description": description,
    }


def _fmt_percent(value: Any) -> str:
    try:
        value_f = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(value_f):
        return "n/a"
    return f"{value_f:.0%}"


def _wandb_image_with_caption(wandb, image: Any, caption: str):
    try:
        return wandb.Image(image, caption=caption)
    except TypeError:
        return wandb.Image(image)


def _segmentation_epoch_caption(
    *,
    epoch: int,
    summary: dict[str, Any],
    model_name: str,
    dataset_name: str,
) -> str:
    union = float(summary.get("avg_union_mask_fraction", float("nan")))
    prompt = float(summary.get("avg_prompt_box_fraction", float("nan")))
    top_classes = str(summary.get("top_classes") or "n/a")
    if np.isfinite(union) and union >= 0.45:
        coverage_text = "SAM masks cover a large fraction of the preview images, so object labels should be visually easy to audit but may also include broad object/background regions."
    elif np.isfinite(union) and union >= 0.15:
        coverage_text = "SAM masks provide moderate object coverage, which is usually enough to inspect object-aligned patch labels while leaving substantial background context."
    elif np.isfinite(union):
        coverage_text = "SAM masks are sparse in this preview, so downstream patch-label conclusions should be read cautiously because many tokens may mostly see background."
    else:
        coverage_text = "Mask coverage could not be summarized reliably for this preview."
    return (
        f"Epoch {epoch} {dataset_name} segmentation preview for {model_name}. The overlays show COCO box prompts, thresholded SAM masks, and the analysis patch grid used to assign patch-token labels. "
        f"Conclusion: {coverage_text} Average prompt-box coverage is {_fmt_percent(prompt)} and average thresholded mask coverage is {_fmt_percent(union)}. The most frequent prompted classes are {top_classes}; repeated or overlapping boxes can make prompt coverage exceed 100%."
    )


def _embedding_animation_caption(*, dataset_name: str, frames: int) -> str:
    return (
        f"{dataset_name} embedding progression animation. Each frame is an epoch-level projection of collected token embeddings colored by label or dimension, depending on the animation builder. "
        f"Conclusion: use this video to judge whether geometry changes smoothly over epochs or whether clusters, gaps, and high-dimension regions appear abruptly. This run contains {frames} frame(s), so a one-frame video should be treated as a static diagnostic rather than a temporal trend."
    )


def _write_token_processing_notes(
    *,
    output_dir: Path,
    config: dict[str, Any],
    model: SamBackboneWrapper,
) -> None:
    """Write a compact note describing how pretrained COCO tokens are produced."""
    processor_size = int(getattr(model, "expected_image_size", config.get("img_size", 0) or 0))
    encoder_patch = int(getattr(model, "patch_size", 0) or 0)
    analysis_patch = int(config.get("analysis_patch_size", 0) or 0)
    embed_dim = int(getattr(model, "embed_dim", 0) or 0)
    payload = {
        "pipeline": "coco_pretrained_sam_vit",
        "model": config.get("sam_model"),
        "processor_image_size": processor_size,
        "encoder_patch_size": encoder_patch,
        "analysis_patch_size": analysis_patch,
        "embedding_dim": embed_dim,
        "token_source": "SAM ViT image encoder feature map pooled onto the requested analysis patch grid",
        "object_source": "COCO instance boxes are used as object prompts for SAM mask prediction",
        "sparse_probe": {
            "enabled": bool(config.get("sparse_probe")),
            "algorithm": config.get("sparse_probe_algorithm"),
            "residual_threshold": config.get("sparse_probe_residual_threshold"),
            "dictionary_size": config.get("sparse_probe_dictionary_size"),
            "max_sparsity": config.get("sparse_probe_max_sparsity"),
        },
    }
    with open(output_dir / "token_processing_summary.json", "w") as fp:
        json.dump(to_serializable(payload), fp, indent=2)
    with open(output_dir / "token_processing_notes.md", "w") as fp:
        fp.write("# Pretrained COCO Token Pipeline\n\n")
        fp.write(f"- Model: `{payload['model']}`\n")
        fp.write(f"- Processor image size: `{processor_size}`\n")
        fp.write(f"- ViT encoder patch size: `{encoder_patch}`\n")
        fp.write(f"- Analysis patch size: `{analysis_patch}`\n")
        fp.write(f"- Token dimension: `{embed_dim}`\n")
        fp.write("- Token generation: denormalized COCO images are processed by the SAM image processor, encoded by the frozen ViT image encoder, and the encoder feature map is adaptively pooled onto the analysis patch grid.\n")
        fp.write("- Object/segmentation signal: COCO instance boxes prompt SAM mask prediction; thresholded masks assign multi-hot object labels to patch-grid tokens.\n")
        fp.write("- Fiber analysis: local kNN volume scaling estimates token-level dimension and change-point irregularity.\n")
        fp.write("- Sparse probe: each eligible token neighborhood fits a local PCA dictionary over raw image patches and measures the sparsity needed to satisfy the configured residual target.\n")


@torch.no_grad()
def collect_sam_patch_tokens(
    *,
    model: SamBackboneWrapper,
    loader,
    device: torch.device,
    dataset: str,
    analysis_patch_size: int,
    num_classes: int,
    max_tokens: int,
    mask_threshold: float,
    max_boxes_per_image: int,
    mask_preview_images: int,
    multimask_output: bool,
    show_progress: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    base_ds = _unwrap_base_dataset(getattr(loader, "dataset", None))
    has_instance_boxes = hasattr(base_ds, "instances_after_eval_transform")

    embeddings = []
    labels = []
    images = []
    bboxes = []
    patch_indices = []
    image_ids = []
    pred_labels = []
    previews: list[dict[str, Any]] = []
    collected = 0

    iterator = tqdm(loader, desc="Collect SAM tokens", leave=False) if show_progress and tqdm else loader
    for batch in iterator:
        if max_tokens is not None and collected >= max_tokens:
            break
        imgs, batch_labels = batch[0].to(device), batch[1].cpu()
        batch_indices = batch[2] if len(batch) > 2 else range(len(batch_labels))

        for item_idx in range(int(imgs.shape[0])):
            if max_tokens is not None and collected >= max_tokens:
                break
            img = imgs[item_idx]
            label = batch_labels[item_idx]
            ds_idx_raw = batch_indices[item_idx]
            ds_idx = int(ds_idx_raw.item()) if isinstance(ds_idx_raw, torch.Tensor) else int(ds_idx_raw)
            image_buffer_idx = len(images)
            images.append(img.detach().cpu())

            inputs, img01 = model.prepare_single_image(img, dataset, device=device)
            embedding_map = model.get_image_embedding_map(inputs["pixel_values"])
            image_h, image_w = int(img01.shape[-2]), int(img01.shape[-1])
            grid_h = max(1, int(round(image_h / max(1, analysis_patch_size))))
            grid_w = max(1, int(round(image_w / max(1, analysis_patch_size))))
            pooled_tokens = _pool_embedding_map(embedding_map, grid_h=grid_h, grid_w=grid_w)
            patch_count = int(pooled_tokens.shape[0])
            grid_boxes = _grid_bboxes(image_h=image_h, image_w=image_w, grid_h=grid_h, grid_w=grid_w)

            prompt_boxes_raw: list[tuple[float, float, float, float, int]] = []
            if has_instance_boxes:
                try:
                    prompt_boxes_raw = _limit_boxes(
                        list(base_ds.instances_after_eval_transform(ds_idx, image_h)),
                        max_boxes=max_boxes_per_image,
                    )
                except Exception:
                    prompt_boxes_raw = []
            prompt_boxes = [[float(x0), float(y0), float(x1), float(y1)] for x0, y0, x1, y1, _cat in prompt_boxes_raw]
            prompt_cats = [int(cat) for *_coords, cat in prompt_boxes_raw]
            box_labels = _boxes_to_patch_multihot(
                prompt_boxes_raw,
                grid_h=grid_h,
                grid_w=grid_w,
                image_h=image_h,
                image_w=image_w,
                num_classes=num_classes,
            )
            fallback_labels = box_labels if box_labels.any() else _fallback_patch_labels(label, patch_count, num_classes)

            masks: list[torch.Tensor] = []
            if prompt_boxes:
                try:
                    masks = model.predict_masks_for_boxes(
                        img=img,
                        dataset_name=dataset,
                        boxes=prompt_boxes,
                        device=device,
                        image_embeddings=embedding_map,
                        multimask_output=multimask_output,
                    )
                except Exception as exc:
                    print(f"[sam] WARNING: mask prediction failed for image {ds_idx}: {exc}")

            patch_labels = _masks_to_patch_multihot(
                masks,
                prompt_cats,
                grid_h=grid_h,
                grid_w=grid_w,
                num_classes=num_classes,
                threshold=mask_threshold,
                fallback=fallback_labels,
            )
            image_pred = _image_level_pred_label(label)

            if masks and len(previews) < mask_preview_images:
                previews.append(
                    {
                        "image": img01.clone(),
                        "boxes": prompt_boxes_raw,
                        "masks": [mask.detach().cpu() for mask in masks],
                        "image_id": ds_idx,
                        "grid_h": grid_h,
                        "grid_w": grid_w,
                        "analysis_patch_size": int(analysis_patch_size),
                        "categories": prompt_cats,
                    }
                )

            for patch_id in range(patch_count):
                if max_tokens is not None and collected >= max_tokens:
                    break
                patch_label = patch_labels[patch_id]
                patch_pred = image_pred
                if patch_label.numel() == num_classes and float(patch_label.max().item()) > 0:
                    patch_pred = int(torch.argmax(patch_label).item())
                embeddings.append(pooled_tokens[patch_id])
                labels.append(patch_label.to(dtype=torch.float32))
                bboxes.append(grid_boxes[patch_id])
                patch_indices.append(torch.tensor(patch_id, dtype=torch.int32))
                image_ids.append(torch.tensor(image_buffer_idx, dtype=torch.int32))
                pred_labels.append(torch.tensor(patch_pred, dtype=torch.int64))
                collected += 1

    if not embeddings:
        empty_labels = torch.empty((0, num_classes), dtype=torch.float32)
        return (
            torch.empty((0, model.embed_dim), dtype=torch.float32),
            empty_labels,
            torch.empty((0, 3, 1, 1), dtype=torch.float32),
            torch.empty((0, 4), dtype=torch.int32),
            torch.empty((0,), dtype=torch.int32),
            torch.empty((0,), dtype=torch.int32),
            torch.empty((0,), dtype=torch.int64),
            previews,
        )

    return (
        torch.stack(embeddings, dim=0),
        torch.stack(labels, dim=0),
        torch.stack(images, dim=0),
        torch.stack(bboxes, dim=0),
        torch.stack(patch_indices, dim=0),
        torch.stack(image_ids, dim=0),
        torch.stack(pred_labels, dim=0),
        previews,
    )


def run_sam_fiber_job(
    *,
    dataset_name: str,
    root: str,
    img_size: int | None,
    batch_size_test: int,
    num_workers: int,
    subset_test: int | None,
    seed: int,
    output_dir: Path,
    checkpoints_dir: Path,
    embeddings_dir: Path,
    analysis_dir: Path,
    sam_cfg: SamFiberConfig,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_name: str,
    wandb_tags: Any,
) -> dict[str, Any]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _train_loader, test_loader, num_classes, _in_chans, final_img_size, _task = create_data_loaders(
        dataset_name,
        root,
        img_size,
        1,
        batch_size_test,
        num_workers,
        None,
        None,
        device,
        distributed=False,
        rank=0,
        world_size=1,
    )
    full_test_dataset = test_loader.dataset
    total_eval_images = len(full_test_dataset)
    sample_size = min(
        total_eval_images,
        int(subset_test) if subset_test is not None and int(subset_test) > 0 else total_eval_images,
    )
    class_names = get_class_names(full_test_dataset, dataset_name)
    model = SamBackboneWrapper(model_name=sam_cfg.model_name).to(device)

    config = {
        "dataset": dataset_name,
        "root": root,
        "img_size": final_img_size,
        "sam_model": sam_cfg.model_name,
        "epochs": int(sam_cfg.epochs),
        "resample_each_epoch": bool(sam_cfg.resample_each_epoch),
        "analysis_patch_size": sam_cfg.analysis_patch_size,
        "max_tokens": sam_cfg.max_tokens,
        "vol_min": sam_cfg.vol_min,
        "vol_max": sam_cfg.vol_max,
        "ws": sam_cfg.ws,
        "alpha": sam_cfg.alpha,
        "nstrat": sam_cfg.nstrat,
        "neighborhood_size": sam_cfg.neighborhood_size,
        "mask_threshold": sam_cfg.mask_threshold,
        "max_boxes_per_image": sam_cfg.max_boxes_per_image,
        "multimask_output": bool(sam_cfg.multimask_output),
        "sparse_probe": bool(sam_cfg.sparse_probe),
        "sparse_probe_radius": sam_cfg.sparse_probe_radius,
        "sparse_probe_auto_neighbor_k": sam_cfg.sparse_probe_auto_neighbor_k,
        "sparse_probe_auto_radius_quantile": sam_cfg.sparse_probe_auto_radius_quantile,
        "sparse_probe_min_patches": sam_cfg.sparse_probe_min_patches,
        "sparse_probe_max_anchors": sam_cfg.sparse_probe_max_anchors,
        "sparse_probe_dictionary_size": sam_cfg.sparse_probe_dictionary_size,
        "sparse_probe_residual_threshold": sam_cfg.sparse_probe_residual_threshold,
        "sparse_probe_max_sparsity": sam_cfg.sparse_probe_max_sparsity,
        "sparse_probe_algorithm": sam_cfg.sparse_probe_algorithm,
        "sparse_probe_iht_steps": sam_cfg.sparse_probe_iht_steps,
        "sparse_probe_iht_lr": sam_cfg.sparse_probe_iht_lr,
        "sparse_probe_heatmap_images": sam_cfg.sparse_probe_heatmap_images,
        "subset_test": subset_test,
        "batch_size_test": batch_size_test,
        "seed": seed,
        "device": str(device),
    }

    wandb = init_wandb_run(
        enabled=wandb_enabled,
        project=wandb_project,
        name=wandb_name,
        tags=list(wandb_tags) if wandb_tags is not None else None,
        config=config,
        missing_message="[wandb] ERROR: SAM fiber logging disabled; wandb is not installed",
        show_url=True,
    )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        _write_token_processing_notes(output_dir=output_dir, config=config, model=model)

        fiber_history: list[dict[str, Any]] = []
        collection_history: list[dict[str, Any]] = []
        embedding_animation_snapshots: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        mask_preview_paths: list[str] = []
        analysis = None

        epochs = max(1, int(sam_cfg.epochs))
        fixed_indices: list[int] | None = None
        for epoch in range(epochs):
            if sample_size < total_eval_images:
                if fixed_indices is None or bool(sam_cfg.resample_each_epoch):
                    rng = np.random.default_rng(int(seed) + epoch)
                    fixed_indices = rng.choice(total_eval_images, size=sample_size, replace=False).tolist()
                epoch_indices = list(fixed_indices)
            else:
                epoch_indices = None

            epoch_loader = _make_epoch_loader(
                full_test_dataset,
                batch_size=batch_size_test,
                num_workers=num_workers,
                indices=epoch_indices,
            )

            print(f"[sam_fiber] Epoch {epoch:03d}: collecting SAM tokens...", flush=True)
            (
                embeddings,
                labels,
                images,
                bboxes,
                patch_indices,
                image_ids,
                pred_labels,
                previews,
            ) = collect_sam_patch_tokens(
                model=model,
                loader=epoch_loader,
                device=device,
                dataset=dataset_name,
                analysis_patch_size=sam_cfg.analysis_patch_size,
                num_classes=num_classes,
                max_tokens=sam_cfg.max_tokens,
                mask_threshold=sam_cfg.mask_threshold,
                max_boxes_per_image=sam_cfg.max_boxes_per_image,
                mask_preview_images=sam_cfg.mask_preview_images,
                multimask_output=bool(sam_cfg.multimask_output),
                show_progress=sam_cfg.progress,
            )
            if embeddings.numel() == 0:
                raise RuntimeError("SAM probe collected no embeddings; check the dataset and max_tokens settings.")

            collection_history.append(
                {
                    "epoch": int(epoch),
                    "num_tokens": int(embeddings.shape[0]),
                    "num_images": int(torch.unique(image_ids).numel()),
                }
            )
            print(f"[sam_fiber] Epoch {epoch:03d}: running fiber analysis...", flush=True)
            analysis = analyze_fiber_epoch(
                epoch=epoch,
                embeddings=embeddings,
                labels=labels,
                images=images,
                bboxes=bboxes,
                patch_indices=patch_indices,
                image_ids=image_ids,
                pred_labels=pred_labels,
                num_classes=num_classes,
                class_names=class_names,
                dataset=dataset_name,
                base_dir=checkpoints_dir,
                analysis_dir=analysis_dir,
                embed_dir=embeddings_dir,
                vol_min=sam_cfg.vol_min,
                vol_max=sam_cfg.vol_max,
                ws=sam_cfg.ws,
                alpha=sam_cfg.alpha,
                nstrat=sam_cfg.nstrat,
                neighborhood_size=int(sam_cfg.neighborhood_size or (sam_cfg.analysis_patch_size + 1)),
                sparse_probe=bool(sam_cfg.sparse_probe),
                sparse_probe_radius=sam_cfg.sparse_probe_radius,
                sparse_probe_auto_neighbor_k=sam_cfg.sparse_probe_auto_neighbor_k,
                sparse_probe_auto_radius_quantile=sam_cfg.sparse_probe_auto_radius_quantile,
                sparse_probe_min_patches=sam_cfg.sparse_probe_min_patches,
                sparse_probe_max_anchors=sam_cfg.sparse_probe_max_anchors,
                sparse_probe_dictionary_size=sam_cfg.sparse_probe_dictionary_size,
                sparse_probe_residual_threshold=sam_cfg.sparse_probe_residual_threshold,
                sparse_probe_max_sparsity=sam_cfg.sparse_probe_max_sparsity,
                sparse_probe_algorithm=sam_cfg.sparse_probe_algorithm,
                sparse_probe_iht_steps=sam_cfg.sparse_probe_iht_steps,
                sparse_probe_iht_lr=sam_cfg.sparse_probe_iht_lr,
                sparse_probe_heatmap_images=sam_cfg.sparse_probe_heatmap_images,
                wandb_module=wandb,
            )
            fiber_history.append(analysis["fiber_summary"])
            if sam_cfg.embedding_animation:
                embedding_animation_snapshots.append(
                    (epoch, embeddings.detach().cpu().clone(), labels.detach().cpu().clone())
                )

            if previews:
                preview_rows = _preview_rows(
                    previews,
                    class_names=class_names,
                    mask_threshold=sam_cfg.mask_threshold,
                )
                epoch_preview_summary = _preview_epoch_summary(
                    previews,
                    preview_rows=preview_rows,
                    class_names=class_names,
                )
                overlay_images = []
                overlay_payload = []
                for preview, row in zip(previews, preview_rows):
                    overlay = _overlay_masks(
                        image=preview["image"],
                        masks=preview["masks"],
                        boxes=preview["boxes"],
                        grid_h=int(preview.get("grid_h", 0)),
                        grid_w=int(preview.get("grid_w", 0)),
                    )
                    if overlay is None:
                        continue
                    overlay_path = analysis_dir / f"epoch_{epoch:03d}_sam_mask_{int(preview['image_id']):05d}.png"
                    overlay.save(overlay_path)
                    overlay_images.append(overlay)
                    overlay_payload.append((overlay_path, row["description"]))
                if overlay_images:
                    grid = _make_image_grid(overlay_images, cols=min(3, len(overlay_images)))
                    if grid is not None:
                        grid_path = analysis_dir / f"epoch_{epoch:03d}_sam_box_masks.png"
                        grid.save(grid_path)
                        mask_preview_paths.append(grid_path.name)
                        if wandb is not None:
                            segmentation_caption = _segmentation_epoch_caption(
                                epoch=epoch,
                                summary=epoch_preview_summary,
                                model_name=sam_cfg.model_name,
                                dataset_name=dataset_name,
                            )
                            payload: dict[str, Any] = {
                                "epoch": epoch,
                                "media/epoch": epoch,
                                "media/log_phase": "segmentation_preview",
                                "media/media_index": epoch * 10 + 1,
                                "segmentation/caption": segmentation_caption,
                            }
                            payload["segmentation/box_prompt_mask_grid"] = _wandb_image_with_caption(
                                wandb,
                                str(grid_path),
                                segmentation_caption,
                            )
                            payload["segmentation/box_prompt_masks"] = [
                                _wandb_image_with_caption(
                                    wandb,
                                    str(path),
                                    f"{description} Conclusion: use this image to check whether the prompted object masks align with the patch grid before interpreting token-level labels.",
                                )
                                for path, description in overlay_payload
                            ]
                            if hasattr(wandb, "Table") and preview_rows:
                                columns = [
                                    "image_id",
                                    "num_prompts",
                                    "classes",
                                    "prompt_box_fraction",
                                    "union_mask_fraction",
                                    "largest_mask_fraction",
                                    "largest_mask_class",
                                    "per_class_mask_coverage",
                                    "coverage_regime",
                                    "description",
                                ]
                                payload["segmentation/summary_table"] = wandb.Table(
                                    columns=columns,
                                    data=[[row.get(column) for column in columns] for row in preview_rows],
                                )
                                payload["segmentation/epoch_summary_table"] = wandb.Table(
                                    columns=[
                                        "num_preview_images",
                                        "avg_prompts",
                                        "avg_union_mask_fraction",
                                        "avg_prompt_box_fraction",
                                        "top_classes",
                                        "description",
                                    ],
                                    data=[
                                        [
                                            epoch_preview_summary["num_preview_images"],
                                            epoch_preview_summary["avg_prompts"],
                                            epoch_preview_summary["avg_union_mask_fraction"],
                                            epoch_preview_summary["avg_prompt_box_fraction"],
                                            epoch_preview_summary["top_classes"],
                                            epoch_preview_summary["description"],
                                        ]
                                    ],
                                )
                            payload["segmentation/epoch_summary"] = epoch_preview_summary["description"]
                            wandb.log(payload)
                    with open(analysis_dir / f"epoch_{epoch:03d}_segmentation_summary.json", "w") as fp:
                        json.dump(
                            {
                                "epoch_summary": epoch_preview_summary,
                                "images": preview_rows,
                            },
                            fp,
                            indent=2,
                        )
                    with open(analysis_dir / f"epoch_{epoch:03d}_segmentation_notes.md", "w") as fp:
                        fp.write(f"# Epoch {epoch:03d} segmentation summary\n\n")
                        fp.write(f"{epoch_preview_summary['description']}\n\n")
                        for row in preview_rows:
                            fp.write(f"- {row['description']}\n")

        animation_path = None
        if sam_cfg.embedding_animation:
            frames = build_embedding_animation_frames(embedding_animation_snapshots)
            if frames:
                animation_path = generate_embedding_animation(
                    frames,
                    title=f"{dataset_name} SAM Embedding Space",
                    output_path=output_dir / "embedding_progression.gif",
                    fps=max(1, int(sam_cfg.embedding_animation_fps)),
                )
                if wandb is not None and hasattr(wandb, "Video"):
                    wandb.log(
                        {
                            "epoch": max(0, epochs - 1),
                            "media/epoch": max(0, epochs - 1),
                            "media/log_phase": "embedding_animation",
                            "media/media_index": max(0, epochs - 1) * 10 + 2,
                            "embeddings/progression": wandb.Video(str(animation_path), format="gif"),
                            "embeddings/progression_frames": len(frames),
                            "embeddings/progression_caption": _embedding_animation_caption(
                                dataset_name=dataset_name,
                                frames=len(frames),
                            ),
                        }
                    )

        results = {
            "config": config,
            "collection_history": collection_history,
            "fiber_history": fiber_history,
            "fiber_summary": analysis["fiber_summary"] if analysis is not None else {},
            "hypothesis_summary": analysis.get("hypothesis_summary", {}) if analysis is not None else {},
            "mask_preview_paths": mask_preview_paths,
            "embedding_animation": animation_path.name if animation_path is not None else None,
        }
        with open(output_dir / "sam_fiber_history.json", "w") as fp:
            json.dump(to_serializable(fiber_history), fp, indent=2)
        with open(output_dir / "sam_collection_history.json", "w") as fp:
            json.dump(to_serializable(collection_history), fp, indent=2)
        with open(output_dir / "sam_fiber_summary.json", "w") as fp:
            json.dump(to_serializable(results), fp, indent=2)
        return results
    finally:
        finish_wandb_run(wandb)
