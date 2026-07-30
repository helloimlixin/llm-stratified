"""Nearest-neighbor evidence sheets for VAR singular-token polysemy.

This script is intentionally artifact-driven: it does not reload the VAR model.
Given a cached ``var_generation_branch_samples`` JSON file and the matching
embedding pack, it shows where each singular branch token starts, retrieves its
nearest neighbors in token-embedding space, and optionally stacks the existing
decoded branch-sampling panel underneath the neighbor gallery.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from fiber.figure_io import save_figure  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def _default_branch_json(run_dir: Path, epoch: int) -> Path:
    return run_dir / "checkpoints" / "fiber_analysis" / f"epoch_{epoch:03d}_var_generation_polysemy_branch_samples.json"


def _default_embedding_pack(run_dir: Path, epoch: int) -> Path:
    return run_dir / "checkpoints" / "embeddings" / f"epoch_{epoch:03d}.pt"


def _denormalize_images(images: torch.Tensor, dataset: str) -> torch.Tensor:
    dataset_name = dataset.upper()
    mean_values = CIFAR_MEAN if dataset_name in {"CIFAR10", "CIFAR100", "SVHN"} else IMAGENET_MEAN
    std_values = CIFAR_STD if dataset_name in {"CIFAR10", "CIFAR100", "SVHN"} else IMAGENET_STD
    mean = torch.tensor(mean_values, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(std_values, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def _to_numpy_image(image_chw: torch.Tensor) -> np.ndarray:
    return image_chw.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def _image_index_for_token(token_index: int, patches_per_image: int) -> int:
    return int(token_index) // int(patches_per_image)


def _patch_id_for_token(token_index: int, patches_per_image: int) -> int:
    return int(token_index) % int(patches_per_image)


def _row_col_for_patch(patch_id: int, grid_size: int) -> tuple[int, int]:
    return int(patch_id) // int(grid_size), int(patch_id) % int(grid_size)


def _patch_bbox_from_row_col(image: np.ndarray, row: int, col: int, grid_size: int) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    x0 = int(round(col * width / grid_size))
    y0 = int(round(row * height / grid_size))
    x1 = int(round((col + 1) * width / grid_size))
    y1 = int(round((row + 1) * height / grid_size))
    return x0, y0, x1, y1


def _bbox_for_token(
    *,
    token_index: int,
    image: np.ndarray,
    bboxes: torch.Tensor | None,
    grid_size: int,
    patches_per_image: int,
) -> tuple[int, int, int, int]:
    if bboxes is not None and 0 <= int(token_index) < int(bboxes.shape[0]):
        x0, y0, x1, y1 = [int(v) for v in bboxes[int(token_index)].tolist()]
        if x1 > x0 and y1 > y0:
            return x0, y0, x1, y1
    patch_id = _patch_id_for_token(token_index, patches_per_image)
    row, col = _row_col_for_patch(patch_id, grid_size)
    return _patch_bbox_from_row_col(image, row, col, grid_size)


def _crop_token(
    image: np.ndarray,
    *,
    token_index: int,
    bboxes: torch.Tensor | None,
    grid_size: int,
    patches_per_image: int,
    pad_tokens: int = 1,
) -> np.ndarray:
    x0, y0, x1, y1 = _bbox_for_token(
        token_index=token_index,
        image=image,
        bboxes=bboxes,
        grid_size=grid_size,
        patches_per_image=patches_per_image,
    )
    patch_w = max(1, x1 - x0)
    patch_h = max(1, y1 - y0)
    height, width = image.shape[:2]
    x0 = max(0, x0 - pad_tokens * patch_w)
    y0 = max(0, y0 - pad_tokens * patch_h)
    x1 = min(width, x1 + pad_tokens * patch_w)
    y1 = min(height, y1 + pad_tokens * patch_h)
    return image[y0:y1, x0:x1]


def _draw_start_box(
    ax: plt.Axes,
    image: np.ndarray,
    *,
    token_index: int,
    bboxes: torch.Tensor | None,
    grid_size: int,
    patches_per_image: int,
    color: str = "#e45756",
) -> None:
    x0, y0, x1, y1 = _bbox_for_token(
        token_index=token_index,
        image=image,
        bboxes=bboxes,
        grid_size=grid_size,
        patches_per_image=patches_per_image,
    )
    ax.add_patch(
        mpatches.Rectangle(
            (x0, y0),
            max(1, x1 - x0),
            max(1, y1 - y0),
            fill=False,
            edgecolor=color,
            linewidth=2.2,
        )
    )
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    ax.plot([cx], [cy], marker="+", color=color, markersize=8, markeredgewidth=1.6)


def _select_singular_anchors(branch_data: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    anchors = [
        item
        for item in branch_data.get("anchors", [])
        if item.get("anchor", {}).get("group") == "singular"
    ]

    def score(item: dict[str, Any]) -> float:
        anchor = item.get("anchor", {})
        irregularity = float(anchor.get("irregularity") or 0.0)
        entropy = float(anchor.get("entropy_norm") or 0.0)
        diversity = float(item.get("mean_pairwise_crop_mse") or 0.0)
        return irregularity * max(entropy, 1e-6) * (1.0 + diversity)

    anchors.sort(key=score, reverse=True)
    return anchors[: max(0, int(limit))]


def _nearest_neighbors(
    normalized_embeddings: torch.Tensor,
    token_index: int,
    *,
    k: int,
    patches_per_image: int,
    cross_image_only: bool,
) -> list[tuple[int, float]]:
    token_index = int(token_index)
    sims = torch.mv(normalized_embeddings, normalized_embeddings[token_index])
    sims[token_index] = -float("inf")
    if cross_image_only:
        image_id = _image_index_for_token(token_index, patches_per_image)
        start = image_id * patches_per_image
        end = min(start + patches_per_image, int(sims.numel()))
        sims[start:end] = -float("inf")
    take = min(int(k), max(0, int(torch.isfinite(sims).sum().item())))
    if take == 0:
        return []
    values, indices = torch.topk(sims, k=take)
    return [(int(idx), float(val)) for idx, val in zip(indices.tolist(), values.tolist())]


def _branch_label(item: dict[str, Any], max_codes: int = 4) -> str:
    anchor = item["anchor"]
    branch_codes = item.get("branch_codes", [])[:max_codes]
    branch_probs = item.get("branch_probs", [])[:max_codes]
    branch = "\n".join(
        f"{int(code)}:{float(prob):.2f}"
        for code, prob in zip(branch_codes, branch_probs)
    )
    if not branch:
        branch = "n/a"
    return (
        f"img {int(anchor['image_id'])} patch {int(anchor['patch_id'])}\n"
        f"row {int(anchor['row'])}, col {int(anchor['col'])}\n"
        f"H {float(anchor.get('entropy_norm') or 0.0):.2f}  "
        f"irr {float(anchor.get('irregularity') or 0.0):.2f}\n"
        f"branches\n{branch}\n"
        f"visual div {float(item.get('mean_pairwise_crop_mse') or 0.0):.4f}"
    )


def _axes_off(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_neighbor_records(
    *,
    branch_data: dict[str, Any],
    embeddings: torch.Tensor,
    limit: int,
    neighbors: int,
    grid_size: int,
    cross_image_only: bool,
) -> list[dict[str, Any]]:
    patches_per_image = grid_size * grid_size
    norm = F.normalize(embeddings.detach().float().cpu(), dim=1)
    records: list[dict[str, Any]] = []
    for item in _select_singular_anchors(branch_data, limit):
        anchor = item["anchor"]
        token_index = int(anchor["token_index"])
        nn = _nearest_neighbors(
            norm,
            token_index,
            k=neighbors,
            patches_per_image=patches_per_image,
            cross_image_only=cross_image_only,
        )
        neighbor_records = []
        for neighbor_idx, cosine in nn:
            patch_id = _patch_id_for_token(neighbor_idx, patches_per_image)
            row, col = _row_col_for_patch(patch_id, grid_size)
            neighbor_records.append(
                {
                    "token_index": neighbor_idx,
                    "image_id": _image_index_for_token(neighbor_idx, patches_per_image),
                    "patch_id": patch_id,
                    "row": row,
                    "col": col,
                    "cosine": cosine,
                }
            )
        records.append(
            {
                "anchor": anchor,
                "target_code": item.get("target_code"),
                "branch_codes": item.get("branch_codes", []),
                "branch_probs": item.get("branch_probs", []),
                "branch_prob_entropy": item.get("branch_prob_entropy"),
                "mean_pairwise_crop_mse": item.get("mean_pairwise_crop_mse"),
                "neighbors": neighbor_records,
            }
        )
    return records


def render_neighbor_gallery(
    *,
    records: list[dict[str, Any]],
    images01: torch.Tensor,
    bboxes: torch.Tensor | None,
    out_path: Path,
    grid_size: int,
    neighbors: int,
    cross_image_only: bool,
) -> Path:
    patches_per_image = grid_size * grid_size
    if not records:
        raise ValueError("No singular anchor records were available for plotting.")

    nrows = len(records)
    n_neighbor_cols = min(neighbors, max(len(row["neighbors"]) for row in records))
    ncols = 3 + n_neighbor_cols
    width_ratios = [1.55, 1.15, 1.10] + [1.0] * n_neighbor_cols
    fig_width = max(12.0, 1.85 * ncols)
    fig_height = max(3.2, 2.25 * nrows + 1.0)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        gridspec_kw={"width_ratios": width_ratios},
    )
    fig.suptitle(
        "Singular Token Polysemy: Start Patch, Embedding Neighbors, and Branch Posterior",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    for row_idx, record in enumerate(records):
        anchor = record["anchor"]
        token_index = int(anchor["token_index"])
        image_idx = int(anchor["image_id"])
        image = _to_numpy_image(images01[image_idx])

        ax = axes[row_idx, 0]
        _axes_off(ax)
        ax.text(
            0.0,
            0.5,
            _branch_label(record),
            va="center",
            ha="left",
            fontsize=8.2,
            linespacing=1.25,
            transform=ax.transAxes,
            clip_on=True,
        )

        ax = axes[row_idx, 1]
        ax.imshow(image)
        _draw_start_box(
            ax,
            image,
            token_index=token_index,
            bboxes=bboxes,
            grid_size=grid_size,
            patches_per_image=patches_per_image,
        )
        _axes_off(ax)
        ax.set_title("context\nSTART", fontsize=9.5, color="#b22222")

        ax = axes[row_idx, 2]
        ax.imshow(
            _crop_token(
                image,
                token_index=token_index,
                bboxes=bboxes,
                grid_size=grid_size,
                patches_per_image=patches_per_image,
                pad_tokens=1,
            )
        )
        _axes_off(ax)
        ax.set_title(f"singular crop\ncode {record.get('target_code')}", fontsize=9.5)

        for col_offset in range(n_neighbor_cols):
            ax = axes[row_idx, 3 + col_offset]
            _axes_off(ax)
            if col_offset >= len(record["neighbors"]):
                continue
            neighbor = record["neighbors"][col_offset]
            neighbor_image = _to_numpy_image(images01[int(neighbor["image_id"])])
            ax.imshow(
                _crop_token(
                    neighbor_image,
                    token_index=int(neighbor["token_index"]),
                    bboxes=bboxes,
                    grid_size=grid_size,
                    patches_per_image=patches_per_image,
                    pad_tokens=1,
                )
            )
            ax.set_title(
                f"NN {col_offset + 1}\n"
                f"img {int(neighbor['image_id'])} p{int(neighbor['patch_id'])}\n"
                f"cos {float(neighbor['cosine']):.2f}",
                fontsize=8.8,
            )

    fig.text(
        0.5,
        0.012,
        "Neighbors are cosine-nearest d30 predicted fine-scale patch embeddings. The red box marks the singular next-scale patch where the branch intervention starts; branch code:prob entries come from the VAR scale-conditioned code posterior.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.035, right=0.99, top=0.93, bottom=0.055, wspace=0.35, hspace=0.55)
    return save_figure(fig, out_path, dpi=180, bbox_inches="tight")


def stack_evidence_sheet(
    *,
    gallery_path: Path,
    branch_figure_path: Path,
    out_path: Path,
    title: str,
) -> Path:
    gallery = Image.open(gallery_path).convert("RGB")
    branch = Image.open(branch_figure_path).convert("RGB")
    width = max(gallery.width, branch.width)
    margin = 48
    title_height = 96
    label_height = 56

    def scale_to_width(image: Image.Image, target_width: int) -> Image.Image:
        if image.width == target_width:
            return image
        target_height = max(1, int(round(image.height * target_width / image.width)))
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    gallery = scale_to_width(gallery, width)
    branch = scale_to_width(branch, width)
    canvas_height = title_height + gallery.height + label_height + branch.height + 3 * margin
    canvas = Image.new("RGB", (width + 2 * margin, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
        title_font = ImageFont.truetype(font_path, 34)
        label_font = ImageFont.truetype(font_path, 24)
    except OSError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    draw.text((margin, 28), title, fill=(25, 25, 25), font=title_font)
    y = title_height
    canvas.paste(gallery, (margin, y))
    y += gallery.height + margin
    draw.text(
        (margin, y),
        "Decoded branch samples after replacing the marked next-scale VQ code with likely alternatives",
        fill=(55, 55, 55),
        font=label_font,
    )
    y += label_height
    canvas.paste(branch, (margin, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def _json_float(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_float(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_float(v) for v in value]
    if isinstance(value, tuple):
        return [_json_float(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    branch_json = Path(args.branch_json).resolve() if args.branch_json else _default_branch_json(run_dir, args.epoch)
    embedding_pack = Path(args.embedding_pack).resolve() if args.embedding_pack else _default_embedding_pack(run_dir, args.epoch)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "checkpoints" / "fiber_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    gallery_path = out_dir / args.out_name

    with branch_json.open("r", encoding="utf-8") as handle:
        branch_data = json.load(handle)
    pack = torch.load(embedding_pack, map_location="cpu")
    images = pack["images"].detach().float().cpu()
    images01 = _denormalize_images(images, args.dataset)
    embeddings = pack["embeddings"].detach().float().cpu()
    bboxes = pack.get("bboxes")
    if bboxes is not None:
        bboxes = bboxes.detach().cpu()

    records = build_neighbor_records(
        branch_data=branch_data,
        embeddings=embeddings,
        limit=args.anchors,
        neighbors=args.neighbors,
        grid_size=args.grid_size,
        cross_image_only=args.cross_image_only,
    )
    render_neighbor_gallery(
        records=records,
        images01=images01,
        bboxes=bboxes,
        out_path=gallery_path,
        grid_size=args.grid_size,
        neighbors=args.neighbors,
        cross_image_only=args.cross_image_only,
    )

    branch_figure = Path(args.branch_figure).resolve() if args.branch_figure else None
    if branch_figure is None:
        raw_figure = branch_data.get("figure")
        branch_figure = Path(raw_figure).resolve() if raw_figure else None
    evidence_path = None
    if branch_figure is not None and branch_figure.exists() and not args.no_composite:
        evidence_path = gallery_path.with_name(gallery_path.stem + "_evidence_sheet.png")
        stack_evidence_sheet(
            gallery_path=gallery_path,
            branch_figure_path=branch_figure,
            out_path=evidence_path,
            title="VAR-d30 singular-token polysemy evidence sheet",
        )

    summary = {
        "run_dir": str(run_dir),
        "branch_json": str(branch_json),
        "embedding_pack": str(embedding_pack),
        "gallery": str(gallery_path),
        "evidence_sheet": str(evidence_path) if evidence_path else None,
        "branch_figure": str(branch_figure) if branch_figure else None,
        "num_anchors": len(records),
        "neighbors_per_anchor": args.neighbors,
        "cross_image_only": bool(args.cross_image_only),
        "mean_branch_entropy": float(
            np.nanmean([float(r.get("branch_prob_entropy") or np.nan) for r in records])
        ),
        "mean_pairwise_branch_crop_mse": float(
            np.nanmean([float(r.get("mean_pairwise_crop_mse") or np.nan) for r in records])
        ),
        "mean_top_neighbor_cosine": float(
            np.nanmean([
                float(r["neighbors"][0]["cosine"])
                for r in records
                if r.get("neighbors")
            ])
        ),
        "records": records,
    }
    summary_path = gallery_path.with_suffix(".json")
    summary_path.write_text(json.dumps(_json_float(summary), indent=2), encoding="utf-8")

    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "run_dir": str(run_dir),
                "branch_json": str(branch_json),
                "embedding_pack": str(embedding_pack),
                "anchors": args.anchors,
                "neighbors": args.neighbors,
                "grid_size": args.grid_size,
                "cross_image_only": args.cross_image_only,
            },
        )
        log_payload: dict[str, Any] = {
            "polysemy_nn/gallery": wandb.Image(str(gallery_path)),
            "polysemy_nn/mean_branch_entropy": summary["mean_branch_entropy"],
            "polysemy_nn/mean_pairwise_branch_crop_mse": summary["mean_pairwise_branch_crop_mse"],
            "polysemy_nn/mean_top_neighbor_cosine": summary["mean_top_neighbor_cosine"],
        }
        if evidence_path is not None:
            log_payload["polysemy_nn/evidence_sheet"] = wandb.Image(str(evidence_path))
        wandb.log(log_payload)
        wandb_run.finish()

    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--branch-json", type=str, default=None)
    parser.add_argument("--branch-figure", type=str, default=None)
    parser.add_argument("--embedding-pack", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--out-name", type=str, default="epoch_000_var_polysemy_nn_gallery.png")
    parser.add_argument("--dataset", type=str, default="COCO")
    parser.add_argument("--anchors", type=int, default=6)
    parser.add_argument("--neighbors", type=int, default=6)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--cross-image-only", action="store_true")
    parser.add_argument("--no-composite", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="stratified-manifold-learning")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default="var-d30-polysemy-nn-gallery")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = run(args)
    print(json.dumps(_json_float({k: v for k, v in summary.items() if k != "records"}), indent=2))


if __name__ == "__main__":
    main()
