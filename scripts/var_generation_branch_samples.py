"""Generate local VAR branches at singular and control patch tokens.

This is a causal-looking follow-up to ``var_generation_polysemy_probe.py``:
instead of only correlating fiber singularity with next-token entropy, it
intervenes on selected final-scale VQ codes, decodes the resulting images, and
visualizes the local branches. A real singularity -> polysemy signal should show
larger and more semantically diverse local branches for singular patches than
for matched low-irregularity controls.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import create_data_loaders  # noqa: E402
from fiber.figure_io import save_figure  # noqa: E402
from models import VarAutoregressiveImageEncoder  # noqa: E402


def _to_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    return tensor.permute(1, 2, 0).numpy()


def _crop_token_window(image: np.ndarray, row: int, col: int, grid_size: int, radius: int) -> np.ndarray:
    h, w = image.shape[:2]
    r0, r1 = max(0, row - radius), min(grid_size, row + radius + 1)
    c0, c1 = max(0, col - radius), min(grid_size, col + radius + 1)
    y0 = int(round(r0 * h / grid_size))
    y1 = int(round(r1 * h / grid_size))
    x0 = int(round(c0 * w / grid_size))
    x1 = int(round(c1 * w / grid_size))
    return image[y0:y1, x0:x1]


def _patch_bounds(image: np.ndarray, row: int, col: int, grid_size: int) -> tuple[float, float, float, float]:
    h, w = image.shape[:2]
    x0 = col * w / grid_size
    y0 = row * h / grid_size
    x1 = (col + 1) * w / grid_size
    y1 = (row + 1) * h / grid_size
    return x0, y0, x1, y1


def _pairwise_mse(crops: list[np.ndarray]) -> float:
    if len(crops) < 2:
        return float("nan")
    values = []
    for i in range(len(crops)):
        for j in range(i + 1, len(crops)):
            a = np.asarray(crops[i], dtype=np.float32)
            b = np.asarray(crops[j], dtype=np.float32)
            if a.shape != b.shape:
                continue
            values.append(float(np.mean((a - b) ** 2)))
    return float(np.mean(values)) if values else float("nan")


def _probability_entropy(probs: list[float]) -> float:
    values = np.asarray(probs, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size < 2:
        return float("nan")
    values = values / float(values.sum())
    return float(-(values * np.log(values)).sum() / math.log(values.size))


def _select_anchors(records: list[dict], *, singular_count: int, anchor_score: str) -> list[dict]:
    singular_candidates = [
        rec for rec in records
        if float(rec.get("irregularity") or 0.0) > 0.0 and rec.get("p_violation") is not None
    ]

    def _score(rec: dict) -> float:
        irregularity = float(rec.get("irregularity") or 0.0)
        entropy = float(rec.get("entropy_norm") or 0.0)
        if anchor_score == "irregularity_entropy":
            return irregularity * max(entropy, 1e-6)
        return irregularity

    singular_candidates.sort(key=_score, reverse=True)
    singular = singular_candidates[:singular_count]

    used = {int(rec["token_index"]) for rec in singular}
    anchors: list[dict] = []
    for rec in singular:
        item = dict(rec)
        item["group"] = "singular"
        anchors.append(item)

        image_id = int(rec["image_id"])
        target_entropy = float(rec.get("entropy_norm") or 0.0)
        controls = [
            candidate for candidate in records
            if int(candidate["image_id"]) == image_id
            and int(candidate["token_index"]) not in used
            and float(candidate.get("irregularity") or 0.0) <= 1e-12
        ]
        if not controls:
            controls = [
                candidate for candidate in records
                if int(candidate["token_index"]) not in used
                and float(candidate.get("irregularity") or 0.0) <= 1e-12
            ]
        controls.sort(
            key=lambda candidate: (
                abs(float(candidate.get("entropy_norm") or 0.0) - target_entropy),
                abs(int(candidate["patch_id"]) - int(rec["patch_id"])),
            )
        )
        if controls:
            control = dict(controls[0])
            used.add(int(control["token_index"]))
            control["group"] = "control"
            control["matched_to"] = int(rec["token_index"])
            anchors.append(control)
    return anchors


def _paired_results(results: list[dict]) -> list[tuple[dict, dict]]:
    singular = {
        int(result["anchor"]["token_index"]): result
        for result in results
        if result["anchor"].get("group") == "singular"
    }
    pairs: list[tuple[dict, dict]] = []
    for result in results:
        anchor = result["anchor"]
        if anchor.get("group") != "control":
            continue
        matched_to = anchor.get("matched_to")
        if matched_to is None:
            continue
        singular_result = singular.get(int(matched_to))
        if singular_result is not None:
            pairs.append((singular_result, result))
    pairs.sort(
        key=lambda pair: (
            -float(pair[0]["anchor"].get("irregularity") or 0.0),
            int(pair[0]["anchor"].get("token_index") or 0),
        )
    )
    return pairs


def _paired_metric_arrays(
    pairs: list[tuple[dict, dict]],
    key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    singular = np.asarray([float(pair[0][key]) for pair in pairs], dtype=np.float64)
    control = np.asarray([float(pair[1][key]) for pair in pairs], dtype=np.float64)
    mask = np.isfinite(singular) & np.isfinite(control)
    singular = singular[mask]
    control = control[mask]
    return singular, control, singular - control


def _sign_test_pvalue(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=np.float64)
    diffs = diffs[np.isfinite(diffs) & (np.abs(diffs) > 1e-12)]
    n = int(diffs.size)
    if n == 0:
        return float("nan")
    positives = int(np.sum(diffs > 0.0))
    tail = min(positives, n - positives)
    prob = sum(math.comb(n, k) for k in range(tail + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * prob))


def _bootstrap_ci(values: np.ndarray, *, rng_seed: int = 1337, reps: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(rng_seed)
    samples = rng.choice(values, size=(int(reps), int(values.size)), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_metric_summary(pairs: list[tuple[dict, dict]], key: str, prefix: str) -> dict[str, float]:
    singular, control, diffs = _paired_metric_arrays(pairs, key)
    n = int(diffs.size)
    ci_low, ci_high = _bootstrap_ci(diffs)
    std = float(np.std(diffs, ddof=1)) if n > 1 else float("nan")
    mean_diff = float(np.mean(diffs)) if n else float("nan")
    control_mean = float(np.mean(control)) if n else float("nan")
    singular_mean = float(np.mean(singular)) if n else float("nan")
    return {
        f"{prefix}_paired_count": float(n),
        f"{prefix}_singular_mean": singular_mean,
        f"{prefix}_control_mean": control_mean,
        f"{prefix}_mean_diff": mean_diff,
        f"{prefix}_median_diff": float(np.median(diffs)) if n else float("nan"),
        f"{prefix}_mean_ratio": (
            float(singular_mean / control_mean)
            if math.isfinite(singular_mean) and math.isfinite(control_mean) and control_mean > 0.0
            else float("nan")
        ),
        f"{prefix}_positive_fraction": float(np.mean(diffs > 0.0)) if n else float("nan"),
        f"{prefix}_sign_test_p": _sign_test_pvalue(diffs),
        f"{prefix}_mean_diff_ci_low": ci_low,
        f"{prefix}_mean_diff_ci_high": ci_high,
        f"{prefix}_paired_cohen_dz": (
            float(mean_diff / std) if math.isfinite(mean_diff) and math.isfinite(std) and std > 0.0 else float("nan")
        ),
    }


def _load_cached_records(run_dir: Path, epoch: int) -> tuple[dict, list[dict]]:
    path = run_dir / "checkpoints" / "fiber_analysis" / f"epoch_{epoch:03d}_var_generation_polysemy.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/var_generation_polysemy_probe.py first.")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return dict(payload.get("summary", {})), list(payload.get("tokens", []))


def _load_images_and_model(
    *,
    dataset: str,
    data_root: str,
    img_size: int,
    subset_test: int,
    max_image_id: int,
    model_name: str,
    device: torch.device,
) -> tuple[VarAutoregressiveImageEncoder, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    _, test_loader, _, _, _, _ = create_data_loaders(
        dataset_name=dataset,
        root=data_root,
        img_size=img_size,
        batch_size_train=1,
        batch_size_test=1,
        num_workers=0,
        subset_train=1,
        subset_test=max(subset_test, max_image_id + 1),
        device=device,
    )
    model = VarAutoregressiveImageEncoder(model_name=model_name).to(device).eval()
    images: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for image_id, batch in enumerate(test_loader):
        if image_id > max_image_id:
            break
        imgs = batch[0].to(device, non_blocking=True)
        pixel_values, imgs01 = model.prepare_images_for_features(imgs, dataset)
        images[image_id] = (pixel_values.to(device), imgs01.to(device))
    return model, images


def _top_branch_codes(
    logits: torch.Tensor,
    *,
    target_code: int,
    num_samples: int,
    top_k: int,
    temperature: float,
) -> tuple[list[int], list[float]]:
    probs = torch.softmax(logits.float() / max(float(temperature), 1e-6), dim=-1)
    k = min(int(top_k), int(probs.numel()))
    top_probs, top_codes = torch.topk(probs, k=k)
    codes: list[int] = []
    code_probs: list[float] = []
    for prob, code in zip(top_probs.detach().cpu().tolist(), top_codes.detach().cpu().tolist()):
        code_i = int(code)
        if code_i == int(target_code):
            continue
        codes.append(code_i)
        code_probs.append(float(prob))
        if len(codes) >= num_samples:
            break
    return codes, code_probs


def _decode_anchor_branches(
    *,
    model: VarAutoregressiveImageEncoder,
    pack: dict[str, torch.Tensor],
    idx_bl: list[torch.Tensor],
    original_rec: torch.Tensor,
    anchor: dict,
    grid_size: int,
    num_samples: int,
    top_k: int,
    temperature: float,
    zoom_radius: int,
) -> dict:
    image_id = int(anchor["image_id"])
    patch_id = int(anchor["patch_id"])
    row, col = int(anchor["row"]), int(anchor["col"])
    with torch.no_grad():
        logits = pack["logits"][0, patch_id]
        target_code = int(idx_bl[-1][0, patch_id].item())
        codes, code_probs = _top_branch_codes(
            logits,
            target_code=target_code,
            num_samples=num_samples,
            top_k=top_k,
            temperature=temperature,
        )
        if not codes:
            raise RuntimeError(f"No alternative codes found for token {anchor['token_index']}")
        base = [level.repeat(len(codes), 1).clone() for level in idx_bl]
        for sample_idx, code in enumerate(codes):
            base[-1][sample_idx, patch_id] = int(code)
        variants = model.vae.idxBl_to_img(base, same_shape=True, last_one=True).add(1.0).mul(0.5)

    original_image = _to_image(original_rec[0])
    variant_images = [_to_image(variants[i]) for i in range(variants.shape[0])]
    original_crop = _crop_token_window(original_image, row, col, grid_size, zoom_radius)
    variant_crops = [_crop_token_window(image, row, col, grid_size, zoom_radius) for image in variant_images]
    from_original = [
        float(np.mean((crop.astype(np.float32) - original_crop.astype(np.float32)) ** 2))
        for crop in variant_crops
        if crop.shape == original_crop.shape
    ]
    return {
        "anchor": anchor,
        "image_id": image_id,
        "patch_id": patch_id,
        "row": row,
        "col": col,
        "target_code": target_code,
        "branch_codes": codes,
        "branch_probs": code_probs,
        "original_image": original_image,
        "original_crop": original_crop,
        "variant_images": variant_images,
        "variant_crops": variant_crops,
        "mean_crop_mse_from_original": float(np.mean(from_original)) if from_original else float("nan"),
        "mean_pairwise_crop_mse": _pairwise_mse(variant_crops),
        "unique_branch_codes": int(len(set(codes))),
        "branch_prob_entropy": _probability_entropy(code_probs),
    }


def _plot_branches(results: list[dict], *, out_path: Path, grid_size: int) -> None:
    if not results:
        raise ValueError("No branch results to plot")
    num_samples = max(len(result["variant_crops"]) for result in results)
    cols = 3 + num_samples
    rows = len(results)
    fig, axes = plt.subplots(rows, cols, figsize=(1.9 * cols + 1.6, 2.35 * rows + 1.7), squeeze=False)
    for r, result in enumerate(results):
        anchor = result["anchor"]
        group = str(anchor["group"])
        label_ax = axes[r, 0]
        label_ax.set_axis_off()
        label = (
            f"{group}\n"
            f"img {result['image_id']} patch {result['patch_id']}\n"
            f"I {float(anchor.get('irregularity') or 0.0):.2f}  H {float(anchor.get('entropy_norm') or 0.0):.2f}\n"
            f"d {float(anchor.get('dimension') or 0.0):.1f}  NLL {float(anchor.get('nll') or 0.0):.1f}\n"
            f"div {result['mean_pairwise_crop_mse']:.4f}"
        )
        label_ax.text(1.0, 0.5, label, ha="right", va="center", fontsize=11)

        full_ax = axes[r, 1]
        full_ax.imshow(result["original_image"])
        x0, y0, x1, y1 = _patch_bounds(result["original_image"], result["row"], result["col"], grid_size)
        full_ax.add_patch(
            mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=1.4)
        )
        full_ax.set_axis_off()
        full_ax.set_title("context" if r == 0 else "", fontsize=11)

        target_ax = axes[r, 2]
        target_ax.imshow(result["original_crop"], interpolation="nearest")
        target_ax.set_axis_off()
        target_ax.set_title("target crop" if r == 0 else "", fontsize=11)

        for c in range(num_samples):
            ax = axes[r, c + 3]
            ax.set_axis_off()
            if c >= len(result["variant_crops"]):
                continue
            ax.imshow(result["variant_crops"][c], interpolation="nearest")
            if r == 0:
                ax.set_title(f"branch {c + 1}", fontsize=11)
            prob = result["branch_probs"][c]
            ax.text(
                0.02,
                0.98,
                f"p={prob:.2f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9.5,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1, "edgecolor": "none"},
            )

    fig.suptitle("VAR Local Generation Branches at Singular and Matched Control Tokens", fontsize=20, y=0.985)
    fig.text(
        0.02,
        0.012,
        "Each row replaces one final-scale VQ code with likely alternative codes from VAR's next-token distribution, "
        "then decodes the image. Higher visual spread in singular rows would support the singularity -> polysemy hypothesis.",
        fontsize=11,
        color="#333333",
    )
    fig.subplots_adjust(left=0.035, right=0.995, top=0.86, bottom=0.095, wspace=0.10, hspace=0.36)
    save_figure(fig, out_path, dpi=180)
    plt.close(fig)


def _plot_branch_aggregate(results: list[dict], *, out_path: Path) -> None:
    pairs = _paired_results(results)
    if not pairs:
        raise ValueError("No matched singular/control pairs to plot")

    metrics = [
        ("mean_pairwise_crop_mse", "pairwise branch diversity"),
        ("mean_crop_mse_from_original", "change from original"),
        ("branch_prob_entropy", "top-branch entropy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.4), squeeze=False)
    axes_flat = axes.ravel()
    irregularity = np.asarray([float(pair[0]["anchor"].get("irregularity") or 0.0) for pair in pairs])
    cmap = plt.get_cmap("magma")
    if irregularity.size:
        lo = float(np.nanmin(irregularity))
        hi = float(np.nanmax(irregularity))
    else:
        lo, hi = 0.0, 1.0
    if math.isclose(lo, hi):
        hi = lo + 1.0

    for ax, (key, label) in zip(axes_flat, metrics):
        singular, control, diffs = _paired_metric_arrays(pairs, key)
        pair_count = int(diffs.size)
        if pair_count == 0:
            ax.set_axis_off()
            continue
        for idx, (s_val, c_val) in enumerate(zip(singular, control)):
            color = cmap((float(irregularity[idx]) - lo) / (hi - lo))
            ax.plot([0, 1], [c_val, s_val], color=color, alpha=0.42, linewidth=1.3)
            ax.scatter([0, 1], [c_val, s_val], color=[color, color], s=28, alpha=0.92, zorder=3)
        ax.scatter([0, 1], [float(np.mean(control)), float(np.mean(singular))], color="black", s=72, zorder=5)
        ci_low, ci_high = _bootstrap_ci(diffs)
        p_value = _sign_test_pvalue(diffs)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["matched\ncontrol", "fiber\nviolation"], fontsize=12)
        ax.set_title(
            f"{label}\nmean diff {float(np.mean(diffs)):.4f} [{ci_low:.4f}, {ci_high:.4f}], sign p={p_value:.3g}",
            fontsize=12,
            pad=10,
        )
        ax.tick_params(labelsize=11)
        ax.grid(True, axis="y", alpha=0.28, linewidth=0.7)
    fig.suptitle("Matched VAR Branch Diversity: Fiber Violations vs Controls", fontsize=18, y=0.985)
    fig.text(
        0.02,
        0.020,
        "Each line is one matched pair. Controls are non-violating tokens from the same image when possible, "
        "chosen to have similar next-token entropy. Black dots mark group means.",
        fontsize=11,
        color="#333333",
    )
    fig.subplots_adjust(left=0.055, right=0.990, top=0.790, bottom=0.185, wspace=0.30)
    save_figure(fig, out_path, dpi=180)
    plt.close(fig)


def _summarize(results: list[dict]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for group in ("singular", "control"):
        group_results = [result for result in results if result["anchor"].get("group") == group]
        for key in ("mean_crop_mse_from_original", "mean_pairwise_crop_mse"):
            values = np.asarray([float(result[key]) for result in group_results], dtype=np.float64)
            values = values[np.isfinite(values)]
            summary[f"{group}_{key}"] = float(values.mean()) if values.size else float("nan")
        summary[f"{group}_count"] = float(len(group_results))
    denom = summary.get("control_mean_pairwise_crop_mse", float("nan"))
    numer = summary.get("singular_mean_pairwise_crop_mse", float("nan"))
    summary["singular_control_pairwise_diversity_ratio"] = (
        float(numer / denom) if math.isfinite(numer) and math.isfinite(denom) and denom > 0 else float("nan")
    )
    pairs = _paired_results(results)
    summary["matched_pair_count"] = float(len(pairs))
    summary.update(_paired_metric_summary(pairs, "mean_pairwise_crop_mse", "paired_pairwise_diversity"))
    summary.update(_paired_metric_summary(pairs, "mean_crop_mse_from_original", "paired_original_mse"))
    summary.update(_paired_metric_summary(pairs, "branch_prob_entropy", "paired_branch_entropy"))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "runs/local/coco_var_d30_sparse_fiber/20260506_232532",
    )
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--singular-count", type=int, default=4)
    parser.add_argument(
        "--anchor-score",
        choices=["irregularity", "irregularity_entropy"],
        default="irregularity_entropy",
        help="How to rank singular anchors before matching controls.",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--zoom-radius", type=int, default=1)
    parser.add_argument(
        "--max-plot-pairs",
        type=int,
        default=4,
        help="Number of matched pairs to show in the visual branch-sample panel. All pairs still enter summaries.",
    )
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--subset-test", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    cached_summary, records = _load_cached_records(run_dir, args.epoch)
    anchors = _select_anchors(
        records,
        singular_count=int(args.singular_count),
        anchor_score=str(args.anchor_score),
    )
    if not anchors:
        raise RuntimeError("No singular anchors were found in the cached polysemy JSON.")

    cfg_path = run_dir / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(cfg_path) if cfg_path.exists() else OmegaConf.create({})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    wandb_cfg = cfg.get("wandb", {})
    dataset = args.dataset or data_cfg.get("name", cached_summary.get("dataset", "COCO"))
    data_root = args.data_root or data_cfg.get("root", str(REPO_ROOT.parent / "data"))
    img_size = int(args.img_size or data_cfg.get("img_size", 256))
    subset_test = int(args.subset_test or data_cfg.get("subset_test", 64))
    model_name = args.model_name or model_cfg.get("frozen_backbone_model", cached_summary.get("model_name", "var_d30"))
    device = torch.device(args.device)

    max_image_id = max(int(anchor["image_id"]) for anchor in anchors)
    model, image_cache = _load_images_and_model(
        dataset=dataset,
        data_root=data_root,
        img_size=img_size,
        subset_test=subset_test,
        max_image_id=max_image_id,
        model_name=model_name,
        device=device,
    )

    state_cache: dict[int, tuple[dict[str, torch.Tensor], list[torch.Tensor], torch.Tensor]] = {}
    results: list[dict] = []
    for anchor in anchors:
        image_id = int(anchor["image_id"])
        if image_id not in state_cache:
            pixel_values, _imgs01 = image_cache[image_id]
            with torch.no_grad():
                pack = model.forward_generation_pack(pixel_values)
                idx_bl = model.vae.img_to_idxBl(pixel_values, v_patch_nums=model.patch_nums)
                original_rec = model.vae.idxBl_to_img(idx_bl, same_shape=True, last_one=True).add(1.0).mul(0.5)
            state_cache[image_id] = (pack, idx_bl, original_rec)
        pack, idx_bl, original_rec = state_cache[image_id]
        result = _decode_anchor_branches(
            model=model,
            pack=pack,
            idx_bl=idx_bl,
            original_rec=original_rec,
            anchor=anchor,
            grid_size=int(args.grid_size),
            num_samples=int(args.num_samples),
            top_k=int(args.top_k),
            temperature=float(args.temperature),
            zoom_radius=int(args.zoom_radius),
        )
        results.append(result)

    analysis_dir = run_dir / "checkpoints" / "fiber_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"epoch_{args.epoch:03d}_var_generation_polysemy_branch_samples"
    fig_path = analysis_dir / f"{prefix}.png"
    aggregate_fig_path = analysis_dir / f"{prefix}_aggregate.png"
    json_path = analysis_dir / f"{prefix}.json"
    sample_pairs = _paired_results(results)[: max(1, int(args.max_plot_pairs))]
    sample_results = [item for pair in sample_pairs for item in pair]
    _plot_branches(sample_results or results, out_path=fig_path, grid_size=int(args.grid_size))
    _plot_branch_aggregate(results, out_path=aggregate_fig_path)

    compact_results = []
    for result in results:
        compact_results.append(
            {
                "anchor": result["anchor"],
                "target_code": int(result["target_code"]),
                "branch_codes": [int(code) for code in result["branch_codes"]],
                "branch_probs": [float(prob) for prob in result["branch_probs"]],
                "mean_crop_mse_from_original": float(result["mean_crop_mse_from_original"]),
                "mean_pairwise_crop_mse": float(result["mean_pairwise_crop_mse"]),
                "unique_branch_codes": int(result["unique_branch_codes"]),
                "branch_prob_entropy": float(result["branch_prob_entropy"]),
            }
        )
    payload = {
        "summary": {
            "run_dir": str(run_dir),
            "epoch": int(args.epoch),
            "dataset": str(dataset),
            "model_name": str(model_name),
            "num_anchors": int(len(results)),
            "num_samples": int(args.num_samples),
            "top_k": int(args.top_k),
            "temperature": float(args.temperature),
            "zoom_radius": int(args.zoom_radius),
            "anchor_score": str(args.anchor_score),
            **_summarize(results),
        },
        "figure": str(fig_path),
        "aggregate_figure": str(aggregate_fig_path),
        "anchors": compact_results,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if args.wandb:
        import wandb

        project = args.wandb_project or wandb_cfg.get("project", "stratified-manifold-learning")
        run_id = args.wandb_run_id
        if run_id is None:
            wandb_dir = run_dir / "wandb" / "wandb"
            matches = sorted(wandb_dir.glob("run-*-*")) if wandb_dir.exists() else []
            if matches:
                run_id = matches[-1].name.rsplit("-", 1)[-1]
        run = wandb.init(project=project, id=run_id, resume="allow", dir=str(run_dir / "wandb"))
        wandb.log(
            {
                "generation_polysemy/branch_samples": wandb.Image(str(fig_path)),
                "generation_polysemy/branch_aggregate": wandb.Image(str(aggregate_fig_path)),
                "generation_polysemy/branch_sample_summary": payload["summary"],
            }
        )
        run.finish()

    print(
        json.dumps(
            {
                "summary": payload["summary"],
                "figure": str(fig_path),
                "aggregate_figure": str(aggregate_fig_path),
                "json": str(json_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
