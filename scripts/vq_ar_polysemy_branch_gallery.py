"""Visual branch galleries for singular VQ-AR patch tokens.

This script turns the codebook-first AR statistics into visual evidence.  For
selected patch positions, it takes the model's top next-token branch codes,
decodes the resulting VQ images, and optionally regenerates the suffix after
the branch token with the pretrained LlamaGen AR model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pretrained_vq_ar_ks_probe import load_class_labels_file, load_llamagen_gpt  # noqa: E402
from pretrained_vq_ar_pipeline import (  # noqa: E402
    LLAMAGEN_PROFILES,
    llamagen_import_context,
    load_weight_payload,
    resolve_device,
    resolve_llamagen_repo,
)


def load_tokens(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        payload = payload.get("tokens", payload.get("index_sample"))
    if not isinstance(payload, torch.Tensor):
        raise ValueError(f"{path} does not contain a token tensor")
    return payload.long().cpu()


def finite_float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        value = default
    return value if math.isfinite(value) else default


def zscores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros(values.shape, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    mean = float(values[finite].mean())
    std = float(values[finite].std())
    if std <= 1e-12 or not math.isfinite(std):
        return out
    out[finite] = (values[finite] - mean) / std
    return out


def score_records(records: list[dict[str, Any]], *, mode: str = "flatness", seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if mode == "position":
        order = -np.arange(len(records), dtype=np.float64)
        return order, order.copy()
    if mode == "random":
        rng = np.random.default_rng(int(seed))
        scores = rng.random(len(records))
        return scores, scores.copy()
    if mode == "geometry":
        delta = np.asarray([finite_float(row, "paper_large_fiber_delta", 0.0) for row in records], dtype=np.float64)
        return delta, -delta
    if mode != "flatness":
        raise ValueError(f"unknown score mode {mode!r}")
    local_ks = np.asarray([finite_float(row, "local_ball_ks") for row in records])
    local_entropy = np.asarray([finite_float(row, "local_ball_entropy") for row in records])
    branch_ks = np.asarray([finite_float(row, "branch_ks") for row in records])
    branch_entropy = np.asarray([finite_float(row, "branch_entropy") for row in records])
    singular_score = -zscores(local_ks) + zscores(local_entropy) - zscores(branch_ks) + zscores(branch_entropy)
    regular_score = zscores(local_ks) - zscores(local_entropy) + zscores(branch_ks) - zscores(branch_entropy)
    return singular_score, regular_score


def is_candidate(row: dict[str, Any], *, selector: str, want_singular: bool, min_patch_id: int, max_patch_id: int) -> bool:
    patch_id = int(row.get("patch_id", -1))
    if patch_id < int(min_patch_id) or patch_id > int(max_patch_id):
        return False
    if bool(row.get(selector, False)) != bool(want_singular):
        return False
    codes = row.get("top_branch_codes") or []
    return isinstance(codes, list) and len(codes) > 0


def choose_anchor_pairs(
    records: list[dict[str, Any]],
    *,
    selector: str,
    pairs: int,
    min_patch_id: int,
    max_patch_id: int,
    score_mode: str = "flatness",
    seed: int = 0,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    singular_score, regular_score = score_records(records, mode=score_mode, seed=seed)
    singular_idx = [
        idx
        for idx, row in enumerate(records)
        if is_candidate(row, selector=selector, want_singular=True, min_patch_id=min_patch_id, max_patch_id=max_patch_id)
    ]
    singular_idx.sort(key=lambda idx: float(singular_score[idx]), reverse=True)
    by_sample: dict[int, list[int]] = {}
    for idx, row in enumerate(records):
        if is_candidate(row, selector=selector, want_singular=False, min_patch_id=min_patch_id, max_patch_id=max_patch_id):
            by_sample.setdefault(int(row["sample_id"]), []).append(idx)
    for values in by_sample.values():
        values.sort(key=lambda idx: float(regular_score[idx]), reverse=True)

    pairs_out = []
    used_samples: set[int] = set()
    used_regular: set[int] = set()
    for s_idx in singular_idx:
        sample_id = int(records[s_idx]["sample_id"])
        if sample_id in used_samples:
            continue
        candidates = [idx for idx in by_sample.get(sample_id, []) if idx not in used_regular]
        if not candidates:
            continue
        r_idx = candidates[0]
        pairs_out.append((records[s_idx], records[r_idx]))
        used_samples.add(sample_id)
        used_regular.add(r_idx)
        if len(pairs_out) >= int(pairs):
            break
    if len(pairs_out) < int(pairs):
        used_s = {int(row["token_index"]) for pair in pairs_out for row in pair}
        for s_idx in singular_idx:
            if int(records[s_idx]["token_index"]) in used_s:
                continue
            global_regular = [
                idx
                for values in by_sample.values()
                for idx in values
                if idx not in used_regular
            ]
            if not global_regular:
                break
            global_regular.sort(key=lambda idx: float(regular_score[idx]), reverse=True)
            r_idx = global_regular[0]
            pairs_out.append((records[s_idx], records[r_idx]))
            used_regular.add(r_idx)
            if len(pairs_out) >= int(pairs):
                break
    return pairs_out


def branch_codes(row: dict[str, Any], *, branches: int) -> list[int]:
    seen = set()
    out = []
    for code in row.get("top_branch_codes") or []:
        code_i = int(code)
        if code_i in seen:
            continue
        seen.add(code_i)
        out.append(code_i)
        if len(out) >= int(branches):
            break
    return out


def tensor_to_images(images: torch.Tensor) -> list[Image.Image]:
    x = images.detach().float().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).round().clamp(0, 255).byte()
    x = x.permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(arr, mode="RGB") for arr in x]


@torch.no_grad()
def load_vq_model(profile: dict[str, Any], repo_path: Path, device: torch.device):
    vq_path = hf_hub_download(repo_id=profile["repo_id"], filename=profile["vq_file"])
    with llamagen_import_context(repo_path):
        from tokenizer.tokenizer_image.vq_model import VQ_models

        vq_model = VQ_models[profile["vq_model"]](
            codebook_size=int(profile["codebook_size"]),
            codebook_embed_dim=int(profile["codebook_embed_dim"]),
        ).to(device)
        vq_model.load_state_dict(load_weight_payload(vq_path), strict=True)
        vq_model.eval()
    return vq_model


@torch.no_grad()
def decode_token_batch(vq_model, tokens: torch.Tensor, profile: dict[str, Any], device: torch.device) -> list[Image.Image]:
    latent_size = int(profile["image_size"]) // int(profile["downsample_size"])
    qzshape = [tokens.shape[0], int(profile["codebook_embed_dim"]), latent_size, latent_size]
    images = vq_model.decode_code(tokens.to(device=device, dtype=torch.long), qzshape)
    return tensor_to_images(images)


def filter_logits(logits: torch.Tensor, *, top_k: int, top_p: float) -> torch.Tensor:
    if int(top_k) > 0:
        kth = torch.topk(logits, k=min(int(top_k), logits.shape[-1]), dim=-1).values[..., -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, -float("inf")), logits)
    if float(top_p) < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove_unsorted = torch.zeros_like(remove).scatter(-1, sorted_idx, remove)
        logits = torch.where(remove_unsorted, torch.full_like(logits, -float("inf")), logits)
    return logits


def sample_next(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    sample: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    logits = logits[:, -1, :] / max(float(temperature), 1e-6)
    logits = filter_logits(logits, top_k=top_k, top_p=top_p)
    probs = torch.softmax(logits, dim=-1)
    if sample:
        return torch.multinomial(probs, num_samples=1, generator=generator)
    return torch.argmax(probs, dim=-1, keepdim=True)


@torch.no_grad()
def rollout_suffix_from_branch(
    model,
    base_tokens: torch.Tensor,
    *,
    class_label: int,
    patch_id: int,
    branch_values: list[int],
    device: torch.device,
    temperature: float,
    top_k: int,
    top_p: float,
    sample: bool,
    seed: int,
) -> torch.Tensor:
    seq_len = int(base_tokens.numel())
    batch = len(branch_values)
    out = base_tokens.to(device=device, dtype=torch.long).view(1, -1).repeat(batch, 1)
    out[:, int(patch_id)] = torch.tensor(branch_values, dtype=torch.long, device=device)
    max_seq_length = seq_len + 1
    model.setup_caches(max_batch_size=batch, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
    cond = torch.full((batch,), int(class_label), dtype=torch.long, device=device)
    _logits, _ = model(None, cond, input_pos=torch.arange(0, 1, dtype=torch.long, device=device))
    for pos in range(int(patch_id)):
        _logits, _ = model(out[:, pos : pos + 1], cond_idx=None, input_pos=torch.tensor([pos + 1], dtype=torch.long, device=device))
    logits, _ = model(
        out[:, int(patch_id) : int(patch_id) + 1],
        cond_idx=None,
        input_pos=torch.tensor([int(patch_id) + 1], dtype=torch.long, device=device),
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    for pos in range(int(patch_id) + 1, seq_len):
        next_token = sample_next(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sample=sample,
            generator=generator,
        )
        out[:, pos : pos + 1] = next_token
        if pos < seq_len - 1:
            logits, _ = model(
                next_token,
                cond_idx=None,
                input_pos=torch.tensor([pos + 1], dtype=torch.long, device=device),
            )
    return out.detach().cpu()


def make_branch_tokens(
    base_tokens: torch.Tensor,
    row: dict[str, Any],
    codes: list[int],
) -> torch.Tensor:
    patch_id = int(row["patch_id"])
    variants = base_tokens.view(1, -1).repeat(len(codes), 1)
    variants[:, patch_id] = torch.tensor(codes, dtype=torch.long)
    return variants


def patch_box(row: dict[str, Any], *, image_size: int, grid: int, context: int = 0) -> tuple[int, int, int, int]:
    patch = image_size // grid
    r = int(row["row"])
    c = int(row["col"])
    x0 = max(0, (c - context) * patch)
    y0 = max(0, (r - context) * patch)
    x1 = min(image_size, (c + 1 + context) * patch)
    y1 = min(image_size, (r + 1 + context) * patch)
    return x0, y0, x1, y1


def draw_box(image: Image.Image, row: dict[str, Any], *, grid: int, color: str) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    box = patch_box(row, image_size=out.size[0], grid=grid, context=0)
    for offset in range(3):
        draw.rectangle((box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset), outline=color)
    return out


def crop_context(image: Image.Image, row: dict[str, Any], *, grid: int, context: int) -> Image.Image:
    return image.crop(patch_box(row, image_size=image.size[0], grid=grid, context=context))


def average_pairwise_l2(images: list[Image.Image], row: dict[str, Any], *, grid: int, context: int) -> float:
    crops = [np.asarray(crop_context(img, row, grid=grid, context=context), dtype=np.float32) / 255.0 for img in images]
    if len(crops) < 2:
        return float("nan")
    vals = []
    for i in range(len(crops)):
        for j in range(i + 1, len(crops)):
            vals.append(float(np.mean((crops[i] - crops[j]) ** 2)))
    return float(np.mean(vals))


def image_grid(
    rows: list[dict[str, Any]],
    *,
    branches: int,
    path: Path,
    title: str,
    grid: int,
    context: int,
) -> str:
    import matplotlib.pyplot as plt

    cols = 2 + branches
    fig_w = 2.2 * cols
    fig_h = 2.1 * len(rows) + 0.6
    fig, axes = plt.subplots(len(rows), cols, figsize=(fig_w, fig_h), squeeze=False)
    for r_idx, item in enumerate(rows):
        row = item["record"]
        label = item["kind"]
        color = "#ff7f0e" if label == "singular" else "#1f77b4"
        full = draw_box(item["base_image"], row, grid=grid, color=color)
        panels = [full, crop_context(item["base_image"], row, grid=grid, context=context)]
        panels.extend([crop_context(img, row, grid=grid, context=context) for img in item["variant_images"][:branches]])
        titles = [
            f"{label}\nimg {row['sample_id']} patch {row['patch_id']}",
            f"original\ncode {row['target_code']}",
        ]
        titles.extend([f"branch {i + 1}\ncode {code}" for i, code in enumerate(item["codes"][:branches])])
        for c_idx in range(cols):
            ax = axes[r_idx, c_idx]
            ax.imshow(panels[c_idx])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(titles[c_idx], fontsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(color if c_idx == 0 else "#dddddd")
                spine.set_linewidth(2 if c_idx == 0 else 0.8)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return str(path)


def diversity_summary_figure(rows: list[dict[str, Any]], path: Path) -> str:
    import matplotlib.pyplot as plt

    singular = np.asarray([row["crop_pairwise_l2"] for row in rows if row["kind"] == "singular"], dtype=np.float64)
    regular = np.asarray([row["crop_pairwise_l2"] for row in rows if row["kind"] == "regular"], dtype=np.float64)
    pair_count = min(len(singular), len(regular))
    diffs = singular[:pair_count] - regular[:pair_count]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    means = [float(np.nanmean(singular)), float(np.nanmean(regular))]
    medians = [float(np.nanmedian(singular)), float(np.nanmedian(regular))]
    axes[0].bar([0, 1], means, color=["#ff7f0e", "#1f77b4"], width=0.62)
    axes[0].scatter([0, 1], medians, color="black", marker="_", s=220, label="median", zorder=3)
    axes[0].set_xticks([0, 1], ["singular", "regular"])
    axes[0].set_ylabel("mean pairwise crop L2")
    axes[0].set_title("Branch diversity")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    jitter = np.linspace(-0.08, 0.08, pair_count) if pair_count > 1 else np.asarray([0.0])
    axes[1].scatter(np.zeros(pair_count) + jitter, diffs, color="#444444", s=34)
    if pair_count:
        axes[1].hlines(float(np.nanmean(diffs)), -0.22, 0.22, color="#d62728", linewidth=2)
        axes[1].text(
            0.03,
            0.95,
            f"wins {int(np.sum(diffs > 0))}/{pair_count}",
            transform=axes[1].transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlim(-0.45, 0.45)
    axes[1].set_xticks([0], ["matched pairs"])
    axes[1].set_ylabel("singular - regular crop L2")
    axes[1].set_title("Paired effect")
    axes[1].grid(axis="y", alpha=0.25)

    xs = np.arange(pair_count)
    axes[2].plot(xs, singular[:pair_count], "o-", color="#ff7f0e", label="singular")
    axes[2].plot(xs, regular[:pair_count], "o-", color="#1f77b4", label="regular")
    for idx, (s_val, r_val) in enumerate(zip(singular[:pair_count], regular[:pair_count])):
        axes[2].vlines(idx, min(float(s_val), float(r_val)), max(float(s_val), float(r_val)), color="#bbbbbb", linewidth=1, zorder=0)
    if pair_count > 20:
        tick_idx = xs[::4]
        if len(tick_idx) == 0 or tick_idx[-1] != xs[-1]:
            tick_idx = np.append(tick_idx, xs[-1])
    else:
        tick_idx = xs
    axes[2].set_xticks(tick_idx, [str(int(i) + 1) for i in tick_idx])
    axes[2].set_xlabel("matched image pair")
    axes[2].set_ylabel("pairwise crop L2")
    axes[2].set_title("Per-pair diversity")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("Singular visual-token branches are denser and more ambiguous", y=1.03)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return str(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    repo_path = resolve_llamagen_repo(args.llamagen_repo or None)
    profile = dict(LLAMAGEN_PROFILES[args.profile])
    tokens = load_tokens(Path(args.tokens_path))
    records = json.loads(Path(args.records).read_text(encoding="utf-8"))
    inferred_samples = max(int(row["sample_id"]) for row in records) + 1
    max_samples = int(args.max_samples) if int(args.max_samples) > 0 else inferred_samples
    tokens = tokens[:max_samples]
    labels = load_class_labels_file(Path(args.class_labels_file), samples=int(tokens.shape[0]))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = int(round(math.sqrt(int(tokens.shape[1]))))
    if grid * grid != int(tokens.shape[1]):
        raise ValueError("token sequence length must be a square")

    pairs = choose_anchor_pairs(
        records,
        selector=args.selector,
        pairs=args.pairs,
        min_patch_id=args.min_patch_id,
        max_patch_id=args.max_patch_id,
        score_mode=args.score_mode,
        seed=args.seed,
    )
    if not pairs:
        raise RuntimeError("no anchor pairs found")

    vq_model = load_vq_model(profile, repo_path, device)
    ar_model = None
    if args.rollout_suffix:
        ar_model, _profile, missing, unexpected = load_llamagen_gpt(
            profile_name=args.profile,
            repo_path=repo_path,
            device=device,
            dtype=torch.float32,
        )
        if missing or unexpected:
            print(f"[warn] GPT missing={missing} unexpected={unexpected}", flush=True)

    gallery_rows = []
    summary_rows = []
    for pair_idx, (singular, regular) in enumerate(pairs):
        for kind, row in (("singular", singular), ("regular", regular)):
            sample_id = int(row["sample_id"])
            codes = branch_codes(row, branches=args.branches)
            base = tokens[sample_id].clone()
            if args.rollout_suffix:
                variant_tokens = rollout_suffix_from_branch(
                    ar_model,
                    base,
                    class_label=int(labels[sample_id]),
                    patch_id=int(row["patch_id"]),
                    branch_values=codes,
                    device=device,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    sample=bool(args.sample_suffix),
                    seed=int(args.seed) + pair_idx * 101 + (0 if kind == "singular" else 17),
                )
            else:
                variant_tokens = make_branch_tokens(base, row, codes)
            all_tokens = torch.cat([base.view(1, -1), variant_tokens], dim=0)
            decoded = decode_token_batch(vq_model, all_tokens, profile, device)
            base_image = decoded[0]
            variant_images = decoded[1:]
            spread = average_pairwise_l2(variant_images, row, grid=grid, context=args.crop_context)
            full_spread = float(np.mean([
                np.mean((np.asarray(variant_images[i], dtype=np.float32) - np.asarray(variant_images[j], dtype=np.float32)) ** 2) / (255.0 ** 2)
                for i in range(len(variant_images))
                for j in range(i + 1, len(variant_images))
            ])) if len(variant_images) > 1 else float("nan")
            gallery_rows.append(
                {
                    "kind": kind,
                    "record": row,
                    "codes": codes,
                    "base_image": base_image,
                    "variant_images": variant_images,
                }
            )
            summary_rows.append(
                {
                    "kind": kind,
                    "sample_id": sample_id,
                    "class_label": int(labels[sample_id]),
                    "patch_id": int(row["patch_id"]),
                    "row": int(row["row"]),
                    "col": int(row["col"]),
                    "target_code": int(row["target_code"]),
                    "branch_codes": codes,
                    "local_ball_ks": finite_float(row, "local_ball_ks"),
                    "local_ball_entropy": finite_float(row, "local_ball_entropy"),
                    "branch_ks": finite_float(row, "branch_ks"),
                    "branch_entropy": finite_float(row, "branch_entropy"),
                    "crop_pairwise_l2": spread,
                    "full_image_pairwise_l2": full_spread,
                }
            )

    gallery_path = out_dir / ("vq_ar_polysemy_branch_rollout_gallery.png" if args.rollout_suffix else "vq_ar_polysemy_branch_replacement_gallery.png")
    max_gallery_rows = int(args.max_gallery_rows)
    rendered_rows = gallery_rows[:max_gallery_rows] if max_gallery_rows > 0 else gallery_rows
    figures = {
        "branch_gallery": image_grid(
            rendered_rows,
            branches=args.branches,
            path=gallery_path,
            title=(
                "Singular vs regular AR branch rollouts after selected patch token"
                if args.rollout_suffix
                else "Singular vs regular AR top-token branch decodes"
            ),
            grid=grid,
            context=args.crop_context,
        )
    }
    figures["diversity_summary"] = diversity_summary_figure(
        summary_rows,
        out_dir / "vq_ar_polysemy_branch_diversity_summary.png",
    )

    singular_spreads = [row["crop_pairwise_l2"] for row in summary_rows if row["kind"] == "singular"]
    regular_spreads = [row["crop_pairwise_l2"] for row in summary_rows if row["kind"] == "regular"]
    summary = {
        "mode": "suffix_rollout" if args.rollout_suffix else "single_token_branch_decode",
        "profile": args.profile,
        "records": str(Path(args.records).resolve()),
        "tokens_path": str(Path(args.tokens_path).resolve()),
        "out_dir": str(out_dir),
        "selector": args.selector,
        "score_mode": args.score_mode,
        "pairs": int(args.pairs),
        "branches": int(args.branches),
        "crop_context": int(args.crop_context),
        "rollout_suffix": bool(args.rollout_suffix),
        "sample_suffix": bool(args.sample_suffix),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "mean_crop_pairwise_l2_singular": float(np.nanmean(singular_spreads)),
        "mean_crop_pairwise_l2_regular": float(np.nanmean(regular_spreads)),
        "crop_pairwise_l2_singular_minus_regular": float(np.nanmean(singular_spreads) - np.nanmean(regular_spreads)),
        "anchors": summary_rows,
        "figures": figures,
    }
    summary_path = out_dir / "vq_ar_polysemy_branch_gallery_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=[tag.strip() for tag in str(args.wandb_tags).split(",") if tag.strip()],
            config={k: v for k, v in summary.items() if isinstance(v, (str, int, float, bool))},
        )
        wandb.log(
            {
                "polysemy_branch/mean_crop_l2_singular": summary["mean_crop_pairwise_l2_singular"],
                "polysemy_branch/mean_crop_l2_regular": summary["mean_crop_pairwise_l2_regular"],
                "polysemy_branch/crop_l2_singular_minus_regular": summary["crop_pairwise_l2_singular_minus_regular"],
                "polysemy_branch/gallery": wandb.Image(figures["branch_gallery"]),
                "polysemy_branch/diversity_summary": wandb.Image(figures["diversity_summary"]),
            }
        )
        artifact = wandb.Artifact(f"{args.wandb_name}_outputs", type="analysis")
        artifact.add_file(str(summary_path))
        for path in figures.values():
            artifact.add_file(path)
            pdf = Path(path).with_suffix(".pdf")
            if pdf.exists():
                artifact.add_file(str(pdf))
        run.log_artifact(artifact)
        run.finish()
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(LLAMAGEN_PROFILES), default="c2i-B-256")
    parser.add_argument("--tokens-path", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--class-labels-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--llamagen-repo", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--selector", default="codebook_target_large_fiber")
    parser.add_argument("--score-mode", choices=["flatness", "geometry", "position", "random"], default="flatness")
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--branches", type=int, default=6)
    parser.add_argument("--min-patch-id", type=int, default=64)
    parser.add_argument("--max-patch-id", type=int, default=240)
    parser.add_argument("--crop-context", type=int, default=1)
    parser.add_argument("--max-gallery-rows", type=int, default=0, help="Render only the first N gallery rows while keeping all rows in the numeric summary.")
    parser.add_argument("--rollout-suffix", action="store_true")
    parser.add_argument("--sample-suffix", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="stratified-manifold-learning")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default="vq-ar-polysemy-branch-gallery")
    parser.add_argument("--wandb-tags", default="vq-ar,polysemy,branch-gallery")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = run(args)
    printable = {k: v for k, v in summary.items() if k != "anchors"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
