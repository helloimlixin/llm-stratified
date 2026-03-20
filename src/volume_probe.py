#!/usr/bin/env python3
"""
No-training volume estimation for token, patch-embedding, and raw-patch spaces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from torchvision.utils import make_grid
    from torchvision.transforms.functional import to_pil_image
except Exception:  # pragma: no cover
    make_grid = None
    to_pil_image = None

from data import make_loaders
from stratified_geometry import (
    normalize_volume_range,
    run_fiber_bundle_test_from_sorted_dists,
    sorted_distance_matrix,
    summarize_stratifications,
)
from models import DinoV2Wrapper, TinyViT, TimmViTWrapper, resolve_patch_size
from utils import denormalize_images, seed_everything, to_serializable


def _flatten_patches(imgs: torch.Tensor, patch_size: int, patch_stride: int | None = None) -> torch.Tensor:
    b, c, h, w = imgs.shape
    ps = int(patch_size)
    stride = int(patch_stride or patch_size)
    if min(h, w) < ps or stride <= 0:
        return torch.empty(0, c * ps * ps)
    patches = torch.nn.functional.unfold(imgs.contiguous(), kernel_size=ps, stride=stride)
    return patches.transpose(1, 2).reshape(b * patches.shape[-1], c * ps * ps)


def _resolve_prefix_tokens(model: torch.nn.Module) -> int:
    if hasattr(model, "num_prefix_tokens"):
        return int(getattr(model, "num_prefix_tokens"))
    return 2 if getattr(model, "has_dist_token", False) else 1


def _prepare_probe_inputs(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    dataset: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "prepare_images_for_features"):
        return model.prepare_images_for_features(imgs, dataset)
    return imgs, denormalize_images(imgs, dataset)


def _forward_feature_pack(model: torch.nn.Module, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
    if hasattr(model, "forward_feature_pack"):
        return model.forward_feature_pack(imgs)
    return {
        "tokens": model.forward_features(imgs),
        "patch_embeddings": _forward_patch_embed(model, imgs),
    }


def _flatten_token_tensor(tokens: torch.Tensor, prefix_tokens: int) -> torch.Tensor:
    start_idx = max(0, min(prefix_tokens, int(tokens.shape[1])))
    return tokens[:, start_idx:, :].reshape(-1, tokens.shape[-1])


def _representation_prefix_tokens(model: torch.nn.Module, rep_name: str) -> int:
    if rep_name == "tokens" or rep_name.startswith("tokens_"):
        return _resolve_prefix_tokens(model)
    if rep_name == "patch_embeddings" and bool(getattr(model, "patch_embeddings_include_prefix_tokens", False)):
        return _resolve_prefix_tokens(model)
    return 0


def _pixel_rep_name(patch_size: int, patch_stride: int) -> str:
    return "patch_pixels" if int(patch_stride) == int(patch_size) else f"patch_pixels_stride_{int(patch_stride)}"


def _forward_patch_embed(model: torch.nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    pe = getattr(model, "patch_embed", None)
    if pe is None and hasattr(model, "backbone"):
        pe = getattr(model.backbone, "patch_embed", None)
    if pe is None:
        raise RuntimeError("Model has no patch_embed for patch embedding extraction.")
    out = pe(imgs)
    if isinstance(out, tuple):
        out = out[0]
    if out.dim() == 4:
        out = out.flatten(2).transpose(1, 2)
    return out


@torch.no_grad()
def collect_representations(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    dataset: str,
    patch_size: int,
    pixel_patch_stride: int | None,
    max_tokens: int,
    show_progress: bool,
    viz_images: int = 16,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    rep_buffers: dict[str, list[torch.Tensor]] = {}
    viz_imgs = []
    seen, warned = 0, False
    pixel_stride = int(pixel_patch_stride or patch_size)
    overlap_name = _pixel_rep_name(patch_size, pixel_stride)
    iterator = tqdm(loader, desc="Collect tokens", leave=False) if show_progress and tqdm else loader
    for batch in iterator:
        imgs = batch[0].to(device)
        model_imgs, denorm = _prepare_probe_inputs(model, imgs, dataset)
        feature_pack = _forward_feature_pack(model, model_imgs)
        flattened = {
            name: _flatten_token_tensor(feats, _representation_prefix_tokens(model, name)).detach().cpu()
            for name, feats in feature_pack.items()
        }
        patch_pix_flat = _flatten_patches(denorm, patch_size, patch_stride=patch_size).to(dtype=torch.float32).cpu()
        overlap_flat = _flatten_patches(denorm, patch_size, patch_stride=pixel_stride).to(dtype=torch.float32).cpu()

        max_viz = max(0, int(viz_images))
        if max_viz and len(viz_imgs) < max_viz:
            take = min(max_viz - len(viz_imgs), int(denorm.shape[0]))
            if take > 0:
                for i in range(take):
                    viz_imgs.append(denorm[i].detach().cpu())

        aligned_counts = [patch_pix_flat.shape[0]] + [rep.shape[0] for rep in flattened.values()]
        count = min(aligned_counts) if aligned_counts else 0
        if not warned and any(rep.shape[0] != count for rep in flattened.values()):
            print(
                "[warn] Patch counts mismatch across representations; truncating to common length "
                f"{count} ({', '.join(f'{name} {rep.shape[0]}' for name, rep in flattened.items())}, pixels {patch_pix_flat.shape[0]})."
            )
            warned = True

        for name, rep in flattened.items():
            rep_buffers.setdefault(name, []).append(rep[:count])
        rep_buffers.setdefault("patch_pixels", []).append(patch_pix_flat[:count])
        if overlap_name != "patch_pixels":
            rep_buffers.setdefault(overlap_name, []).append(overlap_flat)
        seen += count
        if seen >= max_tokens:
            break

    if not rep_buffers:
        return {}, torch.empty(0, 3, 1, 1)
    reps = {name: torch.cat(parts, dim=0)[:max_tokens] for name, parts in rep_buffers.items() if parts}
    return reps, torch.stack(viz_imgs, dim=0) if viz_imgs else torch.empty(0, 3, 1, 1)


def _run_volume_estimation(
    embeddings: torch.Tensor,
    *,
    vol_min: int,
    vol_max: int,
    ws: int,
    alpha: float,
    nstrat: int,
) -> tuple[dict, np.ndarray, list[dict], dict | None]:
    coords = embeddings.cpu().numpy().astype(np.float64)
    dists_sorted = sorted_distance_matrix(coords)
    npts = int(dists_sorted.shape[0])
    if npts < 2:
        return {"num_tokens": npts, "tokens_with_strata": 0}, np.array([], dtype=np.float64), [], None

    vol_min_adj, vol_max_adj = normalize_volume_range(npts, vol_min, vol_max)
    results = run_fiber_bundle_test_from_sorted_dists(
        dists_sorted,
        vol_min=vol_min_adj,
        vol_max=vol_max_adj,
        ws=ws,
        alpha=alpha,
        nstrat=nstrat,
    )
    summary = summarize_stratifications(results, alpha=alpha)
    dims = np.array(
        [res["dimensions"][0] if res and res.get("dimensions") else np.nan for res in results],
        dtype=np.float64,
    )
    knn_curve = None
    if vol_max_adj > vol_min_adj:
        radii_mat = dists_sorted[vol_min_adj:vol_max_adj, :]  # (k, npts)
        qs = np.array([0.1, 0.5, 0.9], dtype=np.float64)
        qvals = np.quantile(radii_mat, qs, axis=1)  # (q, k)
        knn_curve = {
            "k_min": int(vol_min_adj),
            "k_max": int(vol_max_adj - 1),
            "k_values": list(range(int(vol_min_adj), int(vol_max_adj))),
            "quantiles": qs.tolist(),
            "radii": {f"q{int(q * 100)}": qvals[i].astype(float).tolist() for i, q in enumerate(qs.tolist())},
        }
    return summary, dims, results, knn_curve


def run_volume_probe(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    dataset: str,
    patch_size: int,
    pixel_patch_stride: int | None,
    max_tokens: int,
    vol_min: int,
    vol_max: int,
    ws: int,
    alpha: float,
    nstrat: int,
    out_dir: Path,
    save_full: bool = False,
    progress: bool = False,
    seed: int = 1337,
    viz_images: int = 16,
    viz_patches: int = 64,
    viz_nn_anchors: int = 3,
    viz_nn_k: int = 8,
    config: dict | None = None,
) -> dict:
    representations, example_images = collect_representations(
        model=model,
        loader=loader,
        device=device,
        dataset=dataset,
        patch_size=patch_size,
        pixel_patch_stride=pixel_patch_stride,
        max_tokens=max_tokens,
        show_progress=progress,
        viz_images=viz_images,
    )
    patch_pixels = representations.get("patch_pixels", torch.empty(0, 1))
    overlap_pixel_names = [name for name in representations if name.startswith("patch_pixels_stride_")]

    out_dir.mkdir(parents=True, exist_ok=True)
    results_payload = {
        "config": config or {},
        "representations": {},
    }

    # Lightweight visualizations (saved to disk; can be logged to W&B by the caller).
    viz_files: dict[str, str] = {}
    if make_grid is not None and to_pil_image is not None:
        try:
            if isinstance(example_images, torch.Tensor) and example_images.numel() > 0:
                imgs = example_images[: max(0, int(viz_images))].clamp(0, 1)
                if imgs.numel() > 0:
                    grid = make_grid(imgs, nrow=min(4, int(imgs.shape[0])), padding=2)
                    out_path = out_dir / "example_images.png"
                    to_pil_image(grid).save(out_path)
                    viz_files["example_images"] = out_path.name
        except Exception as e:
            print(f"[viz] failed to save example_images: {e}")

        try:
            ps = int(patch_size)
            for rep_name in ["patch_pixels"] + overlap_pixel_names:
                rep = representations.get(rep_name)
                if rep is None:
                    continue
                n = int(rep.shape[0])
                if viz_patches <= 0 or n <= 0 or ps <= 0:
                    continue
                c = int(rep.shape[1] // max(1, ps * ps))
                if c <= 0 or rep.shape[1] != c * ps * ps:
                    continue
                rng = np.random.default_rng(int(seed) + (hash(rep_name) % 10000))
                take = min(int(viz_patches), n)
                idx = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
                patch_imgs = rep[idx].to(dtype=torch.float32).reshape(take, c, ps, ps).clamp(0, 1)
                grid = make_grid(patch_imgs, nrow=min(8, take), padding=2)
                out_path = out_dir / f"example_{rep_name}.png"
                to_pil_image(grid).save(out_path)
                viz_files[f"example_{rep_name}"] = out_path.name
        except Exception as e:
            print(f"[viz] failed to save example_patches: {e}")

        # Nearest-neighbor patch retrieval visualizations (using patch crops for all representations).
        def _save_nn_grid(rep: torch.Tensor, rep_name: str) -> None:
            n = int(rep.shape[0])
            if n <= 1 or viz_nn_anchors <= 0 or viz_nn_k <= 0:
                return
            ps = int(patch_size)
            display_source = rep if rep_name.startswith("patch_pixels_stride_") else patch_pixels
            c = int(display_source.shape[1] // max(1, ps * ps))
            if c <= 0 or display_source.shape[1] != c * ps * ps:
                return
            anchors = min(int(viz_nn_anchors), n)
            nn_k_eff = min(int(viz_nn_k), n - 1)
            rng = np.random.default_rng(int(seed) + (hash(rep_name) % 10000))
            anchor_idx = rng.choice(n, size=anchors, replace=False) if anchors < n else np.arange(n)
            rep_f = rep.to(dtype=torch.float32)
            patches = display_source.to(dtype=torch.float32)
            rows = []
            for a in anchor_idx.tolist():
                x0 = rep_f[int(a)]
                d = ((rep_f - x0) ** 2).sum(dim=1)
                nn = torch.topk(d, k=nn_k_eff + 1, largest=False).indices
                rows.append(patches[nn].reshape(-1, c, ps, ps).clamp(0, 1))
            if not rows:
                return
            grid_imgs = torch.cat(rows, dim=0)
            out_path = out_dir / f"nn_{rep_name}.png"
            grid = make_grid(grid_imgs, nrow=nn_k_eff + 1, padding=2)
            to_pil_image(grid).save(out_path)
            viz_files[f"nn_{rep_name}"] = out_path.name

        try:
            for rep_name, rep in representations.items():
                _save_nn_grid(rep, rep_name)
        except Exception as e:
            print(f"[viz] failed to save nn grids: {e}")

    if viz_files:
        results_payload["viz"] = viz_files

    for name, emb in representations.items():
        summary, dims, results, knn_curve = _run_volume_estimation(
            emb,
            vol_min=vol_min,
            vol_max=vol_max,
            ws=ws,
            alpha=alpha,
            nstrat=nstrat,
        )
        dims_path = out_dir / f"{name}_dims.npy"
        np.save(dims_path, dims)
        results_path = None
        if save_full:
            results_path = out_dir / f"{name}_fiber_results.json"
            with open(results_path, "w") as fp:
                json.dump(to_serializable(results), fp, indent=2)
        results_payload["representations"][name] = {
            "summary": summary,
            "knn_curve": knn_curve,
            "dims_path": dims_path.name,
            "results_path": results_path.name if results_path else None,
        }

    with open(out_dir / "volume_summary.json", "w") as fp:
        json.dump(to_serializable(results_payload), fp, indent=2)
    return results_payload


def _build_model(
    *, args: argparse.Namespace, num_classes: int, in_chans: int, img_size: int
) -> tuple[torch.nn.Module, int]:
    if args.feature_backbone == "dinov2":
        model = DinoV2Wrapper(model_name=args.dinov2_model, token_layers=args.dinov2_layers)
        return model, int(model.patch_size)
    if args.timm_model:
        model = TimmViTWrapper(args.timm_model, num_classes, pretrained=args.timm_pretrained)
        patch_size = resolve_patch_size(model) or args.patch_size
    else:
        if args.patch_size is None:
            raise ValueError("--patch-size is required for TinyViT.")
        model = TinyViT(
            img_size,
            args.patch_size,
            in_chans,
            num_classes,
            args.embed_dim,
            args.depth,
            args.num_heads,
            args.mlp_ratio,
            args.dropout,
        )
        patch_size = args.patch_size

    if patch_size is None:
        raise ValueError("Unable to resolve patch size; pass --patch-size explicitly.")
    if args.patch_size and patch_size != args.patch_size and args.timm_model:
        print(f"[warn] Using timm patch size {patch_size} (overriding --patch-size {args.patch_size}).")
    return model, int(patch_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-training volume estimation baselines.")
    parser.add_argument("--dataset", default="CIFAR10")
    parser.add_argument("--root", default="./data")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subset-test", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--vol-min", type=int, default=8)
    parser.add_argument("--vol-max", type=int, default=64)
    parser.add_argument("--ws", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=5e-3)
    parser.add_argument("--nstrat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--outdir", default="./runs/volume_probe")
    parser.add_argument("--timm-model", default=None)
    parser.add_argument("--timm-pretrained", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-full", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--feature-backbone", choices=["model", "dinov2"], default="model")
    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    parser.add_argument("--dinov2-layers", type=int, nargs="*", default=None)
    parser.add_argument("--pixel-patch-stride", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive.")
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    _, test_loader, num_classes, in_chans, final_img_size, _task = make_loaders(
        args.dataset,
        args.root,
        args.img_size,
        args.batch_size,
        args.batch_size,
        args.num_workers,
        None,
        args.subset_test,
        device,
        distributed=False,
        rank=0,
        world_size=1,
    )

    model, patch_size = _build_model(args=args, num_classes=num_classes, in_chans=in_chans, img_size=final_img_size)
    model = model.to(device)
    model.eval()

    config = {
        "dataset": args.dataset,
        "root": args.root,
        "img_size": final_img_size,
        "patch_size": patch_size,
        "max_tokens": args.max_tokens,
        "vol_min": args.vol_min,
        "vol_max": args.vol_max,
        "ws": args.ws,
        "alpha": args.alpha,
        "nstrat": args.nstrat,
        "seed": args.seed,
        "device": str(device),
        "feature_backbone": args.feature_backbone,
        "timm_model": args.timm_model,
        "timm_pretrained": bool(args.timm_pretrained),
        "dinov2_model": args.dinov2_model,
        "dinov2_layers": list(args.dinov2_layers) if args.dinov2_layers is not None else None,
        "pixel_patch_stride": args.pixel_patch_stride,
    }
    out_dir = Path(args.outdir)
    run_volume_probe(
        model=model,
        loader=test_loader,
        device=device,
        dataset=args.dataset,
        patch_size=patch_size,
        pixel_patch_stride=args.pixel_patch_stride,
        max_tokens=args.max_tokens,
        vol_min=args.vol_min,
        vol_max=args.vol_max,
        ws=args.ws,
        alpha=args.alpha,
        nstrat=args.nstrat,
        out_dir=out_dir,
        save_full=args.save_full,
        progress=args.progress,
        seed=args.seed,
        config=config,
    )
    print(f"Saved volume estimates -> {out_dir}")


if __name__ == "__main__":
    main()
