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
from fiber_bundle import (
    normalize_volume_range,
    run_fiber_bundle_test_from_sorted_dists,
    sorted_distance_matrix,
    summarize_stratifications,
)
from models import TinyViT, TimmViTWrapper, resolve_patch_size
from utils import denormalize_images, seed_everything, to_serializable


def _flatten_patches(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    b, c, h, w = imgs.shape
    ps = int(patch_size)
    gh, gw = h // ps, w // ps
    if gh <= 0 or gw <= 0:
        return torch.empty(0, c * ps * ps)
    imgs = imgs[:, :, : gh * ps, : gw * ps].contiguous()
    return (
        imgs.reshape(b, c, gh, ps, gw, ps)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(b * gh * gw, c * ps * ps)
    )


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
    max_tokens: int,
    show_progress: bool,
    viz_images: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, patch_embeds, patch_pixels = [], [], []
    viz_imgs = []
    seen, warned = 0, False
    iterator = tqdm(loader, desc="Collect tokens", leave=False) if show_progress and tqdm else loader
    for batch in iterator:
        imgs = batch[0].to(device)
        feats = model.forward_features(imgs)
        start_idx = 2 if getattr(model, "has_dist_token", False) else 1
        token_flat = feats[:, start_idx:, :].reshape(-1, feats.shape[-1])

        patch_emb = _forward_patch_embed(model, imgs)
        patch_emb_flat = patch_emb.reshape(-1, patch_emb.shape[-1])

        denorm = denormalize_images(imgs, dataset)
        patch_pix_flat = _flatten_patches(denorm, patch_size).to(dtype=torch.float32)

        max_viz = max(0, int(viz_images))
        if max_viz and len(viz_imgs) < max_viz:
            take = min(max_viz - len(viz_imgs), int(denorm.shape[0]))
            if take > 0:
                for i in range(take):
                    viz_imgs.append(denorm[i].detach().cpu())

        count = min(token_flat.shape[0], patch_emb_flat.shape[0], patch_pix_flat.shape[0])
        if not warned and count != token_flat.shape[0]:
            print(
                "[warn] Patch counts mismatch; truncating to common length "
                f"{count} (tokens {token_flat.shape[0]}, embeds {patch_emb_flat.shape[0]}, pixels {patch_pix_flat.shape[0]})."
            )
            warned = True

        tokens.append(token_flat[:count].detach().cpu())
        patch_embeds.append(patch_emb_flat[:count].detach().cpu())
        patch_pixels.append(patch_pix_flat[:count].detach().cpu())
        seen += count
        if seen >= max_tokens:
            break

    if not tokens:
        return torch.empty(0, 1), torch.empty(0, 1), torch.empty(0, 1), torch.empty(0, 3, 1, 1)
    return (
        torch.cat(tokens, dim=0)[:max_tokens],
        torch.cat(patch_embeds, dim=0)[:max_tokens],
        torch.cat(patch_pixels, dim=0)[:max_tokens],
        torch.stack(viz_imgs, dim=0) if viz_imgs else torch.empty(0, 3, 1, 1),
    )


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
    tokens, patch_embeds, patch_pixels, example_images = collect_representations(
        model=model,
        loader=loader,
        device=device,
        dataset=dataset,
        patch_size=patch_size,
        max_tokens=max_tokens,
        show_progress=progress,
        viz_images=viz_images,
    )

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
            n = int(patch_pixels.shape[0])
            ps = int(patch_size)
            if viz_patches > 0 and n > 0 and ps > 0:
                c = int(patch_pixels.shape[1] // max(1, ps * ps))
                if c > 0 and patch_pixels.shape[1] == c * ps * ps:
                    rng = np.random.default_rng(int(seed))
                    take = min(int(viz_patches), n)
                    idx = rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
                    patch_imgs = patch_pixels[idx].to(dtype=torch.float32).reshape(take, c, ps, ps).clamp(0, 1)
                    grid = make_grid(patch_imgs, nrow=min(8, take), padding=2)
                    out_path = out_dir / "example_patches.png"
                    to_pil_image(grid).save(out_path)
                    viz_files["example_patches"] = out_path.name
        except Exception as e:
            print(f"[viz] failed to save example_patches: {e}")

        # Nearest-neighbor patch retrieval visualizations (using patch crops for all representations).
        def _save_nn_grid(rep: torch.Tensor, rep_name: str) -> None:
            n = int(rep.shape[0])
            if n <= 1 or viz_nn_anchors <= 0 or viz_nn_k <= 0:
                return
            ps = int(patch_size)
            c = int(patch_pixels.shape[1] // max(1, ps * ps))
            if c <= 0 or patch_pixels.shape[1] != c * ps * ps:
                return
            anchors = min(int(viz_nn_anchors), n)
            nn_k_eff = min(int(viz_nn_k), n - 1)
            rng = np.random.default_rng(int(seed) + (hash(rep_name) % 10000))
            anchor_idx = rng.choice(n, size=anchors, replace=False) if anchors < n else np.arange(n)
            rep_f = rep.to(dtype=torch.float32)
            patches = patch_pixels.to(dtype=torch.float32)
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
            _save_nn_grid(tokens, "tokens")
            _save_nn_grid(patch_embeds, "patch_embeddings")
            _save_nn_grid(patch_pixels, "patch_pixels")
        except Exception as e:
            print(f"[viz] failed to save nn grids: {e}")

    if viz_files:
        results_payload["viz"] = viz_files

    for name, emb in [
        ("tokens", tokens),
        ("patch_embeddings", patch_embeds),
        ("patch_pixels", patch_pixels),
    ]:
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
        "timm_model": args.timm_model,
        "timm_pretrained": bool(args.timm_pretrained),
    }
    out_dir = Path(args.outdir)
    run_volume_probe(
        model=model,
        loader=test_loader,
        device=device,
        dataset=args.dataset,
        patch_size=patch_size,
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
