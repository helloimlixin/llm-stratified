from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from data import make_loaders
from models import TinyViT, TimmViTWrapper, resolve_patch_size
from utils import seed_everything
from volume_probe import run_volume_probe

from training.configs import VolumeProbeConfig


def run_volume_probe_job(
    *,
    dataset_name: str,
    root: str,
    img_size: Optional[int],
    patch_size: int,
    embed_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: float,
    dropout_rate: float,
    batch_size_train: int,
    batch_size_test: int,
    num_workers: int,
    subset_train: Optional[int],
    subset_test: Optional[int],
    timm_model: Optional[str],
    timm_pretrained: bool,
    seed: int,
    output_dir: Path,
    volume_cfg: VolumeProbeConfig,
) -> dict:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _train_loader, test_loader, num_classes, in_chans, final_img_size, _task = make_loaders(
        dataset_name,
        root,
        img_size,
        batch_size_train,
        batch_size_test,
        num_workers,
        subset_train,
        subset_test,
        device,
        distributed=False,
        rank=0,
        world_size=1,
    )

    patch_size_used = patch_size
    if timm_model:
        model = TimmViTWrapper(timm_model, num_classes, pretrained=timm_pretrained)
        timm_patch = resolve_patch_size(model)
        if timm_patch:
            patch_size_used = timm_patch
            print(f"[info] Using timm {timm_model} with patch size {patch_size_used}")
    else:
        model = TinyViT(final_img_size, patch_size, in_chans, num_classes, embed_dim, depth, num_heads, mlp_ratio, dropout_rate)

    model = model.to(device)
    model.eval()

    config = {
        "dataset": dataset_name,
        "root": root,
        "img_size": final_img_size,
        "patch_size": patch_size_used,
        "max_tokens": volume_cfg.max_tokens,
        "vol_min": volume_cfg.vol_min,
        "vol_max": volume_cfg.vol_max,
        "ws": volume_cfg.ws,
        "alpha": volume_cfg.alpha,
        "nstrat": volume_cfg.nstrat,
        "seed": seed,
        "device": str(device),
        "batch_size_test": batch_size_test,
        "subset_test": subset_test,
        "timm_model": timm_model,
        "timm_pretrained": timm_pretrained,
    }

    return run_volume_probe(
        model=model,
        loader=test_loader,
        device=device,
        dataset=dataset_name,
        patch_size=patch_size_used,
        max_tokens=volume_cfg.max_tokens,
        vol_min=volume_cfg.vol_min,
        vol_max=volume_cfg.vol_max,
        ws=volume_cfg.ws,
        alpha=volume_cfg.alpha,
        nstrat=volume_cfg.nstrat,
        out_dir=output_dir,
        save_full=volume_cfg.save_full,
        progress=volume_cfg.progress,
        seed=seed,
        viz_images=volume_cfg.viz_images,
        viz_patches=volume_cfg.viz_patches,
        viz_nn_anchors=volume_cfg.viz_nn_anchors,
        viz_nn_k=volume_cfg.viz_nn_k,
        config=config,
    )

