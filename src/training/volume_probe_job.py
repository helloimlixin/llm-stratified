from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from data import make_loaders
from models import DinoV2Wrapper
from utils import seed_everything
from volume_probe import run_volume_probe

from training.configs import VolumeProbeConfig
from training.model_factory import build_classifier_model


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

    if volume_cfg.feature_backbone == "dinov2":
        model = DinoV2Wrapper(model_name=volume_cfg.dinov2_model, token_layers=volume_cfg.dinov2_layers)
        patch_size_used = int(model.patch_size)
        print(f"[info] Using DINOv2 probe backbone {volume_cfg.dinov2_model} with patch size {patch_size_used}")
    else:
        model, patch_size_used = build_classifier_model(
            num_classes=num_classes,
            in_chans=in_chans,
            img_size=final_img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_rate=dropout_rate,
            timm_model=timm_model,
            timm_pretrained=timm_pretrained,
            announce=True,
        )
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
        "feature_backbone": volume_cfg.feature_backbone,
        "timm_model": timm_model,
        "timm_pretrained": timm_pretrained,
        "dinov2_model": volume_cfg.dinov2_model,
        "dinov2_layers": volume_cfg.dinov2_layers,
        "pixel_patch_stride": volume_cfg.pixel_patch_stride,
        "viz_projection_points": volume_cfg.viz_projection_points,
        "viz_curve_anchors": volume_cfg.viz_curve_anchors,
    }

    return run_volume_probe(
        model=model,
        loader=test_loader,
        device=device,
        dataset=dataset_name,
        patch_size=patch_size_used,
        pixel_patch_stride=volume_cfg.pixel_patch_stride,
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
        viz_projection_points=volume_cfg.viz_projection_points,
        viz_curve_anchors=volume_cfg.viz_curve_anchors,
        config=config,
    )
