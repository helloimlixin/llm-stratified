from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional


@dataclass
class FiberConfig:
    enabled: bool = False
    embed_interval: int = 10
    max_tokens: int = 1024
    embed_full_val: bool = False
    embedding_animation: bool = True
    embedding_animation_fps: int = 4
    vol_min: int = 8
    vol_max: int = 64
    ws: int = 8
    alpha: float = 5e-3
    nstrat: int = 3
    neighborhood_size: Optional[int] = None
    polysemy: bool = False
    polysemy_k: int = 48
    polysemy_anchors: int = 12
    polysemy_grid_cols: int = 8
    polysemy_invert: bool = False
    polysemy_invert_steps: int = 200
    polysemy_invert_restarts: int = 6
    polysemy_invert_lr: float = 0.08
    polysemy_invert_tv: float = 1e-3
    polysemy_invert_l2: float = 1e-4
    polysemy_invert_patch_only: bool = True
    polysemy_invert_blur_every: int = 10
    polysemy_invert_blur_sigma: float = 0.8
    vit_token_polysemy: bool = False
    vit_token_polysemy_k: int = 256
    vit_token_polysemy_topk: int = 16
    vit_token_polysemy_ablate: bool = True
    vit_token_polysemy_ablate_batches: int = 10
    vit_token_polysemy_min_count: int = 50
    vit_token_polysemy_ablate_reps: int = 5
    sparse_probe: bool = False
    sparse_probe_radius: Optional[float] = None
    sparse_probe_auto_neighbor_k: int = 32
    sparse_probe_auto_radius_quantile: float = 0.5
    sparse_probe_min_patches: int = 12
    sparse_probe_max_anchors: Optional[int] = None
    sparse_probe_dictionary_size: int = 32
    sparse_probe_residual_threshold: float = 0.15
    sparse_probe_max_sparsity: int = 16


@dataclass
class VolumeProbeConfig:
    enabled: bool = False
    max_tokens: int = 2048
    vol_min: int = 8
    vol_max: int = 64
    ws: int = 8
    alpha: float = 5e-3
    nstrat: int = 3
    save_full: bool = False
    progress: bool = False
    viz_images: int = 16
    viz_patches: int = 64
    viz_nn_anchors: int = 3
    viz_nn_k: int = 8
    viz_projection_points: int = 1024
    viz_curve_anchors: int = 6
    feature_backbone: str = "model"
    dinov2_model: str = "facebook/dinov2-base"
    dinov2_layers: Optional[list[int]] = None
    pixel_patch_stride: Optional[int] = None


@dataclass
class SamFiberConfig:
    enabled: bool = False
    model_name: str = "facebook/sam-vit-base"
    epochs: int = 1
    resample_each_epoch: bool = True
    max_tokens: int = 1536
    analysis_patch_size: int = 16
    vol_min: int = 8
    vol_max: int = 64
    ws: int = 8
    alpha: float = 5e-3
    nstrat: int = 3
    neighborhood_size: Optional[int] = None
    progress: bool = True
    mask_threshold: float = 0.25
    max_boxes_per_image: int = 16
    mask_preview_images: int = 6
    multimask_output: bool = False
    sparse_probe: bool = False
    sparse_probe_radius: Optional[float] = None
    sparse_probe_auto_neighbor_k: int = 32
    sparse_probe_auto_radius_quantile: float = 0.5
    sparse_probe_min_patches: int = 12
    sparse_probe_max_anchors: Optional[int] = None
    sparse_probe_dictionary_size: int = 32
    sparse_probe_residual_threshold: float = 0.15
    sparse_probe_max_sparsity: int = 16
    sparse_probe_algorithm: str = "omp"
    sparse_probe_iht_steps: int = 80
    sparse_probe_iht_lr: Optional[float] = None
    sparse_probe_heatmap_images: int = 8
    embedding_animation: bool = True
    embedding_animation_fps: int = 1


def _read_field(source: Any, name: str, default: Any) -> Any:
    if hasattr(source, "get"):
        try:
            return source.get(name, default)
        except Exception:
            pass
    return getattr(source, name, default)


def _config_values(source: Any, config_type: Any, *, enabled: Optional[bool] = None) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field in fields(config_type):
        default = field.default
        if field.name == "enabled" and enabled is not None:
            values[field.name] = enabled
        else:
            values[field.name] = _read_field(source, field.name, default)
    return values


def _normalize_fiber_config(config: FiberConfig, patch_size: int) -> FiberConfig:
    if config.neighborhood_size is None:
        config.neighborhood_size = max(patch_size * 2, patch_size + 1)
    elif config.neighborhood_size <= patch_size:
        config.neighborhood_size = patch_size + 1
    return config


def make_fiber_config(
    source: Any,
    *,
    enabled: Optional[bool] = None,
    patch_size: Optional[int] = None,
) -> FiberConfig:
    config = FiberConfig(**_config_values(source, FiberConfig, enabled=enabled))
    if patch_size is not None:
        _normalize_fiber_config(config, patch_size)
    return config


def make_volume_probe_config(
    source: Any,
    *,
    enabled: Optional[bool] = None,
) -> VolumeProbeConfig:
    return VolumeProbeConfig(**_config_values(source, VolumeProbeConfig, enabled=enabled))


def _normalize_sam_fiber_config(config: SamFiberConfig) -> SamFiberConfig:
    analysis_patch_size = max(1, int(config.analysis_patch_size))
    config.analysis_patch_size = analysis_patch_size
    if config.neighborhood_size is None:
        config.neighborhood_size = max(analysis_patch_size * 2, analysis_patch_size + 1)
    elif config.neighborhood_size <= analysis_patch_size:
        config.neighborhood_size = analysis_patch_size + 1
    return config


def make_sam_fiber_config(
    source: Any,
    *,
    enabled: Optional[bool] = None,
) -> SamFiberConfig:
    config = SamFiberConfig(**_config_values(source, SamFiberConfig, enabled=enabled))
    return _normalize_sam_fiber_config(config)
