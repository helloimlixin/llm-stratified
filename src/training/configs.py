from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional


@dataclass
class FiberConfig:
    enabled: bool = False
    embed_interval: int = 10
    max_tokens: int = 1024
    embed_full_val: bool = False
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


def _get_field_value(source: Any, name: str, default: Any) -> Any:
    if hasattr(source, "get"):
        try:
            return source.get(name, default)
        except Exception:
            pass
    return getattr(source, name, default)


def normalize_fiber_config(fiber_cfg: FiberConfig, patch_size: int) -> FiberConfig:
    """Ensure fiber config is consistent with the patch size."""
    if fiber_cfg.neighborhood_size is None:
        fiber_cfg.neighborhood_size = max(patch_size * 2, patch_size + 1)
    elif fiber_cfg.neighborhood_size <= patch_size:
        fiber_cfg.neighborhood_size = patch_size + 1
    return fiber_cfg


def build_fiber_config(
    source: Any,
    *,
    enabled: Optional[bool] = None,
    patch_size: Optional[int] = None,
) -> FiberConfig:
    """Build FiberConfig from an args/config object and apply defaults."""
    data: dict[str, Any] = {}
    for field in fields(FiberConfig):
        if field.name == "enabled":
            data[field.name] = enabled if enabled is not None else _get_field_value(source, field.name, field.default)
        else:
            data[field.name] = _get_field_value(source, field.name, field.default)
    fiber_cfg = FiberConfig(**data)
    if patch_size is not None:
        normalize_fiber_config(fiber_cfg, patch_size)
    return fiber_cfg


def build_volume_probe_config(
    source: Any,
    *,
    enabled: Optional[bool] = None,
) -> VolumeProbeConfig:
    data: dict[str, Any] = {}
    for field in fields(VolumeProbeConfig):
        if field.name == "enabled":
            data[field.name] = enabled if enabled is not None else _get_field_value(source, field.name, field.default)
        else:
            data[field.name] = _get_field_value(source, field.name, field.default)
    return VolumeProbeConfig(**data)

