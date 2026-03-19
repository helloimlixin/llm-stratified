from __future__ import annotations

from typing import Optional

import torch.nn as nn

from models import TinyViT, TimmViTWrapper, resolve_patch_size


def build_classifier_model(
    *,
    num_classes: int,
    in_chans: int,
    img_size: int,
    patch_size: int,
    embed_dim: int,
    depth: int,
    num_heads: int,
    mlp_ratio: float,
    dropout_rate: float,
    timm_model: Optional[str],
    timm_pretrained: bool,
    announce: bool = False,
) -> tuple[nn.Module, int]:
    patch_size_used = patch_size
    if timm_model:
        model = TimmViTWrapper(timm_model, num_classes, pretrained=timm_pretrained)
        timm_patch = resolve_patch_size(model)
        if timm_patch:
            patch_size_used = timm_patch
            if announce:
                print(f"[info] Using timm {timm_model} with patch size {patch_size_used}")
        return model, patch_size_used

    model = TinyViT(img_size, patch_size, in_chans, num_classes, embed_dim, depth, num_heads, mlp_ratio, dropout_rate)
    return model, patch_size_used
