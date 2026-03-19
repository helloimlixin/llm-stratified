"""Model definitions for TinyViT, timm wrappers, and frozen DINO feature extractors."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None

try:
    from transformers import AutoImageProcessor, AutoModel
except ImportError:
    AutoImageProcessor = None
    AutoModel = None

from utils import denormalize_images

__all__ = [
    "DinoV2Wrapper",
    "PatchEmbed",
    "MlpBlock",
    "TransformerBlock",
    "TinyViT",
    "resolve_patch_size",
    "TimmViTWrapper",
]


class PatchEmbed(nn.Module):
    """Converts an image batch into a sequence of flattened patch embeddings."""

    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 192) -> None:
        super().__init__()
        self.img_size, self.patch_size, self.in_chans, self.embed_dim = img_size, patch_size, in_chans, embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        x = self.proj(x)
        _, _, h2, w2 = x.shape
        return x.flatten(2).transpose(1, 2), h2 * w2


class MlpBlock(nn.Module):
    """Feed-forward block used inside the transformer encoder."""

    def __init__(self, embed_dim: int, mlp_ratio: float = 2.0, dropout_rate: float = 0.1) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.drop(self.act(self.fc1(x))))))


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block with pre-norm and dropout."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.drop_path1 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MlpBlock(embed_dim, mlp_ratio, dropout_rate)
        self.drop_path2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop_path1(y)
        return x + self.drop_path2(self.mlp(self.norm2(x)))


class TinyViT(nn.Module):
    """Minimal Vision Transformer for small/medium-sized datasets."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 8,
        num_heads: int = 3,
        mlp_ratio: float = 2.0,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim, self.num_classes = embed_dim, num_classes
        self.num_prefix_tokens = 1
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout_rate) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x, _ = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(batch_size, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x)[:, 0])


def resolve_patch_size(model) -> int | None:
    """Best-effort patch size extraction for TinyViT/timm ViT."""
    pe = getattr(model, "patch_embed", None)
    if pe is None:
        return None
    ps = getattr(pe, "patch_size", None)
    if ps is None:
        return None
    if isinstance(ps, (tuple, list)):
        return int(ps[0])
    if hasattr(ps, "numel") and ps.numel() > 0:
        return int(ps[0])
    try:
        return int(ps)
    except Exception:
        return None


class TimmViTWrapper(nn.Module):
    """Wrap timm ViT to expose patch tokens via forward_features."""

    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for --timm-model; pip install timm")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.has_dist_token = getattr(self.backbone, "dist_token", None) is not None
        self.num_prefix_tokens = 2 if self.has_dist_token else 1
        self.embed_dim = getattr(self.backbone, "embed_dim", None) or getattr(self.backbone, "num_features", None)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.backbone.patch_embed(x)
        cls_tokens = self.backbone.cls_token.expand(batch_size, -1, -1)
        if self.has_dist_token:
            x = torch.cat((cls_tokens, self.backbone.dist_token.expand(batch_size, -1, -1), x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        return self.backbone.norm(self.backbone.blocks(x))

    def tokens_to_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_head"):
            return self.backbone.forward_head(tokens, pre_logits=False)
        head = getattr(self.backbone, "head", None)
        return head(tokens[:, 0]) if head is not None else tokens[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokens_to_logits(self.forward_features(x))


def _resolve_processor_size(processor, config) -> int:
    size = getattr(processor, "size", None)
    if isinstance(size, dict):
        for key in ("height", "shortest_edge", "width"):
            value = size.get(key)
            if value is not None:
                return int(value)
        if size:
            return int(next(iter(size.values())))
    if isinstance(size, (tuple, list)) and size:
        return int(size[0])
    if isinstance(size, int):
        return int(size)
    return int(getattr(config, "image_size", 224))


class DinoV2Wrapper(nn.Module):
    """Frozen DINOv2 feature extractor with layer-wise patch token access."""

    def __init__(self, model_name: str = "facebook/dinov2-base", token_layers: Optional[list[int]] = None):
        super().__init__()
        if AutoImageProcessor is None or AutoModel is None:
            raise ImportError("transformers is required for DINOv2 probing; pip install transformers")

        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        self.embed_dim = int(getattr(self.backbone.config, "hidden_size", 0))
        self.patch_size = int(getattr(self.backbone.config, "patch_size", 14))
        self.expected_image_size = _resolve_processor_size(self.processor, self.backbone.config)
        self.image_mean = tuple(float(x) for x in getattr(self.processor, "image_mean", (0.485, 0.456, 0.406)))
        self.image_std = tuple(float(x) for x in getattr(self.processor, "image_std", (0.229, 0.224, 0.225)))
        self.num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))
        self.num_prefix_tokens = 1 + max(0, self.num_register_tokens)
        self.patch_embeddings_include_prefix_tokens = True
        self.token_layers = [int(layer) for layer in (token_layers or [-1])]

    def prepare_images_for_features(
        self,
        imgs: torch.Tensor,
        dataset_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        imgs01 = denormalize_images(imgs, dataset_name)
        if imgs01.shape[-2:] != (self.expected_image_size, self.expected_image_size):
            imgs01 = F.interpolate(
                imgs01,
                size=(self.expected_image_size, self.expected_image_size),
                mode="bilinear",
                align_corners=False,
            )
        mean = torch.tensor(self.image_mean, device=imgs01.device, dtype=imgs01.dtype).view(1, -1, 1, 1)
        std = torch.tensor(self.image_std, device=imgs01.device, dtype=imgs01.dtype).view(1, -1, 1, 1)
        pixel_values = (imgs01 - mean) / std
        return pixel_values, imgs01

    def _resolve_hidden_state_index(self, layer: int, num_states: int) -> int:
        idx = int(layer)
        if idx < 0:
            idx = num_states + idx
        if idx < 0 or idx >= num_states:
            raise ValueError(f"Requested DINO hidden state {layer} outside valid range [0, {num_states - 1}]")
        return idx

    def _layer_key(self, layer: int, resolved_idx: int, *, single_layer: bool) -> str:
        if single_layer:
            return "tokens"
        if layer < 0:
            return f"tokens_layer_last{abs(layer) - 1}" if layer < -1 else "tokens_layer_last"
        return f"tokens_layer_{resolved_idx:02d}"

    def forward_feature_pack(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        hidden_states = tuple(outputs.hidden_states or ())
        if not hidden_states:
            raise RuntimeError("DINOv2 backbone did not return hidden states.")

        pack: dict[str, torch.Tensor] = {
            "patch_embeddings": hidden_states[0],
        }
        single_layer = len(self.token_layers) == 1
        for layer in self.token_layers:
            resolved_idx = self._resolve_hidden_state_index(layer, len(hidden_states))
            pack[self._layer_key(layer, resolved_idx, single_layer=single_layer)] = hidden_states[resolved_idx]
        return pack
