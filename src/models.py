"""Model definitions for TinyViT, timm wrappers, and frozen feature extractors."""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None

try:
    from transformers import Aimv2VisionModel, AutoImageProcessor, AutoModel, SamModel, SamProcessor
except ImportError:
    Aimv2VisionModel = None
    AutoImageProcessor = None
    AutoModel = None
    SamModel = None
    SamProcessor = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

from utils import denormalize_images

__all__ = [
    "DinoV2FeatureExtractor",
    "DinoV2Wrapper",
    "FeedForwardBlock",
    "HfVisionFeatureExtractor",
    "PatchEmbeddingLayer",
    "PatchEmbed",
    "SamBackboneWrapper",
    "SamImageEncoder",
    "TinyViT",
    "VarAutoregressiveImageEncoder",
    "resolve_patch_size",
    "TimmVisionTransformer",
    "TimmViTWrapper",
    "VisionTransformerEncoderBlock",
    "TransformerBlock",
]


class PatchEmbeddingLayer(nn.Module):
    """Converts an image batch into a sequence of flattened patch embeddings."""

    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 192) -> None:
        super().__init__()
        self.img_size, self.patch_size, self.in_chans, self.embed_dim = img_size, patch_size, in_chans, embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        x = self.proj(x)
        _, _, h2, w2 = x.shape
        return x.flatten(2).transpose(1, 2), h2 * w2


class FeedForwardBlock(nn.Module):
    """Feed-forward block used inside the transformer encoder."""

    def __init__(self, embed_dim: int, mlp_ratio: float = 2.0, dropout_rate: float = 0.1) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class VisionTransformerEncoderBlock(nn.Module):
    """Standard Transformer encoder block with pre-norm and dropout."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.drop_path1 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForwardBlock(embed_dim, mlp_ratio, dropout_rate)
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
        self.patch_embed = PatchEmbeddingLayer(img_size, patch_size, in_chans, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList(
            [VisionTransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout_rate) for _ in range(depth)]
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
    if pe is None and hasattr(model, "backbone"):
        pe = getattr(model.backbone, "patch_embed", None)
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


class TimmVisionTransformer(nn.Module):
    """Wrap timm ViT to expose patch tokens via forward_features."""

    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for --timm-model; pip install timm")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.has_dist_token = getattr(self.backbone, "dist_token", None) is not None
        self.num_prefix_tokens = int(
            getattr(self.backbone, "num_prefix_tokens", 2 if self.has_dist_token else 1) or 0
        )
        self.embed_dim = getattr(self.backbone, "embed_dim", None) or getattr(self.backbone, "num_features", None)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        if isinstance(feats, dict):
            for key in ("x", "last_hidden_state", "tokens"):
                value = feats.get(key)
                if isinstance(value, torch.Tensor) and value.dim() == 3:
                    return value
            for value in feats.values():
                if isinstance(value, torch.Tensor) and value.dim() == 3:
                    return value
            raise RuntimeError(f"timm model returned feature dict without token tensor: {list(feats.keys())}")
        if not isinstance(feats, torch.Tensor) or feats.dim() != 3:
            raise RuntimeError(
                f"timm model returned unsupported feature shape {getattr(feats, 'shape', None)}; expected B x N x C"
            )
        return feats

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


class DinoV2FeatureExtractor(nn.Module):
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


class HfVisionFeatureExtractor(nn.Module):
    """Frozen generic Hugging Face vision encoder with patch-token access.

    This wrapper is for vision-first foundation models such as SigLIP2 and
    AIMv2 whose image encoders expose a last hidden state but do not need a
    model-specific probing path. It returns patch tokens only; the classifier
    wrapper adds a mean-token prefix for compatibility with the training/fiber
    pipeline.
    """

    def __init__(self, model_name: str, *, trust_remote_code: bool = True):
        super().__init__()
        if AutoImageProcessor is None or AutoModel is None:
            raise ImportError("transformers is required for HF vision probing; pip install transformers")

        self.model_name = str(model_name)
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
        )
        if "aimv2" in self.model_name.lower() and Aimv2VisionModel is not None:
            self.backbone = Aimv2VisionModel.from_pretrained(self.model_name)
        else:
            self.backbone = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=trust_remote_code,
            )
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        config = getattr(self.backbone, "config", None)
        vision_cfg = getattr(config, "vision_config", None) or config
        self.embed_dim = int(
            getattr(vision_cfg, "hidden_size", 0)
            or getattr(config, "hidden_size", 0)
            or getattr(vision_cfg, "projection_dim", 0)
        )
        self.patch_size = int(getattr(vision_cfg, "patch_size", getattr(config, "patch_size", 16)))
        self.expected_image_size = _resolve_processor_size(self.processor, vision_cfg or config)
        self.image_mean = tuple(float(x) for x in getattr(self.processor, "image_mean", (0.5, 0.5, 0.5)))
        self.image_std = tuple(float(x) for x in getattr(self.processor, "image_std", (0.5, 0.5, 0.5)))
        self.num_prefix_tokens = 0

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

    def _vision_outputs(self, pixel_values: torch.Tensor):
        kwargs = {"pixel_values": pixel_values, "output_hidden_states": True, "return_dict": True}
        try:
            return self.backbone(**kwargs)
        except (TypeError, ValueError):
            vision_model = getattr(self.backbone, "vision_model", None)
            if vision_model is None:
                raise
            return vision_model(**kwargs)

    def _last_hidden_state(self, outputs) -> torch.Tensor:
        hidden = getattr(outputs, "last_hidden_state", None)
        if isinstance(hidden, torch.Tensor):
            return hidden
        vision_outputs = getattr(outputs, "vision_model_output", None)
        hidden = getattr(vision_outputs, "last_hidden_state", None)
        if isinstance(hidden, torch.Tensor):
            return hidden
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            return hidden_states[-1]
        if isinstance(outputs, dict):
            for key in ("last_hidden_state", "tokens"):
                value = outputs.get(key)
                if isinstance(value, torch.Tensor):
                    return value
            vision_outputs = outputs.get("vision_model_output")
            if isinstance(vision_outputs, dict):
                value = vision_outputs.get("last_hidden_state")
                if isinstance(value, torch.Tensor):
                    return value
        raise RuntimeError(f"HF vision model {self.model_name} did not expose patch-token hidden states")

    def forward_feature_pack(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self._vision_outputs(pixel_values)
        hidden = self._last_hidden_state(outputs)
        if hidden.dim() != 3:
            raise RuntimeError(f"HF vision model returned unsupported hidden shape {tuple(hidden.shape)}")
        expected_grid = max(1, int(round(self.expected_image_size / max(1, self.patch_size))))
        expected_tokens = expected_grid * expected_grid
        if hidden.shape[1] >= expected_tokens:
            tokens = hidden[:, -expected_tokens:, :]
        else:
            tokens = hidden
        return {"tokens": tokens}


class SamImageEncoder(nn.Module):
    """Frozen SAM image encoder wrapper with optional box-prompted mask prediction."""

    def __init__(self, model_name: str = "facebook/sam-vit-base"):
        super().__init__()
        if SamModel is None or SamProcessor is None:
            raise ImportError("transformers is required for SAM probing; pip install transformers")

        self.model_name = model_name
        self.processor = SamProcessor.from_pretrained(model_name)
        self.backbone = SamModel.from_pretrained(model_name)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        self.num_prefix_tokens = 0
        self.embed_dim = int(getattr(self.backbone.config, "output_channels", 256))
        vision_cfg = getattr(self.backbone.config, "vision_config", None)
        prompt_cfg = getattr(self.backbone.config, "prompt_encoder_config", None)
        self.patch_size = int(
            getattr(prompt_cfg, "patch_size", getattr(vision_cfg, "patch_size", 16))
        )
        image_processor = getattr(self.processor, "image_processor", self.processor)
        self.expected_image_size = _resolve_processor_size(image_processor, vision_cfg or self.backbone.config)
        self.image_mean = tuple(float(x) for x in getattr(image_processor, "image_mean", (0.485, 0.456, 0.406)))
        self.image_std = tuple(float(x) for x in getattr(image_processor, "image_std", (0.229, 0.224, 0.225)))

    def _image_numpy(self, img: torch.Tensor, dataset_name: str) -> tuple[np.ndarray, torch.Tensor]:
        img01 = denormalize_images(img.unsqueeze(0), dataset_name).squeeze(0).detach().cpu()
        np_img = (img01.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
        return np_img, img01

    def prepare_single_image(self, img: torch.Tensor, dataset_name: str, *, device: torch.device) -> tuple[dict, torch.Tensor]:
        np_img, img01 = self._image_numpy(img, dataset_name)
        inputs = self.processor(images=[np_img], return_tensors="pt", do_rescale=False)
        inputs = inputs.to(device)
        return inputs, img01

    def prepare_images_for_features(
        self,
        imgs: torch.Tensor,
        dataset_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        np_imgs = []
        img_list = []
        for img in imgs:
            np_img, img01 = self._image_numpy(img, dataset_name)
            np_imgs.append(np_img)
            img_list.append(img01)
        inputs = self.processor(images=np_imgs, return_tensors="pt", do_rescale=False)
        return inputs["pixel_values"].to(imgs.device), torch.stack(img_list, dim=0).to(imgs.device)

    def get_image_embedding_map(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.backbone.get_image_embeddings(pixel_values)

    def forward_feature_pack(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding_map = self.get_image_embedding_map(pixel_values)
        tokens = embedding_map.flatten(2).transpose(1, 2)
        return {"tokens": tokens}

    @torch.no_grad()
    def predict_masks_for_boxes(
        self,
        *,
        img: torch.Tensor,
        dataset_name: str,
        boxes: list[list[float]],
        device: torch.device,
        image_embeddings: torch.Tensor | None = None,
        multimask_output: bool = False,
    ) -> list[torch.Tensor]:
        if not boxes:
            return []

        np_img, _img01 = self._image_numpy(img, dataset_name)
        inputs = self.processor(
            images=[np_img],
            input_boxes=[boxes],
            return_tensors="pt",
            do_rescale=False,
        ).to(device)
        if image_embeddings is None:
            image_embeddings = self.get_image_embedding_map(inputs["pixel_values"])

        outputs = self.backbone(
            image_embeddings=image_embeddings,
            input_boxes=inputs["input_boxes"],
            multimask_output=multimask_output,
        )
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            inputs["reshaped_input_sizes"].detach().cpu(),
        )
        if not masks:
            return []

        mask_batch = masks[0]
        if isinstance(mask_batch, (list, tuple)):
            return [torch.as_tensor(mask, dtype=torch.float32).squeeze() for mask in mask_batch]

        mask_tensor = torch.as_tensor(mask_batch, dtype=torch.float32)
        if mask_tensor.ndim == 4:
            if mask_tensor.shape[1] == 1:
                mask_tensor = mask_tensor.squeeze(1)
            else:
                mask_tensor = mask_tensor[:, 0]
        elif mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        return [mask_tensor[i].squeeze() for i in range(mask_tensor.shape[0])]


def _resolve_var_repo_path(var_repo_path: str | None = None) -> Path:
    candidates: list[Path] = []
    if var_repo_path:
        candidates.append(Path(var_repo_path))
    env_path = os.environ.get("VAR_REPO_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path.cwd() / "external" / "VAR",
            Path(__file__).resolve().parents[1] / "external" / "VAR",
            Path(__file__).resolve().parents[2] / "external" / "VAR",
        ]
    )
    for candidate in candidates:
        if (candidate / "models" / "__init__.py").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find the FoundationVision/VAR checkout. Clone it to external/VAR "
        "or set VAR_REPO_PATH to the checkout directory."
    )


def _import_var_build_vae_var(var_repo_path: str | None = None):
    """Import FoundationVision/VAR despite this repo's top-level models.py name."""
    repo_path = _resolve_var_repo_path(var_repo_path)
    repo_str = str(repo_path)
    saved_models = sys.modules.get("models")
    saved_path = list(sys.path)
    try:
        sys.path.insert(0, repo_str)
        if saved_models is not None:
            del sys.modules["models"]
        var_models = importlib.import_module("models")
        return var_models.build_vae_var
    finally:
        if saved_models is not None:
            sys.modules["models"] = saved_models
        else:
            sys.modules.pop("models", None)
        sys.path[:] = saved_path


def _parse_var_depth(model_name: str) -> int:
    stem = Path(str(model_name)).stem.lower()
    for part in stem.replace("-", "_").split("_"):
        if part.startswith("d") and part[1:].isdigit():
            return int(part[1:])
    raise ValueError(f"Could not parse VAR depth from model_name={model_name!r}; expected e.g. 'var_d30'.")


def _load_torch_state_dict(path: str, *, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class VarAutoregressiveImageEncoder(nn.Module):
    """Frozen VAR image generator wrapper exposing teacher-forced final-scale tokens.

    VAR is class-conditional on ImageNet, while our COCO probe is analysis-only.
    We therefore use the unconditional class embedding and teacher-force the visual
    code sequence obtained from VAR's VQ tokenizer. The exposed tokens are the
    final 16x16 next-scale hidden states before the vocabulary head.
    """

    def __init__(
        self,
        model_name: str = "var_d30",
        *,
        repo_id: str = "FoundationVision/var",
        var_repo_path: str | None = None,
        vae_filename: str = "vae_ch160v4096z32.pth",
        patch_nums: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        num_classes: int = 1000,
        class_label: int = -1,
    ):
        super().__init__()
        if hf_hub_download is None:
            raise ImportError("huggingface_hub is required for VAR probing; pip install huggingface_hub")

        self.model_name = str(model_name)
        self.repo_id = str(repo_id)
        self.patch_nums = tuple(int(p) for p in patch_nums)
        self.expected_image_size = 16 * int(self.patch_nums[-1])
        self.patch_size = self.expected_image_size // int(self.patch_nums[-1])
        self.num_prefix_tokens = 0
        self.class_label = int(class_label)

        depth = _parse_var_depth(self.model_name)
        self.depth = depth
        self.embed_dim = depth * 64

        build_vae_var = _import_var_build_vae_var(var_repo_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae, self.var = build_vae_var(
            device=device,
            patch_nums=self.patch_nums,
            depth=depth,
            num_classes=num_classes,
            shared_aln=False,
        )

        vae_path = hf_hub_download(repo_id=self.repo_id, filename=vae_filename)
        var_filename = self.model_name if self.model_name.endswith(".pth") else f"{self.model_name}.pth"
        var_path = hf_hub_download(repo_id=self.repo_id, filename=var_filename)
        self.vae.load_state_dict(_load_torch_state_dict(vae_path, map_location=device), strict=True)
        self.var.load_state_dict(_load_torch_state_dict(var_path, map_location=device), strict=True)

        self.var.cond_drop_rate = 0.0
        self.vae.eval()
        self.var.eval()
        for param in self.parameters():
            param.requires_grad_(False)

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
        return imgs01.mul(2.0).sub(1.0).clamp(-1.0, 1.0), imgs01

    def _hidden_states(
        self,
        label_B: torch.LongTensor,
        x_BLCv_wo_first_l: torch.Tensor,
    ) -> torch.Tensor:
        var = self.var
        _bg, ed = var.begin_ends[var.prog_si] if var.prog_si >= 0 else (0, var.L)
        B = x_BLCv_wo_first_l.shape[0]
        autocast_off = (
            torch.amp.autocast(device_type="cuda", enabled=False)
            if hasattr(torch, "amp") and label_B.is_cuda
            else nullcontext()
        )
        with autocast_off:
            sos = cond_BD = var.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, var.first_l, -1) + var.pos_start.expand(B, var.first_l, -1)
            if var.prog_si == 0:
                x_BLC = sos
            else:
                x_BLC = torch.cat((sos, var.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC = x_BLC + var.lvl_embed(var.lvl_1L[:, :ed].expand(B, -1)) + var.pos_1LC[:, :ed]

        attn_bias = var.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = var.shared_ada_lin(cond_BD)
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        for block in var.blocks:
            x_BLC = block(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        return x_BLC.float()

    def forward_feature_pack(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        idx_Bl = self.vae.img_to_idxBl(pixel_values, v_patch_nums=self.patch_nums)
        x_BLCv_wo_first_l = self.vae.quantize.idxBl_to_var_input(idx_Bl)
        label_value = self.var.num_classes if self.class_label < 0 else self.class_label
        labels = torch.full(
            (pixel_values.shape[0],),
            int(label_value),
            dtype=torch.long,
            device=pixel_values.device,
        )
        hidden = self._hidden_states(labels, x_BLCv_wo_first_l)
        final_start, final_end = self.var.begin_ends[-1]
        return {"tokens": hidden[:, final_start:final_end, :]}

    @torch.no_grad()
    def forward_generation_pack(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return final-scale teacher-forced tokens plus VAR vocabulary logits.

        The fiber pipeline uses the hidden states as frozen patch-token features.
        Generation-side diagnostics also need the actual next-scale VQ-code
        distribution, so this method exposes the matching final-scale logits
        and VQ targets.
        """
        idx_Bl = self.vae.img_to_idxBl(pixel_values, v_patch_nums=self.patch_nums)
        x_BLCv_wo_first_l = self.vae.quantize.idxBl_to_var_input(idx_Bl)
        label_value = self.var.num_classes if self.class_label < 0 else self.class_label
        labels = torch.full(
            (pixel_values.shape[0],),
            int(label_value),
            dtype=torch.long,
            device=pixel_values.device,
        )
        hidden = self._hidden_states(labels, x_BLCv_wo_first_l)
        logits = self.var(labels, x_BLCv_wo_first_l)
        targets = torch.cat(idx_Bl, dim=1)
        final_start, final_end = self.var.begin_ends[-1]
        return {
            "tokens": hidden[:, final_start:final_end, :],
            "logits": logits[:, final_start:final_end, :],
            "targets": targets[:, final_start:final_end],
        }


class _PatchEmbedShim(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = int(patch_size)


class FrozenBackboneClassifier(nn.Module):
    """Attach a trainable linear classifier to a frozen DINOv2 or SAM backbone.

    Exposes the ``forward_features`` / ``tokens_to_logits`` / ``patch_embed``
    interface that the fiber analysis pipeline expects.  The backbone is fully
    frozen; only ``head`` receives gradients.
    """

    def __init__(self, backbone_kind: str, num_classes: int, **backbone_kwargs):
        super().__init__()
        self.backbone_kind = backbone_kind.lower()
        if self.backbone_kind == "dinov2":
            self.backbone = DinoV2FeatureExtractor(**backbone_kwargs)
        elif self.backbone_kind in {"hf_vision", "siglip", "siglip2", "aimv2"}:
            self.backbone = HfVisionFeatureExtractor(**backbone_kwargs)
        elif self.backbone_kind == "sam":
            self.backbone = SamImageEncoder(**backbone_kwargs)
        elif self.backbone_kind == "var":
            self.backbone = VarAutoregressiveImageEncoder(**backbone_kwargs)
        else:
            raise ValueError(
                f"Unknown backbone_kind '{backbone_kind}'; expected 'dinov2', 'hf_vision', 'sam', or 'var'"
            )

        self.num_classes = int(num_classes)
        self.embed_dim = int(self.backbone.embed_dim)
        self.patch_size = int(self.backbone.patch_size)
        self.has_dist_token = False
        self.patch_embed = _PatchEmbedShim(self.patch_size)
        self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

    def _dataset_name(self) -> str:
        return getattr(self, "_dataset_name_cached", "IMAGENET")

    def set_dataset_name(self, dataset_name: str) -> None:
        self._dataset_name_cached = str(dataset_name)

    def _extract_last_layer_patch_tokens(self, imgs: torch.Tensor) -> torch.Tensor:
        pixel_values, _ = self.backbone.prepare_images_for_features(imgs, self._dataset_name())
        pack = self.backbone.forward_feature_pack(pixel_values)
        if self.backbone_kind == "dinov2":
            last_key = next((k for k in ("tokens", "tokens_layer_last") if k in pack), None)
            if last_key is None:
                last_key = [k for k in pack.keys() if k != "patch_embeddings"][-1]
            tokens = pack[last_key]
            n_register = int(getattr(self.backbone, "num_register_tokens", 0))
            cls = tokens[:, :1, :]
            patch = tokens[:, 1 + n_register :, :]
            return torch.cat([cls, patch], dim=1)
        tokens = pack["tokens"]
        cls = tokens.mean(dim=1, keepdim=True)
        return torch.cat([cls, tokens], dim=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            tokens = self._extract_last_layer_patch_tokens(x)
        return tokens

    def tokens_to_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(tokens[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokens_to_logits(self.forward_features(x))


PatchEmbed = PatchEmbeddingLayer
MlpBlock = FeedForwardBlock
TransformerBlock = VisionTransformerEncoderBlock
TimmViTWrapper = TimmVisionTransformer
DinoV2Wrapper = DinoV2FeatureExtractor
SamBackboneWrapper = SamImageEncoder
