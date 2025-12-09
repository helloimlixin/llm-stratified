"""Building blocks for a compact Vision Transformer prototype."""

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Convert an image into a sequence of flattened patch embeddings."""

    def __init__(
        self,
        feature_dim: int,
        img_dim: int,
        patch_dim: int,
        num_channels: int,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.num_channels = num_channels

        self.linear_projection = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.feature_dim,
            kernel_size=self.patch_dim,
            stride=self.patch_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return embeddings shaped as (batch, num_patches, feature_dim)."""

        x = self.linear_projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x
