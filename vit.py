import torch # vit using the pytorch framework
import torch.nn as nn
import torchvision.transforms as T # image manipulations (resize, to tensors, patching)
from torch.optim import Adam # using the Adam optimizer
from torchvision.datasets.mnist import MNIST # using MNIST for the moment
from torch.utils.data import DataLoader # data loading
import numpy as np # basic math operations like sin and cos for positional encodings

class PatchEmbedding(nn.Module):
    '''
    Embedding layer for image patches.
    '''
    def __init__(self, feature_dim, img_dim, patch_dim, num_channels):
        super().__init__()

        self.feature_dim = feature_dim # latent feature dimension
        self.img_dim = img_dim # image dimension
        self.patch_dim = patch_dim # patch size
        self.num_channels = num_channels # number of input channels

        self.linear_projection = nn.Conv2d(
                self.num_channels,
                self.feature_dim,
                kernel_size = self.patch_dim,
                stride=self.patch_dim)
    def forward(self, x):
        x = self.linear_projection(x) # (B, C, H, W)
        x = x.flatten(2) # (B, feature_dim, h, w) -> (B, feature_dim, p)
        x = x.transpose(1, 2) # (B, feature_dim, p)

        return x


