"""Dictionary learning and sparse coding components."""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TopKSTE(torch.autograd.Function):
    """Top-K Straight-Through Estimator for sparse coding."""
    
    @staticmethod
    def forward(ctx, x, k):
        """Forward pass: select top-k elements by absolute value."""
        values, indices = torch.topk(torch.abs(x), k, dim=1)
        mask = torch.zeros_like(x)
        mask.scatter_(1, indices, 1.0)
        return x * mask
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: straight-through estimator."""
        return grad_output, None


def topk_st(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Apply top-k straight-through estimator.
    
    Args:
        x: Input tensor
        k: Number of top elements to keep
        
    Returns:
        Tensor with only top-k elements preserved
    """
    return TopKSTE.apply(x, k)


class LISTALayer(nn.Module):
    """Single layer of the LISTA (Learned ISTA) algorithm."""
    
    def __init__(self, input_dim: int, code_dim: int, sparsity_level: int):
        """
        Initialize LISTA layer.
        
        Args:
            input_dim: Input dimension
            code_dim: Code dimension
            sparsity_level: Number of non-zero elements to keep
        """
        super().__init__()
        self.W = nn.Linear(input_dim, code_dim, bias=False)
        self.S = nn.Linear(code_dim, code_dim, bias=False)
        self.sparsity_level = sparsity_level
    
    def forward(self, x: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of LISTA layer.
        
        Args:
            x: Input data
            z_prev: Previous sparse code
            
        Returns:
            Updated sparse code
        """
        update = self.W(x) + self.S(z_prev)
        new_z = topk_st(update, self.sparsity_level)
        return new_z


class DictionaryExpertLISTA(nn.Module):
    """Dictionary learning expert using LISTA algorithm."""
    
    def __init__(self, input_dim: int, code_dim: int, 
                 num_layers: int = 5, sparsity_level: int = 5):
        """
        Initialize dictionary expert.
        
        Args:
            input_dim: Input dimension
            code_dim: Dictionary/code dimension
            num_layers: Number of LISTA layers
            sparsity_level: Sparsity level for codes
        """
        super().__init__()
        self.sparsity_level = sparsity_level
        self.dictionary = nn.Parameter(torch.randn(input_dim, code_dim) * 0.01)
        self.num_layers = num_layers
        
        # Create LISTA layers
        self.lista_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = LISTALayer(input_dim, code_dim, sparsity_level)
            self.lista_layers.append(layer)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: sparse coding and reconstruction.
        
        Args:
            x: Input data
            
        Returns:
            Tuple of (reconstruction, sparse_code)
        """
        batch_size = x.size(0)
        device = x.device
        
        # Initialize sparse code
        z = torch.zeros(batch_size, self.dictionary.shape[1], device=device)
        
        # Apply LISTA layers iteratively
        for layer in self.lista_layers:
            z = layer(x, z)
        
        # Reconstruct using dictionary
        recon = torch.matmul(z, self.dictionary.T)
        
        return recon, z
    
    def get_dictionary(self) -> torch.Tensor:
        """Get the learned dictionary."""
        return self.dictionary.detach()
    
    def get_sparsity_stats(self, z: torch.Tensor) -> dict:
        """
        Get sparsity statistics for codes.
        
        Args:
            z: Sparse codes
            
        Returns:
            Dictionary with sparsity statistics
        """
        with torch.no_grad():
            non_zero = (z != 0).float()
            actual_sparsity = non_zero.sum(dim=1).mean().item()
            target_sparsity = self.sparsity_level
            
            return {
                'actual_sparsity': actual_sparsity,
                'target_sparsity': target_sparsity,
                'sparsity_ratio': actual_sparsity / target_sparsity if target_sparsity > 0 else 0,
                'code_norm': z.norm(dim=1).mean().item()
            }
