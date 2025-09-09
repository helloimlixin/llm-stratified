"""
Mixture of Experts (MoE) Models for Stratified Manifold Learning

This module contains the Top-K Straight-Through Estimator, LISTA-based dictionary experts,
gating networks, and the main Mixture of Dictionary Experts model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TopKSTE(torch.autograd.Function):
    """
    Top-K Straight-Through Estimator for differentiable sparse selection.
    
    This function implements a differentiable version of top-k selection that
    allows gradients to flow through during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, k):
        """
        Forward pass: select top-k elements and zero out the rest.
        
        Args:
            x: Input tensor
            k: Number of top elements to keep
            
        Returns:
            Tensor with top-k elements preserved, others zeroed
        """
        values, indices = torch.topk(torch.abs(x), k, dim=1)
        mask = torch.zeros_like(x)
        mask.scatter_(1, indices, 1.0)
        return x * mask
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: pass gradients through unchanged.
        
        Args:
            grad_output: Gradient from the next layer
            
        Returns:
            Gradient to pass to the previous layer
        """
        return grad_output, None


def topk_st(x, k):
    """
    Convenience function for Top-K Straight-Through Estimator.
    
    Args:
        x: Input tensor
        k: Number of top elements to keep
        
    Returns:
        Tensor with top-k elements preserved
    """
    return TopKSTE.apply(x, k)


class LISTALayer(nn.Module):
    """
    LISTA (Learned ISTA) layer for iterative sparse coding.
    
    This layer implements one iteration of the Learned ISTA algorithm,
    which learns to solve sparse coding problems iteratively.
    """
    def __init__(self, input_dim, code_dim, sparsity_level):
        """
        Initialize LISTA layer.
        
        Args:
            input_dim: Dimension of input features
            code_dim: Dimension of sparse codes
            sparsity_level: Number of non-zero elements to keep
        """
        super().__init__()
        self.W = nn.Linear(input_dim, code_dim, bias=False)
        self.S = nn.Linear(code_dim, code_dim, bias=False)
        self.sparsity_level = sparsity_level
    
    def forward(self, x, z_prev):
        """
        Forward pass of LISTA layer.
        
        Args:
            x: Input features
            z_prev: Previous sparse codes
            
        Returns:
            Updated sparse codes
        """
        update = self.W(x) + self.S(z_prev)
        new_z = topk_st(update, self.sparsity_level)
        return new_z


class DictionaryExpertLISTA(nn.Module):
    """
    Dictionary Expert using LISTA for sparse representation learning.
    
    This expert learns a dictionary and uses LISTA layers to compute
    sparse representations of input data.
    """
    def __init__(self, input_dim, code_dim, num_layers=5, sparsity_level=5):
        """
        Initialize Dictionary Expert.
        
        Args:
            input_dim: Dimension of input features
            code_dim: Dimension of sparse codes
            num_layers: Number of LISTA layers
            sparsity_level: Sparsity level for each layer
        """
        super().__init__()
        self.sparsity_level = sparsity_level
        self.dictionary = nn.Parameter(torch.randn(input_dim, code_dim) * 0.01)
        self.num_layers = num_layers
        self.lista_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            layer = LISTALayer(input_dim, code_dim, sparsity_level)
            self.lista_layers.append(layer)
    
    def forward(self, x):
        """
        Forward pass of Dictionary Expert.
        
        Args:
            x: Input features
            
        Returns:
            Tuple of (reconstruction, sparse_codes)
        """
        batch_size = x.size(0)
        device = x.device
        z = torch.zeros(batch_size, self.dictionary.shape[1], device=device)
        
        for layer in self.lista_layers:
            z = layer(x, z)
        
        recon = torch.matmul(z, self.dictionary.T)
        return recon, z


class GatingNetworkAttention(nn.Module):
    """
    Attention-based Gating Network for expert selection.
    
    This network learns to assign input samples to different experts
    based on attention mechanisms.
    """
    def __init__(self, input_dim, query_dim, K):
        """
        Initialize Gating Network.
        
        Args:
            input_dim: Dimension of input features
            query_dim: Dimension of query projections
            K: Number of experts
        """
        super().__init__()
        self.query_proj = nn.Linear(input_dim, query_dim)
        self.keys = nn.Parameter(torch.randn(K, query_dim))
    
    def forward(self, x):
        """
        Forward pass of Gating Network.
        
        Args:
            x: Input features
            
        Returns:
            Gating probabilities for each expert
        """
        query = self.query_proj(x)
        logits = torch.matmul(query, self.keys.T) / (query.shape[-1] ** 0.5)
        probs = torch.softmax(logits, dim=1)
        return probs


class MixtureOfDictionaryExperts(nn.Module):
    """
    Mixture of Dictionary Experts with varying sparsity levels.
    
    This model combines multiple dictionary experts with different sparsity levels
    and uses a gating network to select the most appropriate expert for each input.
    """
    def __init__(self, input_dim, query_dim, code_dim, K, projection_dim=64, 
                 num_lista_layers=5, sparsity_levels=None, threshold=0.9):
        """
        Initialize Mixture of Dictionary Experts.
        
        Args:
            input_dim: Dimension of input features
            query_dim: Dimension of query projections
            code_dim: Dimension of sparse codes
            K: Number of experts
            projection_dim: Dimension of final projection
            num_lista_layers: Number of LISTA layers per expert
            sparsity_levels: List of sparsity levels for each expert
            threshold: Threshold for expert selection
        """
        super().__init__()
        self.K = K
        self.threshold = threshold
        self.gating_net = GatingNetworkAttention(input_dim, query_dim, K)
        
        if sparsity_levels is None:
            sparsity_levels = list(map(int, np.linspace(5, code_dim, K)))
        
        self.experts = nn.ModuleList([
            DictionaryExpertLISTA(input_dim, code_dim, num_layers=num_lista_layers, 
                                sparsity_level=sparsity_levels[i])
            for i in range(K)
        ])
        
        self.projection_head = nn.Sequential(
            nn.Linear(code_dim, code_dim),
            nn.ReLU(),
            nn.Linear(code_dim, projection_dim)
        )
        
        self.register_buffer("sparsity_levels", torch.tensor(sparsity_levels, dtype=torch.float32))
    
    def forward(self, x):
        """
        Forward pass of Mixture of Dictionary Experts.
        
        Args:
            x: Input features
            
        Returns:
            Projected features from selected expert
        """
        gating_probs = self.gating_net(x)
        batch_size = gating_probs.size(0)
        
        # Get sparse codes from all experts
        zs = []
        for expert in self.experts:
            _, z = expert(x)
            zs.append(z)
        zs = torch.stack(zs, dim=0)
        
        # Select expert based on gating probabilities and sparsity preference
        selected_z = torch.empty(batch_size, zs.size(2), device=x.device)
        
        for i in range(batch_size):
            p = gating_probs[i]
            p_max = torch.max(p)
            
            # Find eligible experts (above threshold)
            eligible = (p >= self.threshold * p_max).nonzero(as_tuple=True)[0]
            
            if eligible.numel() == 0:
                # If no expert is above threshold, select the one with highest probability
                idx = torch.argmax(p)
            else:
                # Among eligible experts, select the one with lowest sparsity
                eligible_sparsity = self.sparsity_levels[eligible]
                min_idx = torch.argmin(eligible_sparsity)
                idx = eligible[min_idx]
            
            selected_z[i] = zs[idx, i, :]
        
        # Project the selected sparse codes
        projection = self.projection_head(selected_z)
        return projection
    
    def get_expert_assignments(self, x):
        """
        Get expert assignments for input data.
        
        Args:
            x: Input features
            
        Returns:
            Expert assignment indices
        """
        gating_probs = self.gating_net(x)
        batch_size = gating_probs.size(0)
        assignments = []
        
        for i in range(batch_size):
            p = gating_probs[i]
            p_max = torch.max(p)
            
            eligible = (p >= self.threshold * p_max).nonzero(as_tuple=True)[0]
            
            if eligible.numel() == 0:
                idx = torch.argmax(p)
            else:
                eligible_sparsity = self.sparsity_levels[eligible]
                min_idx = torch.argmin(eligible_sparsity)
                idx = eligible[min_idx]
            
            assignments.append(idx.item())
        
        return np.array(assignments)
    
    def get_gating_probabilities(self, x):
        """
        Get gating probabilities for input data.
        
        Args:
            x: Input features
            
        Returns:
            Gating probabilities for each expert
        """
        with torch.no_grad():
            return self.gating_net(x).cpu().numpy()
