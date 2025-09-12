"""Gating networks for mixture-of-experts models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class GatingNetworkAttention(nn.Module):
    """Attention-based gating network for expert selection."""
    
    def __init__(self, input_dim: int, query_dim: int, K: int):
        """
        Initialize attention-based gating network.
        
        Args:
            input_dim: Input feature dimension
            query_dim: Query projection dimension
            K: Number of experts
        """
        super().__init__()
        self.input_dim = input_dim
        self.query_dim = query_dim
        self.K = K
        
        self.query_proj = nn.Linear(input_dim, query_dim)
        self.keys = nn.Parameter(torch.randn(K, query_dim))
        
        # Initialize keys with Xavier initialization
        nn.init.xavier_uniform_(self.keys)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute gating probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Gating probabilities of shape (batch_size, K)
        """
        # Project input to query space
        query = self.query_proj(x)  # (batch_size, query_dim)
        
        # Compute attention scores
        logits = torch.matmul(query, self.keys.T) / (self.query_dim ** 0.5)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=1)
        
        return probs
    
    def get_expert_keys(self) -> torch.Tensor:
        """Get the expert keys."""
        return self.keys.detach()
    
    def get_key_similarities(self) -> torch.Tensor:
        """Compute pairwise similarities between expert keys."""
        with torch.no_grad():
            keys_norm = F.normalize(self.keys, p=2, dim=1)
            similarities = torch.matmul(keys_norm, keys_norm.T)
        return similarities


class SimpleGatingNetwork(nn.Module):
    """Simple MLP-based gating network."""
    
    def __init__(self, input_dim: int, hidden_dim: int, K: int, 
                 dropout: float = 0.1):
        """
        Initialize simple gating network.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            K: Number of experts
            dropout: Dropout rate
        """
        super().__init__()
        self.K = K
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, K)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute gating probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Gating probabilities
        """
        logits = self.network(x)
        return torch.softmax(logits, dim=1)


class TopKGatingNetwork(nn.Module):
    """Gating network with top-k expert selection."""
    
    def __init__(self, input_dim: int, query_dim: int, K: int, 
                 top_k: int = 2, noise_std: float = 0.1):
        """
        Initialize top-k gating network.
        
        Args:
            input_dim: Input dimension
            query_dim: Query dimension
            K: Number of experts
            top_k: Number of top experts to select
            noise_std: Standard deviation of gating noise
        """
        super().__init__()
        self.K = K
        self.top_k = min(top_k, K)
        self.noise_std = noise_std
        
        self.query_proj = nn.Linear(input_dim, query_dim)
        self.keys = nn.Parameter(torch.randn(K, query_dim))
        
        nn.init.xavier_uniform_(self.keys)
    
    def forward(self, x: torch.Tensor, training: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with top-k expert selection.
        
        Args:
            x: Input tensor
            training: Whether in training mode
            
        Returns:
            Tuple of (gating_probs, expert_indices)
        """
        query = self.query_proj(x)
        logits = torch.matmul(query, self.keys.T) / (query.shape[-1] ** 0.5)
        
        # Add noise during training for exploration
        if training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        
        # Select top-k experts
        top_logits, top_indices = torch.topk(logits, self.top_k, dim=1)
        
        # Create sparse gating probabilities
        gating_probs = torch.zeros_like(logits)
        top_probs = torch.softmax(top_logits, dim=1)
        gating_probs.scatter_(1, top_indices, top_probs)
        
        return gating_probs, top_indices


class AdaptiveGatingNetwork(nn.Module):
    """Adaptive gating network that learns to select appropriate experts."""
    
    def __init__(self, input_dim: int, K: int, temperature: float = 1.0):
        """
        Initialize adaptive gating network.
        
        Args:
            input_dim: Input dimension
            K: Number of experts
            temperature: Temperature for softmax (lower = more selective)
        """
        super().__init__()
        self.K = K
        self.temperature = temperature
        
        # Learn expert selection based on input characteristics
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU()
        )
        
        self.expert_selector = nn.Linear(input_dim // 4, K)
        
        # Learnable temperature
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive expert selection.
        
        Args:
            x: Input tensor
            
        Returns:
            Gating probabilities
        """
        features = self.feature_extractor(x)
        logits = self.expert_selector(features)
        
        # Apply learnable temperature
        temperature = torch.exp(self.log_temperature)
        probs = torch.softmax(logits / temperature, dim=1)
        
        return probs
    
    def get_temperature(self) -> float:
        """Get current temperature value."""
        return torch.exp(self.log_temperature).item()


# Utility functions
def compute_expert_utilization(gating_probs: torch.Tensor) -> dict:
    """
    Compute expert utilization statistics.
    
    Args:
        gating_probs: Gating probabilities of shape (batch_size, K)
        
    Returns:
        Dictionary with utilization statistics
    """
    with torch.no_grad():
        # Average probability per expert
        avg_probs = gating_probs.mean(dim=0)
        
        # Expert selection frequency (hard assignment)
        expert_assignments = gating_probs.argmax(dim=1)
        selection_counts = torch.bincount(expert_assignments, minlength=gating_probs.size(1))
        selection_freq = selection_counts.float() / gating_probs.size(0)
        
        # Entropy of expert distribution
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))
        max_entropy = torch.log(torch.tensor(float(gating_probs.size(1))))
        normalized_entropy = entropy / max_entropy
        
        return {
            'avg_probabilities': avg_probs.cpu().numpy(),
            'selection_frequencies': selection_freq.cpu().numpy(),
            'entropy': entropy.item(),
            'normalized_entropy': normalized_entropy.item(),
            'most_used_expert': expert_assignments.mode().values.item(),
            'expert_diversity': (selection_freq > 0.01).sum().item()  # Experts used > 1%
        }
