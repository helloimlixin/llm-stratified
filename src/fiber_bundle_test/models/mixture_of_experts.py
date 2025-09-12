"""Mixture-of-experts models for stratified manifold learning."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Any
import logging

from .dictionary_learning import DictionaryExpertLISTA
from .gating_networks import GatingNetworkAttention, compute_expert_utilization

logger = logging.getLogger(__name__)


class MixtureOfDictionaryExperts(nn.Module):
    """Mixture-of-experts model using dictionary learning experts."""
    
    def __init__(self, 
                 input_dim: int, 
                 query_dim: int, 
                 code_dim: int, 
                 K: int,
                 projection_dim: int = 64, 
                 num_lista_layers: int = 5, 
                 sparsity_levels: Optional[List[int]] = None, 
                 threshold: float = 0.9):
        """
        Initialize mixture-of-experts model.
        
        Args:
            input_dim: Input dimension
            query_dim: Query dimension for gating
            code_dim: Dictionary code dimension
            K: Number of experts
            projection_dim: Final projection dimension
            num_lista_layers: Number of LISTA layers per expert
            sparsity_levels: Sparsity level for each expert
            threshold: Threshold for expert selection
        """
        super().__init__()
        self.K = K
        self.threshold = threshold
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        
        # Gating network
        self.gating_net = GatingNetworkAttention(input_dim, query_dim, K)
        
        # Set default sparsity levels if not provided
        if sparsity_levels is None:
            sparsity_levels = list(map(int, np.linspace(5, code_dim, K)))
        
        # Create dictionary experts with different sparsity levels
        self.experts = nn.ModuleList([
            DictionaryExpertLISTA(
                input_dim, 
                code_dim, 
                num_layers=num_lista_layers, 
                sparsity_level=sparsity_levels[i]
            )
            for i in range(K)
        ])
        
        # Projection head for final embeddings
        self.projection_head = nn.Sequential(
            nn.Linear(code_dim, code_dim),
            nn.ReLU(),
            nn.Linear(code_dim, projection_dim)
        )
        
        # Store sparsity levels as buffer
        self.register_buffer("sparsity_levels", torch.tensor(sparsity_levels, dtype=torch.float32))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with hard expert assignment.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Projected embeddings of shape (batch_size, projection_dim)
        """
        # Compute gating probabilities
        gating_probs = self.gating_net(x)
        batch_size = gating_probs.size(0)
        
        # Get sparse codes from all experts
        zs = []
        for expert in self.experts:
            _, z = expert(x)
            zs.append(z)
        zs = torch.stack(zs, dim=0)  # (K, batch_size, code_dim)
        
        # Hard expert assignment based on gating probabilities and sparsity
        selected_z = torch.empty(batch_size, zs.size(2), device=x.device)
        
        for i in range(batch_size):
            p = gating_probs[i]
            p_max = torch.max(p)
            
            # Find experts above threshold
            eligible = (p >= self.threshold * p_max).nonzero(as_tuple=True)[0]
            
            if eligible.numel() == 0:
                # If no expert meets threshold, use the best one
                idx = torch.argmax(p)
            else:
                # Among eligible experts, choose the one with lowest sparsity
                eligible_sparsity = self.sparsity_levels[eligible]
                min_idx = torch.argmin(eligible_sparsity)
                idx = eligible[min_idx]
            
            selected_z[i] = zs[idx, i, :]
        
        # Project to final embedding space
        projection = self.projection_head(selected_z)
        
        return projection
    
    def get_expert_assignment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get hard expert assignments for input.
        
        Args:
            x: Input tensor
            
        Returns:
            Expert indices for each sample
        """
        with torch.no_grad():
            gating_probs = self.gating_net(x)
            batch_size = gating_probs.size(0)
            assignments = torch.empty(batch_size, dtype=torch.long, device=x.device)
            
            for i in range(batch_size):
                p = gating_probs[i]
                p_max = torch.max(p)
                eligible = (p >= self.threshold * p_max).nonzero(as_tuple=True)[0]
                
                if eligible.numel() == 0:
                    assignments[i] = torch.argmax(p)
                else:
                    eligible_sparsity = self.sparsity_levels[eligible]
                    min_idx = torch.argmin(eligible_sparsity)
                    assignments[i] = eligible[min_idx]
            
            return assignments
    
    def get_gating_stats(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Get detailed gating statistics.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with gating statistics
        """
        with torch.no_grad():
            gating_probs = self.gating_net(x)
            assignments = self.get_expert_assignment(x)
            
            # Compute utilization statistics
            utilization = compute_expert_utilization(gating_probs)
            
            # Add assignment statistics
            assignment_counts = torch.bincount(assignments, minlength=self.K)
            assignment_freq = assignment_counts.float() / len(assignments)
            
            utilization.update({
                'hard_assignments': assignments.cpu().numpy(),
                'hard_assignment_frequencies': assignment_freq.cpu().numpy(),
                'gating_probabilities': gating_probs.cpu().numpy()
            })
            
            return utilization
    
    def get_expert_codes(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get sparse codes from all experts.
        
        Args:
            x: Input tensor
            
        Returns:
            List of sparse codes from each expert
        """
        codes = []
        with torch.no_grad():
            for expert in self.experts:
                _, z = expert(x)
                codes.append(z)
        return codes
    
    def analyze_expert_specialization(self, x: torch.Tensor, 
                                    labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Analyze how experts specialize on different types of data.
        
        Args:
            x: Input tensor
            labels: Optional labels for analysis
            
        Returns:
            Analysis results
        """
        with torch.no_grad():
            assignments = self.get_expert_assignment(x)
            gating_stats = self.get_gating_stats(x)
            
            analysis = {
                'expert_assignments': assignments.cpu().numpy(),
                'utilization_stats': gating_stats,
                'sparsity_levels': self.sparsity_levels.cpu().numpy()
            }
            
            if labels is not None:
                # Analyze expert-label relationships
                labels_np = labels.cpu().numpy()
                expert_label_dist = {}
                
                for expert_idx in range(self.K):
                    expert_mask = (assignments == expert_idx)
                    if expert_mask.sum() > 0:
                        expert_labels = labels_np[expert_mask.cpu().numpy()]
                        unique_labels, counts = np.unique(expert_labels, return_counts=True)
                        expert_label_dist[expert_idx] = {
                            'labels': unique_labels.tolist(),
                            'counts': counts.tolist(),
                            'total_samples': len(expert_labels)
                        }
                
                analysis['expert_label_distribution'] = expert_label_dist
            
            return analysis


def contrastive_loss_with_labels(embeddings: torch.Tensor, 
                               labels: torch.Tensor, 
                               margin: float = 1.0) -> torch.Tensor:
    """
    Contrastive loss using actual labels.
    
    Args:
        embeddings: Embedding tensor of shape (batch_size, embedding_dim)
        labels: Label tensor of shape (batch_size,)
        margin: Margin for negative pairs
        
    Returns:
        Contrastive loss value
    """
    batch_size = embeddings.size(0)
    
    # Compute pairwise distances
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    
    # Create similarity matrix based on labels
    labels = labels.unsqueeze(1)
    sim_matrix = (labels == labels.T).float()
    
    # Positive pairs: minimize distance
    pos_loss = sim_matrix * (dist_matrix ** 2)
    
    # Negative pairs: maximize distance up to margin
    neg_loss = (1 - sim_matrix) * torch.clamp(margin - dist_matrix, min=0.0) ** 2
    
    return (pos_loss + neg_loss).mean()


def compute_load_balancing_loss(gating_probs: torch.Tensor, 
                              target_utilization: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute load balancing loss to encourage equal expert utilization.
    
    Args:
        gating_probs: Gating probabilities
        target_utilization: Target utilization per expert (default: uniform)
        
    Returns:
        Load balancing loss
    """
    K = gating_probs.size(1)
    
    # Compute actual utilization
    actual_utilization = gating_probs.mean(dim=0)
    
    # Target uniform utilization if not provided
    if target_utilization is None:
        target_utilization = torch.ones(K, device=gating_probs.device) / K
    
    # L2 loss between actual and target utilization
    load_loss = torch.mean((actual_utilization - target_utilization) ** 2)
    
    return load_loss
