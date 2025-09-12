"""Mixture-of-experts specific training utilities."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

from .contrastive_training import ContrastiveTrainer, TrainingConfig

logger = logging.getLogger(__name__)


class MoETrainer(ContrastiveTrainer):
    """Specialized trainer for mixture-of-experts models."""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        """
        Initialize MoE trainer.
        
        Args:
            model: MoE model to train
            config: Training configuration
        """
        super().__init__(model, config)
        self.load_balancing_weight = 0.01  # Weight for load balancing loss
    
    def compute_load_balancing_loss(self, gating_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage equal expert utilization.
        
        Args:
            gating_probs: Gating probabilities
            
        Returns:
            Load balancing loss
        """
        K = gating_probs.size(1)
        
        # Compute actual utilization
        actual_utilization = gating_probs.mean(dim=0)
        
        # Target uniform utilization
        target_utilization = torch.ones(K, device=gating_probs.device) / K
        
        # L2 loss between actual and target utilization
        load_loss = torch.mean((actual_utilization - target_utilization) ** 2)
        
        return load_loss
    
    def train_epoch(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch with MoE-specific losses."""
        self.model.train()
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_load_loss = 0.0
        total_samples = 0
        expert_stats = []
        
        for batch_idx, batch in enumerate(data_loader):
            x, batch_labels = batch[0].to(self.device), batch[1].to(self.device)
            
            # Forward pass
            embeddings = self.model(x)
            
            # Contrastive loss
            contrastive_loss = self.contrastive_loss_with_labels(embeddings, batch_labels)
            
            # Load balancing loss (if model has gating network)
            load_loss = torch.tensor(0.0, device=self.device)
            if hasattr(self.model, 'gating_net'):
                gating_probs = self.model.gating_net(x)
                load_loss = self.compute_load_balancing_loss(gating_probs)
            
            # Total loss
            total_loss_batch = contrastive_loss + self.load_balancing_weight * load_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += total_loss_batch.item() * x.size(0)
            total_contrastive_loss += contrastive_loss.item() * x.size(0)
            total_load_loss += load_loss.item() * x.size(0)
            total_samples += x.size(0)
            
            # Collect expert utilization statistics
            if hasattr(self.model, 'get_gating_stats'):
                with torch.no_grad():
                    stats = self.model.get_gating_stats(x)
                    expert_stats.append(stats)
        
        avg_loss = total_loss / total_samples
        avg_contrastive_loss = total_contrastive_loss / total_samples
        avg_load_loss = total_load_loss / total_samples
        
        # Aggregate expert statistics
        if expert_stats:
            avg_expert_stats = self._aggregate_expert_stats(expert_stats)
        else:
            avg_expert_stats = {}
        
        return {
            'loss': avg_loss,
            'contrastive_loss': avg_contrastive_loss,
            'load_balancing_loss': avg_load_loss,
            'expert_stats': avg_expert_stats
        }
