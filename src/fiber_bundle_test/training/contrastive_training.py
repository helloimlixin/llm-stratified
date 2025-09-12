"""Contrastive training utilities for mixture-of-experts models."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from pathlib import Path
import logging
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for contrastive training."""
    num_epochs: int = 100
    learning_rate: float = 1e-3
    margin: float = 1.0
    batch_size: int = 32
    weight_decay: float = 1e-5
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    early_stopping_patience: int = 10
    min_delta: float = 1e-6
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10


class ContrastiveTrainer:
    """Trainer for contrastive learning with mixture-of-experts."""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        """
        Initialize contrastive trainer.
        
        Args:
            model: Model to train
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # Setup optimizer and scheduler
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'expert_utilization': []
        }
        
        # Early stopping
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Setup checkpoint directory
        if config.save_checkpoints:
            self.checkpoint_dir = Path(config.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def contrastive_loss_with_labels(self, embeddings: torch.Tensor, 
                                   labels: torch.Tensor, 
                                   margin: float = None) -> torch.Tensor:
        """
        Compute contrastive loss using labels.
        
        Args:
            embeddings: Embedding tensor
            labels: Label tensor
            margin: Margin for negative pairs
            
        Returns:
            Contrastive loss
        """
        if margin is None:
            margin = self.config.margin
        
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
    
    def train_epoch(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        expert_stats = []
        
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Training")):
            x, batch_labels = batch[0].to(self.device), batch[1].to(self.device)
            
            # Forward pass
            embeddings = self.model(x)
            loss = self.contrastive_loss_with_labels(embeddings, batch_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            
            # Collect expert utilization statistics
            if hasattr(self.model, 'get_gating_stats'):
                with torch.no_grad():
                    stats = self.model.get_gating_stats(x)
                    expert_stats.append(stats)
        
        avg_loss = total_loss / total_samples
        
        # Aggregate expert statistics
        if expert_stats:
            avg_expert_stats = self._aggregate_expert_stats(expert_stats)
        else:
            avg_expert_stats = {}
        
        return {
            'loss': avg_loss,
            'expert_stats': avg_expert_stats
        }
    
    def validate(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Validation"):
                x, batch_labels = batch[0].to(self.device), batch[1].to(self.device)
                embeddings = self.model(x)
                loss = self.contrastive_loss_with_labels(embeddings, batch_labels)
                
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
        
        return {'loss': total_loss / total_samples}
    
    def train(self, train_loader: torch.utils.data.DataLoader,
              val_loader: Optional[torch.utils.data.DataLoader] = None) -> Dict[str, List]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            # Training
            train_stats = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_stats['loss'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            if train_stats['expert_stats']:
                self.history['expert_utilization'].append(train_stats['expert_stats'])
            
            # Validation
            if val_loader is not None:
                val_stats = self.validate(val_loader)
                self.history['val_loss'].append(val_stats['loss'])
                current_loss = val_stats['loss']
            else:
                current_loss = train_stats['loss']
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Logging
            if epoch % self.config.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                if val_loader is not None:
                    logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - "
                              f"Train Loss: {train_stats['loss']:.6f}, "
                              f"Val Loss: {self.history['val_loss'][-1]:.6f}, "
                              f"LR: {lr:.2e}")
                else:
                    logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - "
                              f"Train Loss: {train_stats['loss']:.6f}, "
                              f"LR: {lr:.2e}")
            
            # Early stopping
            if current_loss < self.best_loss - self.config.min_delta:
                self.best_loss = current_loss
                self.patience_counter = 0
                
                # Save best model
                if self.config.save_checkpoints:
                    self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Regular checkpoint
            if (self.config.save_checkpoints and 
                epoch % (self.config.log_interval * 2) == 0):
                self.save_checkpoint(epoch)
        
        logger.info("Training completed")
        return self.history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config,
            'best_loss': self.best_loss
        }
        
        filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'
        filepath = self.checkpoint_dir / filename
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> int:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_loss = checkpoint['best_loss']
        
        epoch = checkpoint['epoch']
        logger.info(f"Checkpoint loaded from epoch {epoch}")
        
        return epoch
    
    def _aggregate_expert_stats(self, expert_stats_list: List[Dict]) -> Dict[str, Any]:
        """Aggregate expert statistics across batches."""
        if not expert_stats_list:
            return {}
        
        # Average utilization metrics
        avg_entropy = np.mean([stats['normalized_entropy'] for stats in expert_stats_list])
        avg_diversity = np.mean([stats['expert_diversity'] for stats in expert_stats_list])
        
        # Most common expert assignments
        all_assignments = np.concatenate([stats['hard_assignments'] for stats in expert_stats_list])
        unique_assignments, counts = np.unique(all_assignments, return_counts=True)
        
        return {
            'avg_normalized_entropy': avg_entropy,
            'avg_expert_diversity': avg_diversity,
            'assignment_distribution': dict(zip(unique_assignments.tolist(), counts.tolist())),
            'total_samples': len(all_assignments)
        }


# Convenience function from notebook
def train_contrastive_moe_with_labels(model: nn.Module, 
                                    data_loader: torch.utils.data.DataLoader, 
                                    num_epochs: int = 100, 
                                    lr: float = 1e-3, 
                                    margin: float = 1.0, 
                                    device: str = "cpu") -> nn.Module:
    """
    Train mixture-of-experts model with contrastive loss (notebook compatibility).
    
    Args:
        model: Model to train
        data_loader: Training data loader
        num_epochs: Number of epochs
        lr: Learning rate
        margin: Contrastive loss margin
        device: Device to train on
        
    Returns:
        Trained model
    """
    config = TrainingConfig(
        num_epochs=num_epochs,
        learning_rate=lr,
        margin=margin,
        log_interval=max(1, num_epochs // 10)
    )
    
    trainer = ContrastiveTrainer(model, config)
    trainer.train(data_loader)
    
    return model
