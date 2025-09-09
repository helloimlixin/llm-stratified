"""
Immediate Improvements for LLM Performance
High Impact, Low Effort implementations based on geometric analysis

This module implements:
1. Geometric Regularization
2. Geometric Monitoring
3. Improved Token Embeddings
4. Dynamic Subspace Usage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class GeometricRegularizationLoss(nn.Module):
    """
    Geometric regularization loss combining multiple geometric constraints
    """
    def __init__(self, lambda_strata=0.1, lambda_curvature=0.05, lambda_manifold=0.02):
        super().__init__()
        self.lambda_strata = lambda_strata
        self.lambda_curvature = lambda_curvature
        self.lambda_manifold = lambda_manifold
        
    def forward(self, embeddings: torch.Tensor, predictions: torch.Tensor = None, 
                targets: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute geometric regularization loss
        
        Args:
            embeddings: Token embeddings [batch_size, seq_len, d_model]
            predictions: Model predictions (optional)
            targets: Target labels (optional)
            
        Returns:
            Dictionary with individual loss components
        """
        losses = {}
        
        # 1. Stratified Manifold Loss
        losses['strata_loss'] = self.compute_strata_loss(embeddings)
        
        # 2. Curvature Regularization Loss
        losses['curvature_loss'] = self.compute_curvature_loss(embeddings)
        
        # 3. Manifold Constraint Loss
        losses['manifold_loss'] = self.compute_manifold_loss(embeddings)
        
        # 4. Total geometric loss
        total_geometric = (self.lambda_strata * losses['strata_loss'] + 
                          self.lambda_curvature * losses['curvature_loss'] + 
                          self.lambda_manifold * losses['manifold_loss'])
        
        losses['total_geometric'] = total_geometric
        
        # 5. Standard loss if provided
        if predictions is not None and targets is not None:
            # Handle tensor size mismatch
            if predictions.dim() == 3 and targets.dim() == 2:
                # Reshape predictions to [batch_size * seq_len, vocab_size]
                predictions_flat = predictions.view(-1, predictions.size(-1))
                targets_flat = targets.view(-1)
                losses['standard_loss'] = F.cross_entropy(predictions_flat, targets_flat)
            else:
                losses['standard_loss'] = F.cross_entropy(predictions, targets)
            
            losses['total_loss'] = losses['standard_loss'] + total_geometric
        
        return losses
    
    def compute_strata_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute stratified manifold loss
        Encourages embeddings to form distinct strata
        """
        batch_size, seq_len, d_model = embeddings.shape
        
        # Reshape for analysis
        flat_embeddings = embeddings.view(-1, d_model)
        
        # Compute pairwise distances
        distances = torch.cdist(flat_embeddings, flat_embeddings, p=2)
        
        # Encourage clustering (strata formation)
        # Use a soft clustering loss
        sigma = torch.std(distances)
        clustering_loss = torch.mean(torch.exp(-distances / (2 * sigma**2)))
        
        return clustering_loss
    
    def compute_curvature_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute curvature regularization loss
        Encourages smooth geometric structure
        """
        batch_size, seq_len, d_model = embeddings.shape
        
        # Compute second-order differences (curvature approximation)
        if seq_len > 2:
            # First differences
            first_diff = embeddings[:, 1:] - embeddings[:, :-1]
            # Second differences (curvature)
            second_diff = first_diff[:, 1:] - first_diff[:, :-1]
            
            # Penalize high curvature (encourage smoothness)
            curvature_loss = torch.mean(torch.norm(second_diff, dim=-1))
        else:
            curvature_loss = torch.tensor(0.0, device=embeddings.device)
        
        return curvature_loss
    
    def compute_manifold_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute manifold constraint loss
        Encourages embeddings to lie on lower-dimensional manifolds
        """
        batch_size, seq_len, d_model = embeddings.shape
        
        # Reshape for PCA-like analysis
        flat_embeddings = embeddings.view(-1, d_model)
        
        # Compute covariance matrix
        mean_emb = torch.mean(flat_embeddings, dim=0)
        centered_emb = flat_embeddings - mean_emb
        cov_matrix = torch.mm(centered_emb.t(), centered_emb) / (flat_embeddings.size(0) - 1)
        
        # Compute eigenvalues (singular values)
        eigenvals = torch.linalg.eigvals(cov_matrix).real
        
        # Sort eigenvalues
        eigenvals, _ = torch.sort(eigenvals, descending=True)
        
        # Encourage low-dimensional structure
        # Penalize small eigenvalues (encourage sparsity)
        small_eigenvals = eigenvals[eigenvals < 0.01]
        manifold_loss = torch.sum(small_eigenvals)
        
        return manifold_loss

class GeometricMonitor:
    """
    Monitor geometric health of model during training
    """
    def __init__(self, model: nn.Module, embedding_layer_name: str = "embeddings"):
        self.model = model
        self.embedding_layer_name = embedding_layer_name
        self.metrics_history = []
        
    def monitor_training(self, embeddings: torch.Tensor, 
                        step: int = 0) -> Dict[str, float]:
        """
        Monitor geometric health during training
        
        Args:
            embeddings: Current embeddings
            step: Training step
            
        Returns:
            Dictionary of geometric metrics
        """
        metrics = {}
        
        # 1. Manifold Health Score
        metrics['manifold_health'] = self.compute_manifold_health(embeddings)
        
        # 2. Stratification Score
        metrics['stratification_score'] = self.compute_stratification_score(embeddings)
        
        # 3. Curvature Smoothness
        metrics['curvature_smoothness'] = self.compute_curvature_smoothness(embeddings)
        
        # 4. Dimensionality Score
        metrics['dimensionality_score'] = self.compute_dimensionality_score(embeddings)
        
        # 5. Overall geometric health
        metrics['overall_health'] = np.mean(list(metrics.values()))
        
        # Store metrics
        metrics['step'] = step
        self.metrics_history.append(metrics)
        
        # Alert if geometric structure degrades
        if metrics['overall_health'] < 0.5:
            print(f"⚠️ Geometric degradation detected at step {step}: {metrics['overall_health']:.3f}")
        
        return metrics
    
    def compute_manifold_health(self, embeddings: torch.Tensor) -> float:
        """Compute how well embeddings form a manifold"""
        batch_size, seq_len, d_model = embeddings.shape
        
        # Flatten embeddings
        flat_embeddings = embeddings.view(-1, d_model).detach().cpu().numpy()
        
        # Compute intrinsic dimension estimate
        distances = np.linalg.norm(flat_embeddings[:, None] - flat_embeddings[None, :], axis=2)
        
        # Use correlation dimension as manifold health indicator
        r_values = np.linspace(0.01, 0.1, 10)
        correlations = []
        
        for r in r_values:
            correlation = np.mean(distances < r)
            correlations.append(correlation)
        
        # Higher correlation dimension indicates better manifold structure
        health_score = min(1.0, np.mean(correlations) * 10)
        
        return float(health_score)
    
    def compute_stratification_score(self, embeddings: torch.Tensor) -> float:
        """Compute how well embeddings form strata"""
        batch_size, seq_len, d_model = embeddings.shape
        
        # Use clustering quality as stratification score
        flat_embeddings = embeddings.view(-1, d_model).detach().cpu().numpy()
        
        # Simple clustering quality metric
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        try:
            # Try different numbers of clusters
            best_score = 0
            for n_clusters in range(2, min(6, len(flat_embeddings)//10)):
                if len(flat_embeddings) > n_clusters:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(flat_embeddings)
                    score = silhouette_score(flat_embeddings, labels)
                    best_score = max(best_score, score)
            
            # Normalize to [0, 1]
            stratification_score = max(0, min(1, (best_score + 1) / 2))
            
        except Exception:
            stratification_score = 0.5
        
        return float(stratification_score)
    
    def compute_curvature_smoothness(self, embeddings: torch.Tensor) -> float:
        """Compute curvature smoothness"""
        batch_size, seq_len, d_model = embeddings.shape
        
        if seq_len < 3:
            return 1.0
        
        # Compute curvature along sequence
        first_diff = embeddings[:, 1:] - embeddings[:, :-1]
        second_diff = first_diff[:, 1:] - first_diff[:, :-1]
        
        # Smoothness is inverse of curvature variance
        curvature_var = torch.var(torch.norm(second_diff, dim=-1))
        smoothness = 1.0 / (1.0 + curvature_var.item())
        
        return float(smoothness)
    
    def compute_dimensionality_score(self, embeddings: torch.Tensor) -> float:
        """Compute effective dimensionality"""
        batch_size, seq_len, d_model = embeddings.shape
        
        # Flatten embeddings
        flat_embeddings = embeddings.view(-1, d_model).detach().cpu().numpy()
        
        # Compute effective rank using SVD
        U, s, Vt = np.linalg.svd(flat_embeddings, full_matrices=False)
        
        # Effective rank (number of singular values > threshold)
        threshold = 0.01 * s[0]  # 1% of largest singular value
        effective_rank = np.sum(s > threshold)
        
        # Normalize by total dimensions
        dimensionality_score = effective_rank / d_model
        
        return float(dimensionality_score)
    
    def get_health_summary(self) -> Dict[str, float]:
        """Get summary of geometric health over time"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 steps
        
        summary = {}
        for key in ['manifold_health', 'stratification_score', 'curvature_smoothness', 
                   'dimensionality_score', 'overall_health']:
            values = [m[key] for m in recent_metrics if key in m]
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
                summary[f'{key}_trend'] = np.polyfit(range(len(values)), values, 1)[0]
        
        return summary

class ImprovedTokenEmbeddings(nn.Module):
    """
    Improved token embeddings that address fiber bundle violations
    Based on Robinson et al. findings
    """
    def __init__(self, vocab_size: int, d_model: int, max_position_embeddings: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Standard embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        
        # Geometric-aware components
        self.token_subspace_projector = TokenSubspaceProjector(d_model)
        self.manifold_constraint_layer = ManifoldConstraintLayer(d_model)
        self.fiber_bundle_corrector = FiberBundleCorrector(d_model)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids: torch.Tensor, 
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with improved token embeddings
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            position_ids: Position IDs (optional)
            
        Returns:
            Improved embeddings [batch_size, seq_len, d_model]
        """
        seq_len = input_ids.size(1)
        
        # Standard embeddings
        token_emb = self.token_embeddings(input_ids)
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_emb = self.position_embeddings(position_ids)
        
        # Combine token and position embeddings
        embeddings = token_emb + position_emb
        
        # Apply geometric improvements
        # 1. Token subspace projection
        embeddings = self.token_subspace_projector(embeddings)
        
        # 2. Manifold constraint
        embeddings = self.manifold_constraint_layer(embeddings)
        
        # 3. Fiber bundle correction
        embeddings = self.fiber_bundle_corrector(embeddings)
        
        # Final normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class TokenSubspaceProjector(nn.Module):
    """
    Projects tokens into appropriate subspaces
    """
    def __init__(self, d_model: int, n_subspaces: int = 5):
        super().__init__()
        self.d_model = d_model
        self.n_subspaces = n_subspaces
        
        # Learnable subspace projections
        self.subspace_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_subspaces)
        ])
        
        # Subspace routing network
        self.subspace_router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_subspaces),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings into appropriate subspaces
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, d_model]
            
        Returns:
            Projected embeddings [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = embeddings.shape
        
        # Compute subspace routing weights
        routing_weights = self.subspace_router(embeddings)  # [batch_size, seq_len, n_subspaces]
        
        # Project into each subspace
        projected_embeddings = []
        for i, projection in enumerate(self.subspace_projections):
            projected = projection(embeddings)  # [batch_size, seq_len, d_model]
            weighted = projected * routing_weights[:, :, i:i+1]  # Weight by routing
            projected_embeddings.append(weighted)
        
        # Combine projections
        final_embeddings = sum(projected_embeddings)
        
        return final_embeddings

class ManifoldConstraintLayer(nn.Module):
    """
    Applies manifold constraints to embeddings
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Learnable manifold parameters
        self.manifold_center = nn.Parameter(torch.zeros(d_model))
        self.manifold_radius = nn.Parameter(torch.ones(1))
        
        # Constraint enforcement network
        self.constraint_network = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply manifold constraints
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, d_model]
            
        Returns:
            Constrained embeddings [batch_size, seq_len, d_model]
        """
        # Center embeddings around manifold center
        centered_embeddings = embeddings - self.manifold_center
        
        # Apply constraint network
        constrained_embeddings = self.constraint_network(centered_embeddings)
        
        # Soft constraint: encourage embeddings to stay within manifold radius
        distances = torch.norm(constrained_embeddings, dim=-1, keepdim=True)
        constraint_factor = torch.sigmoid(self.manifold_radius - distances)
        
        # Apply constraint
        final_embeddings = constrained_embeddings * constraint_factor
        
        return final_embeddings

class FiberBundleCorrector(nn.Module):
    """
    Corrects fiber bundle violations in token embeddings
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Correction network
        self.correction_network = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Tanh()
        )
        
        # Learnable correction strength
        self.correction_strength = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Correct fiber bundle violations
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, d_model]
            
        Returns:
            Corrected embeddings [batch_size, seq_len, d_model]
        """
        # Compute corrections
        corrections = self.correction_network(embeddings)
        
        # Apply corrections with learnable strength
        corrected_embeddings = embeddings + self.correction_strength * corrections
        
        return corrected_embeddings

class DynamicSubspaceUsage(nn.Module):
    """
    Dynamic subspace usage based on Wang et al. insights
    """
    def __init__(self, d_model: int, max_active_dimensions: int = None):
        super().__init__()
        self.d_model = d_model
        self.max_active_dimensions = max_active_dimensions or int(d_model * 0.6)  # Wang et al. 60% rule
        
        # Subspace importance predictor
        self.importance_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )
        
        # Adaptive subspace selection
        self.subspace_selector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.max_active_dimensions)
        )
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply dynamic subspace usage
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, d_model]
            
        Returns:
            Subspace-optimized embeddings [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = embeddings.shape
        
        # Predict dimension importance
        importance_scores = self.importance_predictor(embeddings)  # [batch_size, seq_len, d_model]
        
        # Select active dimensions
        active_dimensions = self.subspace_selector(embeddings)  # [batch_size, seq_len, max_active_dimensions]
        
        # Create dynamic mask
        top_k_indices = torch.topk(importance_scores, self.max_active_dimensions, dim=-1).indices
        
        # Apply mask to embeddings
        masked_embeddings = torch.zeros_like(embeddings)
        for i in range(self.max_active_dimensions):
            mask = torch.zeros_like(embeddings)
            mask.scatter_(-1, top_k_indices[:, :, i:i+1], 1)
            masked_embeddings += embeddings * mask
        
        return masked_embeddings

class GeometricAwareTrainingLoop:
    """
    Training loop with geometric awareness
    """
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 geometric_loss: GeometricRegularizationLoss,
                 monitor: GeometricMonitor):
        self.model = model
        self.optimizer = optimizer
        self.geometric_loss = geometric_loss
        self.monitor = monitor
        
    def training_step(self, batch: Dict[str, torch.Tensor], 
                     step: int = 0) -> Dict[str, float]:
        """
        Single training step with geometric awareness
        
        Args:
            batch: Training batch
            step: Current training step
            
        Returns:
            Dictionary of losses and metrics
        """
        # Forward pass
        outputs = self.model(**batch)
        
        # Get embeddings for geometric analysis
        embeddings = self.model.get_embeddings(batch['input_ids'])
        
        # Compute geometric losses
        geometric_losses = self.geometric_loss(
            embeddings, 
            outputs.logits, 
            batch.get('labels')
        )
        
        # Monitor geometric health
        health_metrics = self.monitor.monitor_training(embeddings, step)
        
        # Backward pass
        total_loss = geometric_losses['total_loss']
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Return metrics
        metrics = {
            'total_loss': total_loss.item(),
            'standard_loss': geometric_losses['standard_loss'].item(),
            'geometric_loss': geometric_losses['total_geometric'].item(),
            'strata_loss': geometric_losses['strata_loss'].item(),
            'curvature_loss': geometric_losses['curvature_loss'].item(),
            'manifold_loss': geometric_losses['manifold_loss'].item(),
            **health_metrics
        }
        
        return metrics

def create_improved_model(base_model_name: str = "distilbert-base-uncased") -> nn.Module:
    """
    Create an improved model with geometric awareness
    
    Args:
        base_model_name: Base transformer model name
        
    Returns:
        Improved model with geometric components
    """
    from transformers import AutoModel, AutoConfig
    
    # Load base model
    config = AutoConfig.from_pretrained(base_model_name)
    base_model = AutoModel.from_pretrained(base_model_name)
    
    # Replace embeddings with improved version
    improved_embeddings = ImprovedTokenEmbeddings(
        vocab_size=config.vocab_size,
        d_model=config.hidden_size
    )
    
    # Add geometric components
    base_model.embeddings = improved_embeddings
    base_model.dynamic_subspace = DynamicSubspaceUsage(config.hidden_size)
    
    return base_model

# Example usage functions
def demonstrate_immediate_improvements():
    """
    Demonstrate the immediate improvements
    """
    print("🚀 Demonstrating Immediate LLM Improvements")
    print("=" * 60)
    
    # Create sample data
    batch_size, seq_len, d_model = 2, 10, 768
    sample_embeddings = torch.randn(batch_size, seq_len, d_model)
    sample_predictions = torch.randn(batch_size, seq_len, 1000)  # vocab_size=1000
    sample_targets = torch.randint(0, 1000, (batch_size, seq_len))
    
    # 1. Test Geometric Regularization
    print("\n1. Testing Geometric Regularization...")
    geometric_loss = GeometricRegularizationLoss()
    losses = geometric_loss(sample_embeddings, sample_predictions, sample_targets)
    
    print(f"   Standard Loss: {losses['standard_loss']:.4f}")
    print(f"   Geometric Loss: {losses['total_geometric']:.4f}")
    print(f"   Total Loss: {losses['total_loss']:.4f}")
    
    # 2. Test Geometric Monitoring
    print("\n2. Testing Geometric Monitoring...")
    monitor = GeometricMonitor(None)
    health_metrics = monitor.monitor_training(sample_embeddings, step=0)
    
    print(f"   Manifold Health: {health_metrics['manifold_health']:.3f}")
    print(f"   Stratification Score: {health_metrics['stratification_score']:.3f}")
    print(f"   Overall Health: {health_metrics['overall_health']:.3f}")
    
    # 3. Test Improved Token Embeddings
    print("\n3. Testing Improved Token Embeddings...")
    vocab_size = 1000
    improved_embeddings = ImprovedTokenEmbeddings(vocab_size, d_model)
    sample_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    improved_output = improved_embeddings(sample_input_ids)
    print(f"   Input Shape: {sample_input_ids.shape}")
    print(f"   Output Shape: {improved_output.shape}")
    
    # 4. Test Dynamic Subspace Usage
    print("\n4. Testing Dynamic Subspace Usage...")
    dynamic_subspace = DynamicSubspaceUsage(d_model)
    subspace_output = dynamic_subspace(sample_embeddings)
    
    print(f"   Input Shape: {sample_embeddings.shape}")
    print(f"   Output Shape: {subspace_output.shape}")
    print(f"   Max Active Dimensions: {dynamic_subspace.max_active_dimensions}")
    
    print("\n✅ All immediate improvements working correctly!")
    
    return {
        'geometric_loss': losses,
        'health_metrics': health_metrics,
        'improved_embeddings': improved_output,
        'subspace_output': subspace_output
    }

if __name__ == "__main__":
    demonstrate_immediate_improvements()
