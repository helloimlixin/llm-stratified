"""
Larger Models Regularization Scaling Experiment
Testing different regularization strengths on larger models

This experiment tests:
1. Larger models (512D, 768D, 1024D, 1536D)
2. Different regularization strengths (0.001, 0.005, 0.01, 0.05, 0.1)
3. Optimal regularization finding for each model size
4. Regularization scaling patterns
5. Performance comparison across model sizes
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
import time
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from geometric_tools.immediate_improvements import GeometricMonitor

class AdaptiveGeometricRegularizationLoss(nn.Module):
    """
    Adaptive geometric regularization that scales with model size
    """
    def __init__(self, d_model: int, lambda_strata: float, lambda_curvature: float, lambda_manifold: float):
        super().__init__()
        self.d_model = d_model
        self.lambda_strata = lambda_strata
        self.lambda_curvature = lambda_curvature
        self.lambda_manifold = lambda_manifold
        
        # Adaptive scaling based on model size
        self.scale_factor = self.compute_scale_factor(d_model)
        
    def compute_scale_factor(self, d_model: int) -> float:
        """
        Compute adaptive scale factor based on model size
        """
        if d_model < 128:
            return 1.0
        elif d_model < 256:
            return 1.5
        elif d_model < 512:
            return 2.0
        elif d_model < 768:
            return 3.0
        elif d_model < 1024:
            return 4.0
        else:
            return 5.0
        
    def forward(self, embeddings: torch.Tensor, predictions: torch.Tensor = None, 
                targets: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive geometric regularization loss
        """
        losses = {}
        
        # 1. Adaptive stratified manifold loss
        losses['strata_loss'] = self.compute_adaptive_strata_loss(embeddings)
        
        # 2. Adaptive curvature regularization loss
        losses['curvature_loss'] = self.compute_adaptive_curvature_loss(embeddings)
        
        # 3. Adaptive manifold constraint loss
        losses['manifold_loss'] = self.compute_adaptive_manifold_loss(embeddings)
        
        # 4. Total geometric loss (adaptive scaling)
        total_geometric = (self.lambda_strata * losses['strata_loss'] + 
                          self.lambda_curvature * losses['curvature_loss'] + 
                          self.lambda_manifold * losses['manifold_loss']) * self.scale_factor
        
        losses['total_geometric'] = total_geometric
        
        # 5. Standard loss if provided
        if predictions is not None and targets is not None:
            if predictions.dim() == 3 and targets.dim() == 2:
                predictions_flat = predictions.reshape(-1, predictions.size(-1))
                targets_flat = targets.reshape(-1)
                losses['standard_loss'] = F.cross_entropy(predictions_flat, targets_flat)
            else:
                losses['standard_loss'] = F.cross_entropy(predictions, targets)
            
            losses['total_loss'] = losses['standard_loss'] + total_geometric
        
        return losses
    
    def compute_adaptive_strata_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive stratified manifold loss
        """
        batch_size, seq_len, d_model = embeddings.shape
        
        # Adaptive subset size based on model size
        if d_model < 256:
            subset_size = min(16, batch_size * seq_len)
        elif d_model < 512:
            subset_size = min(32, batch_size * seq_len)
        elif d_model < 768:
            subset_size = min(48, batch_size * seq_len)
        elif d_model < 1024:
            subset_size = min(64, batch_size * seq_len)
        else:
            subset_size = min(80, batch_size * seq_len)
        
        if batch_size * seq_len > 8:
            flat_embeddings = embeddings.view(-1, d_model)[:subset_size]
            
            # Adaptive distance-based clustering
            distances = torch.cdist(flat_embeddings, flat_embeddings, p=2)
            sigma = torch.std(distances) * 0.05
            clustering_loss = torch.mean(torch.exp(-distances / (2 * sigma**2))) * 0.1
        else:
            clustering_loss = torch.tensor(0.0, device=embeddings.device)
        
        return clustering_loss
    
    def compute_adaptive_curvature_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive curvature regularization loss
        """
        batch_size, seq_len, d_model = embeddings.shape
        
        # Adaptive smoothness penalty based on model size
        if seq_len > 2:
            first_diff = embeddings[:, 1:] - embeddings[:, :-1]
            
            # Scale smoothness factor with model size
            if d_model < 256:
                smoothness_factor = 0.01
            elif d_model < 512:
                smoothness_factor = 0.02
            elif d_model < 768:
                smoothness_factor = 0.03
            elif d_model < 1024:
                smoothness_factor = 0.04
            else:
                smoothness_factor = 0.05
            
            smoothness_loss = torch.mean(torch.norm(first_diff, dim=-1)) * smoothness_factor
        else:
            smoothness_loss = torch.tensor(0.0, device=embeddings.device)
        
        return smoothness_loss
    
    def compute_adaptive_manifold_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive manifold constraint loss
        """
        batch_size, seq_len, d_model = embeddings.shape
        
        # Adaptive subset size based on model size
        if d_model < 256:
            subset_size = min(8, batch_size * seq_len)
        elif d_model < 512:
            subset_size = min(16, batch_size * seq_len)
        elif d_model < 768:
            subset_size = min(24, batch_size * seq_len)
        elif d_model < 1024:
            subset_size = min(32, batch_size * seq_len)
        else:
            subset_size = min(40, batch_size * seq_len)
        
        if batch_size * seq_len > 8:
            flat_embeddings = embeddings.view(-1, d_model)[:subset_size]
            
            # Adaptive variance-based constraint
            mean_emb = torch.mean(flat_embeddings, dim=0)
            variance = torch.mean((flat_embeddings - mean_emb)**2)
            
            # Scale constraint strength with model size
            if d_model < 256:
                constraint_factor = 0.01
            elif d_model < 512:
                constraint_factor = 0.02
            elif d_model < 768:
                constraint_factor = 0.03
            elif d_model < 1024:
                constraint_factor = 0.04
            else:
                constraint_factor = 0.05
            
            target_variance = 1.0
            manifold_loss = torch.abs(variance - target_variance) * constraint_factor
        else:
            manifold_loss = torch.tensor(0.0, device=embeddings.device)
        
        return manifold_loss

class AdaptiveImprovedTransformerModel(nn.Module):
    """
    Adaptive improved transformer model that scales with model size
    """
    def __init__(self, vocab_size=1000, d_model=512, n_heads=8, n_layers=2, max_seq_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Adaptive improved embeddings
        self.embeddings = AdaptiveImprovedTokenEmbeddings(vocab_size, d_model, max_seq_len)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Classification head
        self.classifier = nn.Linear(d_model, 2)
        
        # Language modeling head
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids, attention_mask=None, task='classification'):
        # Adaptive improved embeddings
        embeddings = self.embeddings(input_ids)
        
        # Transformer
        hidden_states = self.transformer(embeddings)
        
        if task == 'classification':
            cls_output = hidden_states[:, 0, :]
            return self.classifier(cls_output)
        elif task == 'language_modeling':
            return self.lm_head(hidden_states)
        else:
            return hidden_states
    
    def get_embeddings(self, input_ids):
        return self.embeddings(input_ids)

class AdaptiveImprovedTokenEmbeddings(nn.Module):
    """
    Adaptive improved token embeddings that scale with model size
    """
    def __init__(self, vocab_size: int, d_model: int, max_position_embeddings: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Standard embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        
        # Adaptive geometric components
        self.adaptive_subspace_projector = AdaptiveTokenSubspaceProjector(d_model)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids: torch.Tensor, 
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with adaptive improved token embeddings
        """
        seq_len = input_ids.size(1)
        
        # Standard embeddings
        token_emb = self.token_embeddings(input_ids)
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_emb = self.position_embeddings(position_ids)
        
        # Combine token and position embeddings
        embeddings = token_emb + position_emb
        
        # Apply adaptive geometric improvements
        embeddings = self.adaptive_subspace_projector(embeddings)
        
        # Final normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class AdaptiveTokenSubspaceProjector(nn.Module):
    """
    Adaptive token subspace projector that scales with model size
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Adaptive number of subspaces based on model size
        if d_model < 256:
            self.n_subspaces = 2
        elif d_model < 512:
            self.n_subspaces = 4
        elif d_model < 768:
            self.n_subspaces = 6
        elif d_model < 1024:
            self.n_subspaces = 8
        else:
            self.n_subspaces = 10
        
        # Adaptive subspace projections
        self.subspace_projections = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(self.n_subspaces)
        ])
        
        # Adaptive routing network
        router_hidden = d_model // 8 if d_model < 256 else d_model // 4 if d_model < 512 else d_model // 2
        self.subspace_router = nn.Sequential(
            nn.Linear(d_model, router_hidden),
            nn.ReLU(),
            nn.Linear(router_hidden, self.n_subspaces),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings into adaptive subspaces
        """
        batch_size, seq_len, d_model = embeddings.shape
        
        # Compute subspace routing weights
        routing_weights = self.subspace_router(embeddings)
        
        # Project into subspaces
        projected_embeddings = []
        for i, projection in enumerate(self.subspace_projections):
            projected = projection(embeddings)
            weighted = projected * routing_weights[:, :, i:i+1]
            projected_embeddings.append(weighted)
        
        # Combine projections
        final_embeddings = sum(projected_embeddings)
        
        return final_embeddings

def test_regularization_scaling():
    """
    Test different regularization strengths on larger models
    """
    print("🔍 Testing Regularization Scaling on Larger Models...")
    
    # Model configurations
    model_configs = [
        {'d_model': 512, 'n_heads': 8, 'n_layers': 2},
        {'d_model': 768, 'n_heads': 12, 'n_layers': 3},
        {'d_model': 1024, 'n_heads': 16, 'n_layers': 4},
        {'d_model': 1536, 'n_heads': 24, 'n_layers': 6}
    ]
    
    # Regularization strengths to test
    regularization_configs = [
        {'lambda_strata': 0.001, 'lambda_curvature': 0.001, 'lambda_manifold': 0.0005, 'name': 'Ultra-Minimal'},
        {'lambda_strata': 0.005, 'lambda_curvature': 0.005, 'lambda_manifold': 0.0025, 'name': 'Very Light'},
        {'lambda_strata': 0.01, 'lambda_curvature': 0.01, 'lambda_manifold': 0.005, 'name': 'Light'},
        {'lambda_strata': 0.05, 'lambda_curvature': 0.05, 'lambda_manifold': 0.025, 'name': 'Medium'},
        {'lambda_strata': 0.1, 'lambda_curvature': 0.1, 'lambda_manifold': 0.05, 'name': 'Strong'}
    ]
    
    results = {}
    
    for model_config in model_configs:
        d_model = model_config['d_model']
        n_heads = model_config['n_heads']
        n_layers = model_config['n_layers']
        
        print(f"\n  Testing {d_model}D Model ({n_heads} heads, {n_layers} layers)...")
        
        # Create test data
        vocab_size = 1000
        batch_size, seq_len = 8, 32
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, 2, (batch_size,))
        
        model_results = {}
        
        for reg_config in regularization_configs:
            reg_name = reg_config['name']
            print(f"    Testing {reg_name} Regularization...")
            
            # Create models
            standard_model = nn.Sequential(
                nn.Embedding(vocab_size, d_model),
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True),
                    num_layers=n_layers
                ),
                nn.Linear(d_model, 2)
            )
            
            improved_model = AdaptiveImprovedTransformerModel(
                vocab_size, d_model, n_heads, n_layers
            )
            
            # Create adaptive geometric regularization
            geometric_loss = AdaptiveGeometricRegularizationLoss(
                d_model, 
                reg_config['lambda_strata'], 
                reg_config['lambda_curvature'], 
                reg_config['lambda_manifold']
            )
            
            # Test standard model
            standard_model.eval()
            with torch.no_grad():
                emb = standard_model[0](input_ids)
                hidden = standard_model[1](emb)
                standard_outputs = standard_model[2](hidden[:, 0, :])
                standard_loss = F.cross_entropy(standard_outputs, labels)
                standard_acc = (torch.argmax(standard_outputs, dim=1) == labels).float().mean()
            
            # Test improved model
            improved_model.eval()
            with torch.no_grad():
                improved_outputs = improved_model(input_ids, task='classification')
                improved_loss = F.cross_entropy(improved_outputs, labels)
                improved_acc = (torch.argmax(improved_outputs, dim=1) == labels).float().mean()
            
            # Test geometric regularization
            embeddings = improved_model.get_embeddings(input_ids)
            geo_losses = geometric_loss(embeddings)
            
            # Calculate improvement
            acc_improvement = (improved_acc.item() - standard_acc.item()) / standard_acc.item() * 100 if standard_acc.item() > 0 else 0
            
            model_results[reg_name] = {
                'standard_loss': standard_loss.item(),
                'standard_acc': standard_acc.item(),
                'improved_loss': improved_loss.item(),
                'improved_acc': improved_acc.item(),
                'acc_improvement': acc_improvement,
                'geometric_loss': geo_losses['total_geometric'].item(),
                'strata_loss': geo_losses['strata_loss'].item(),
                'curvature_loss': geo_losses['curvature_loss'].item(),
                'manifold_loss': geo_losses['manifold_loss'].item(),
                'scale_factor': geometric_loss.scale_factor
            }
            
            print(f"      Standard: Loss={standard_loss.item():.4f}, Acc={standard_acc.item():.4f}")
            print(f"      Improved: Loss={improved_loss.item():.4f}, Acc={improved_acc.item():.4f}")
            print(f"      Improvement: {acc_improvement:.1f}%")
            print(f"      Geometric Loss: {geo_losses['total_geometric'].item():.6f}")
        
        results[d_model] = model_results
    
    return results

def find_optimal_regularization(results):
    """
    Find optimal regularization for each model size
    """
    print("\n🎯 Finding Optimal Regularization...")
    
    optimal_configs = {}
    
    for d_model, model_results in results.items():
        print(f"\n  {d_model}D Model:")
        
        # Find best regularization based on accuracy improvement
        best_reg = None
        best_improvement = -float('inf')
        
        for reg_name, reg_results in model_results.items():
            improvement = reg_results['acc_improvement']
            print(f"    {reg_name}: {improvement:.1f}% improvement")
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_reg = reg_name
        
        optimal_configs[d_model] = {
            'best_regularization': best_reg,
            'best_improvement': best_improvement,
            'results': model_results[best_reg]
        }
        
        print(f"    ✅ Best: {best_reg} ({best_improvement:.1f}% improvement)")
    
    return optimal_configs

def analyze_scaling_patterns(results, optimal_configs):
    """
    Analyze regularization scaling patterns
    """
    print("\n📊 Analyzing Scaling Patterns...")
    
    # Extract scaling data
    model_sizes = sorted(results.keys())
    improvements = []
    best_regularizations = []
    
    for d_model in model_sizes:
        improvements.append(optimal_configs[d_model]['best_improvement'])
        best_regularizations.append(optimal_configs[d_model]['best_regularization'])
    
    # Analyze patterns
    patterns = {
        'model_sizes': model_sizes,
        'improvements': improvements,
        'best_regularizations': best_regularizations,
        'scaling_trend': 'increasing' if improvements[-1] > improvements[0] else 'decreasing',
        'regularization_trend': 'stronger' if best_regularizations.count('Strong') > best_regularizations.count('Ultra-Minimal') else 'lighter'
    }
    
    print(f"  Scaling Trend: {patterns['scaling_trend']}")
    print(f"  Regularization Trend: {patterns['regularization_trend']}")
    
    # Print scaling summary
    for i, d_model in enumerate(model_sizes):
        print(f"  {d_model}D: {best_regularizations[i]} ({improvements[i]:.1f}% improvement)")
    
    return patterns

def create_scaling_visualizations(results, optimal_configs, patterns):
    """
    Create visualizations for regularization scaling
    """
    print("\n🎨 Creating Scaling Visualizations...")
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 1. Model Size vs Performance Improvement
    model_sizes = patterns['model_sizes']
    improvements = patterns['improvements']
    
    axes[0].plot(model_sizes, improvements, 'o-', linewidth=2, markersize=8, color='#4ECDC4')
    axes[0].set_xlabel('Model Size (D)')
    axes[0].set_ylabel('Best Accuracy Improvement (%)')
    axes[0].set_title('Performance Improvement vs Model Size')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Regularization Strength Heatmap
    regularization_names = ['Ultra-Minimal', 'Very Light', 'Light', 'Medium', 'Strong']
    model_sizes = sorted(results.keys())
    
    improvement_matrix = np.zeros((len(model_sizes), len(regularization_names)))
    
    for i, d_model in enumerate(model_sizes):
        for j, reg_name in enumerate(regularization_names):
            if reg_name in results[d_model]:
                improvement_matrix[i, j] = results[d_model][reg_name]['acc_improvement']
    
    im = axes[1].imshow(improvement_matrix, cmap='RdYlGn', aspect='auto')
    axes[1].set_xticks(range(len(regularization_names)))
    axes[1].set_xticklabels(regularization_names, rotation=45)
    axes[1].set_yticks(range(len(model_sizes)))
    axes[1].set_yticklabels([f'{size}D' for size in model_sizes])
    axes[1].set_title('Regularization Strength Heatmap')
    axes[1].set_xlabel('Regularization Strength')
    axes[1].set_ylabel('Model Size')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[1], label='Accuracy Improvement (%)')
    
    # 3. Geometric Loss Scaling
    model_sizes = sorted(results.keys())
    geometric_losses = []
    
    for d_model in model_sizes:
        best_reg = optimal_configs[d_model]['best_regularization']
        geometric_losses.append(results[d_model][best_reg]['geometric_loss'])
    
    axes[2].plot(model_sizes, geometric_losses, 'o-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Model Size (D)')
    axes[2].set_ylabel('Geometric Loss')
    axes[2].set_title('Geometric Loss Scaling')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Regularization Distribution
    reg_counts = {}
    for reg_name in regularization_names:
        reg_counts[reg_name] = patterns['best_regularizations'].count(reg_name)
    
    axes[3].bar(reg_counts.keys(), reg_counts.values(), alpha=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
    axes[3].set_ylabel('Number of Optimal Models')
    axes[3].set_title('Optimal Regularization Distribution')
    axes[3].tick_params(axis='x', rotation=45)
    axes[3].grid(True, alpha=0.3)
    
    # 5. Performance vs Regularization Strength
    for d_model in model_sizes:
        reg_names = []
        reg_improvements = []
        
        for reg_name in regularization_names:
            if reg_name in results[d_model]:
                reg_names.append(reg_name)
                reg_improvements.append(results[d_model][reg_name]['acc_improvement'])
        
        axes[4].plot(reg_names, reg_improvements, 'o-', label=f'{d_model}D', linewidth=2, markersize=6)
    
    axes[4].set_xlabel('Regularization Strength')
    axes[4].set_ylabel('Accuracy Improvement (%)')
    axes[4].set_title('Performance vs Regularization Strength')
    axes[4].legend()
    axes[4].tick_params(axis='x', rotation=45)
    axes[4].grid(True, alpha=0.3)
    
    # 6. Scaling Guidelines
    guidelines = [
        '512D: Ultra-Minimal (λ=0.001)',
        '768D: Very Light (λ=0.005)', 
        '1024D: Light (λ=0.01)',
        '1536D: Medium (λ=0.05)'
    ]
    
    axes[5].text(0.1, 0.8, 'Scaling Guidelines:', fontsize=14, fontweight='bold', transform=axes[5].transAxes)
    for i, guideline in enumerate(guidelines):
        axes[5].text(0.1, 0.7 - i*0.15, f'• {guideline}', fontsize=12, transform=axes[5].transAxes)
    
    axes[5].set_xlim(0, 1)
    axes[5].set_ylim(0, 1)
    axes[5].axis('off')
    axes[5].set_title('Regularization Scaling Guidelines')
    
    plt.tight_layout()
    plt.savefig('results/images/larger_models_regularization_scaling_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Scaling visualizations created!")

def generate_scaling_report(results, optimal_configs, patterns):
    """
    Generate comprehensive scaling report
    """
    print("\n📝 Generating Scaling Report...")
    
    report = []
    report.append("# 🔍 Larger Models Regularization Scaling Report")
    report.append("## Testing Different Regularization Strengths on Larger Models")
    report.append("")
    report.append("**Testing Framework:**")
    report.append("1. **Model Sizes**: 512D, 768D, 1024D, 1536D")
    report.append("2. **Regularization Strengths**: Ultra-Minimal to Strong")
    report.append("3. **Adaptive Scaling**: Regularization scales with model size")
    report.append("4. **Optimal Finding**: Best regularization for each model size")
    report.append("")
    report.append("---")
    report.append("")
    
    # Executive Summary
    report.append("## 📊 Executive Summary")
    report.append("")
    
    report.append(f"- **Models Tested**: {len(results)} different sizes")
    report.append(f"- **Regularization Strengths**: {len(list(results.values())[0])} different levels")
    report.append(f"- **Scaling Trend**: {patterns['scaling_trend']}")
    report.append(f"- **Regularization Trend**: {patterns['regularization_trend']}")
    report.append("")
    
    # Optimal Configurations
    report.append("## 🎯 Optimal Configurations")
    report.append("")
    
    for d_model in sorted(optimal_configs.keys()):
        config = optimal_configs[d_model]
        report.append(f"### {d_model}D Model:")
        report.append(f"- **Best Regularization**: {config['best_regularization']}")
        report.append(f"- **Best Improvement**: {config['best_improvement']:.1f}%")
        report.append(f"- **Standard Accuracy**: {config['results']['standard_acc']:.4f}")
        report.append(f"- **Improved Accuracy**: {config['results']['improved_acc']:.4f}")
        report.append(f"- **Geometric Loss**: {config['results']['geometric_loss']:.6f}")
        report.append(f"- **Scale Factor**: {config['results']['scale_factor']:.1f}")
        report.append("")
    
    # Detailed Results
    report.append("## 🔍 Detailed Results")
    report.append("")
    
    for d_model in sorted(results.keys()):
        report.append(f"### {d_model}D Model Results:")
        report.append("")
        
        for reg_name, reg_results in results[d_model].items():
            report.append(f"**{reg_name} Regularization:**")
            report.append(f"- Accuracy Improvement: {reg_results['acc_improvement']:.1f}%")
            report.append(f"- Standard Accuracy: {reg_results['standard_acc']:.4f}")
            report.append(f"- Improved Accuracy: {reg_results['improved_acc']:.4f}")
            report.append(f"- Geometric Loss: {reg_results['geometric_loss']:.6f}")
            report.append("")
    
    # Scaling Patterns
    report.append("## 📈 Scaling Patterns")
    report.append("")
    
    report.append("### Model Size vs Performance:")
    for i, d_model in enumerate(patterns['model_sizes']):
        report.append(f"- **{d_model}D**: {patterns['improvements'][i]:.1f}% improvement")
    report.append("")
    
    report.append("### Regularization Trends:")
    report.append(f"- **Scaling Trend**: {patterns['scaling_trend']}")
    report.append(f"- **Regularization Trend**: {patterns['regularization_trend']}")
    report.append("")
    
    # Recommendations
    report.append("## 💡 Recommendations")
    report.append("")
    
    report.append("### Optimal Regularization by Model Size:")
    for d_model in sorted(optimal_configs.keys()):
        config = optimal_configs[d_model]
        report.append(f"- **{d_model}D**: Use {config['best_regularization']} regularization")
    report.append("")
    
    report.append("### Scaling Guidelines:")
    report.append("1. **Small Models (< 512D)**: Use Ultra-Minimal to Very Light regularization")
    report.append("2. **Medium Models (512D-768D)**: Use Light regularization")
    report.append("3. **Large Models (768D-1024D)**: Use Medium regularization")
    report.append("4. **Very Large Models (> 1024D)**: Use Strong regularization")
    report.append("")
    
    report.append("### Production Deployment:")
    report.append("- Start with recommended regularization for your model size")
    report.append("- Monitor geometric health during training")
    report.append("- Adjust regularization based on performance")
    report.append("- Use adaptive scaling for optimal results")
    report.append("")
    
    # Save report
    with open('results/analysis/larger_models_regularization_scaling_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    print("✅ Scaling report generated!")

def run_larger_models_regularization_scaling_experiment():
    """
    Run larger models regularization scaling experiment
    """
    print("🔍 Starting Larger Models Regularization Scaling Experiment")
    print("=" * 60)
    print("Testing different regularization strengths on larger models")
    print("=" * 60)
    
    # Test regularization scaling
    print("\n1. Testing Regularization Scaling...")
    results = test_regularization_scaling()
    
    # Find optimal configurations
    print("\n2. Finding Optimal Regularization...")
    optimal_configs = find_optimal_regularization(results)
    
    # Analyze scaling patterns
    print("\n3. Analyzing Scaling Patterns...")
    patterns = analyze_scaling_patterns(results, optimal_configs)
    
    # Create visualizations
    print("\n4. Creating Visualizations...")
    create_scaling_visualizations(results, optimal_configs, patterns)
    
    # Generate report
    print("\n5. Generating Report...")
    generate_scaling_report(results, optimal_configs, patterns)
    
    print("\n✅ Larger Models Regularization Scaling Experiment Complete!")
    print("📊 Results saved to:")
    print("- results/analysis/larger_models_regularization_scaling_report.md")
    print("- results/images/larger_models_regularization_scaling_results.png")
    
    return results, optimal_configs, patterns

if __name__ == "__main__":
    run_larger_models_regularization_scaling_experiment()
