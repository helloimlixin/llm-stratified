"""
Ultra-Minimal Regularization Experiment
Testing extremely light geometric regularization for small models

Based on findings:
- Even λ=0.01 is too aggressive for small models
- Need ultra-minimal regularization (λ=0.001-0.005)
- Focus on essential components only
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
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from geometric_tools.immediate_improvements import GeometricMonitor

class UltraMinimalGeometricRegularizationLoss(nn.Module):
    """
    Ultra-minimal geometric regularization for small models
    """
    def __init__(self, lambda_strata=0.001, lambda_curvature=0.001, lambda_manifold=0.0005):
        super().__init__()
        self.lambda_strata = lambda_strata
        self.lambda_curvature = lambda_curvature
        self.lambda_manifold = lambda_manifold
        
    def forward(self, embeddings: torch.Tensor, predictions: torch.Tensor = None, 
                targets: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute ultra-minimal geometric regularization loss
        """
        losses = {}
        
        # 1. Ultra-minimal stratified manifold loss
        losses['strata_loss'] = self.compute_minimal_strata_loss(embeddings)
        
        # 2. Ultra-minimal curvature regularization loss
        losses['curvature_loss'] = self.compute_minimal_curvature_loss(embeddings)
        
        # 3. Ultra-minimal manifold constraint loss
        losses['manifold_loss'] = self.compute_minimal_manifold_loss(embeddings)
        
        # 4. Total geometric loss (ultra-minimal)
        total_geometric = (self.lambda_strata * losses['strata_loss'] + 
                          self.lambda_curvature * losses['curvature_loss'] + 
                          self.lambda_manifold * losses['manifold_loss'])
        
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
    
    def compute_minimal_strata_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute ultra-minimal stratified manifold loss
        """
        batch_size, seq_len, d_model = embeddings.shape
        
        # Minimal clustering loss - just encourage tiny bit of structure
        if batch_size * seq_len > 8:
            # Use very small subset for efficiency
            subset_size = min(16, batch_size * seq_len)
            flat_embeddings = embeddings.view(-1, d_model)[:subset_size]
            
            # Minimal distance-based clustering
            distances = torch.cdist(flat_embeddings, flat_embeddings, p=2)
            sigma = torch.std(distances) * 0.05  # Very small sigma
            clustering_loss = torch.mean(torch.exp(-distances / (2 * sigma**2))) * 0.1
        else:
            clustering_loss = torch.tensor(0.0, device=embeddings.device)
        
        return clustering_loss
    
    def compute_minimal_curvature_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute ultra-minimal curvature regularization loss
        """
        batch_size, seq_len, d_model = embeddings.shape
        
        # Minimal smoothness penalty
        if seq_len > 2:
            first_diff = embeddings[:, 1:] - embeddings[:, :-1]
            smoothness_loss = torch.mean(torch.norm(first_diff, dim=-1)) * 0.01
        else:
            smoothness_loss = torch.tensor(0.0, device=embeddings.device)
        
        return smoothness_loss
    
    def compute_minimal_manifold_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute ultra-minimal manifold constraint loss
        """
        batch_size, seq_len, d_model = embeddings.shape
        
        # Minimal manifold constraint
        if batch_size * seq_len > 8:
            subset_size = min(8, batch_size * seq_len)
            flat_embeddings = embeddings.view(-1, d_model)[:subset_size]
            
            # Minimal variance-based constraint
            mean_emb = torch.mean(flat_embeddings, dim=0)
            variance = torch.mean((flat_embeddings - mean_emb)**2)
            
            # Very gentle variance constraint
            target_variance = 1.0
            manifold_loss = torch.abs(variance - target_variance) * 0.01
        else:
            manifold_loss = torch.tensor(0.0, device=embeddings.device)
        
        return manifold_loss

class UltraMinimalImprovedTokenEmbeddings(nn.Module):
    """
    Ultra-minimal improved token embeddings for small models
    """
    def __init__(self, vocab_size: int, d_model: int, max_position_embeddings: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Standard embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        
        # Ultra-minimal geometric components
        self.minimal_subspace_projector = MinimalTokenSubspaceProjector(d_model)
        
        # Light normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.05)  # Very light dropout
        
    def forward(self, input_ids: torch.Tensor, 
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with ultra-minimal improved token embeddings
        """
        seq_len = input_ids.size(1)
        
        # Standard embeddings
        token_emb = self.token_embeddings(input_ids)
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_emb = self.position_embeddings(position_ids)
        
        # Combine token and position embeddings
        embeddings = token_emb + position_emb
        
        # Apply ultra-minimal geometric improvements
        embeddings = self.minimal_subspace_projector(embeddings)
        
        # Final normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class MinimalTokenSubspaceProjector(nn.Module):
    """
    Ultra-minimal token subspace projector
    """
    def __init__(self, d_model: int, n_subspaces: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_subspaces = n_subspaces
        
        # Minimal subspace projections
        self.subspace_projections = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_subspaces)
        ])
        
        # Ultra-lightweight routing
        self.subspace_router = nn.Sequential(
            nn.Linear(d_model, d_model // 8),  # Much smaller
            nn.ReLU(),
            nn.Linear(d_model // 8, n_subspaces),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings into minimal subspaces
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

class UltraMinimalImprovedTransformerModel(nn.Module):
    """
    Ultra-minimal improved transformer model with minimal geometric enhancements
    """
    def __init__(self, vocab_size=1000, d_model=128, n_heads=4, n_layers=1, max_seq_len=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Ultra-minimal improved embeddings
        self.embeddings = UltraMinimalImprovedTokenEmbeddings(vocab_size, d_model, max_seq_len)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Classification head
        self.classifier = nn.Linear(d_model, 2)
        
        # Language modeling head
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids, attention_mask=None, task='classification'):
        # Ultra-minimal improved embeddings
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

def create_synthetic_classification_data():
    """
    Create synthetic classification data for testing
    """
    texts = [
        "This movie is absolutely fantastic and amazing!",
        "I hate this terrible product, it's awful.",
        "The weather today is nice and sunny.",
        "This book is boring and uninteresting.",
        "I love spending time with my family.",
        "The traffic is horrible and frustrating.",
        "This restaurant has delicious food.",
        "The service was poor and disappointing.",
        "I enjoy reading books in my free time.",
        "This game is fun and entertaining.",
        "The movie was okay, nothing special.",
        "I dislike waiting in long lines.",
        "This music is beautiful and relaxing.",
        "The food tastes bad and salty.",
        "I appreciate good customer service.",
        "This place is dirty and unpleasant.",
        "The sunset looks gorgeous tonight.",
        "I'm frustrated with this situation.",
        "This coffee tastes perfect.",
        "The noise is annoying and loud."
    ] * 25  # Smaller dataset for efficiency
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 25
    
    return {
        'texts': texts,
        'labels': labels,
        'num_classes': 2
    }

def create_tokenizer_and_vocab(datasets):
    """
    Create a simple tokenizer and vocabulary from datasets
    """
    print("🔤 Creating tokenizer and vocabulary...")
    
    # Collect all text
    all_texts = []
    for dataset_name, dataset in datasets.items():
        if isinstance(dataset, dict):
            if 'texts' in dataset:
                all_texts.extend(dataset['texts'])
    
    # Create vocabulary
    all_words = []
    for text in all_texts:
        words = text.lower().split()
        all_words.extend(words)
    
    # Get unique words and create vocab
    unique_words = list(set(all_words))
    vocab_size = min(100, len(unique_words))  # Even smaller vocab
    
    # Create word to id mapping
    word_to_id = {word: i for i, word in enumerate(unique_words[:vocab_size])}
    word_to_id['<PAD>'] = vocab_size
    word_to_id['<UNK>'] = vocab_size + 1
    vocab_size += 2
    
    print(f"  ✅ Created vocabulary with {vocab_size} tokens")
    
    def tokenize(text, max_length=16):  # Even shorter sequences
        words = text.lower().split()[:max_length]
        token_ids = [word_to_id.get(word, word_to_id['<UNK>']) for word in words]
        
        # Pad to max_length
        while len(token_ids) < max_length:
            token_ids.append(word_to_id['<PAD>'])
        
        return torch.tensor(token_ids[:max_length])
    
    return tokenize, vocab_size

def test_ultra_minimal_regularization():
    """
    Test ultra-minimal regularization on classification task
    """
    print("🪶 Testing Ultra-Minimal Regularization...")
    
    # Create datasets
    datasets = {'synthetic_classification': create_synthetic_classification_data()}
    
    # Create tokenizer
    tokenizer, vocab_size = create_tokenizer_and_vocab(datasets)
    
    # Prepare data
    data = datasets['synthetic_classification']
    texts = data['texts'][:100]  # Even smaller dataset
    labels = data['labels'][:100]
    
    # Tokenize data
    input_ids = torch.stack([tokenizer(text) for text in texts])
    labels_tensor = torch.tensor(labels)
    
    # Create models
    standard_model = nn.Sequential(
        nn.Embedding(vocab_size, 64),  # Smaller model
        nn.TransformerEncoder(
            nn.TransformerEncoderLayer(64, 2, batch_first=True),  # Smaller
            num_layers=1
        ),
        nn.Linear(64, 2)
    )
    
    improved_model = UltraMinimalImprovedTransformerModel(vocab_size, d_model=64, n_heads=2, n_layers=1)
    
    # Create optimizers
    standard_optimizer = torch.optim.Adam(standard_model.parameters(), lr=1e-3)
    improved_optimizer = torch.optim.Adam(improved_model.parameters(), lr=1e-3)
    
    # Create ultra-minimal geometric components
    ultra_minimal_loss = UltraMinimalGeometricRegularizationLoss(lambda_strata=0.001, lambda_curvature=0.001)
    monitor = GeometricMonitor(improved_model)
    
    # Training parameters
    num_epochs = 3
    batch_size = 8  # Smaller batches
    num_batches = len(input_ids) // batch_size
    
    results = {
        'standard': {'losses': [], 'accuracies': [], 'times': []},
        'improved': {'losses': [], 'accuracies': [], 'times': [], 'health_metrics': []}
    }
    
    print(f"  Training for {num_epochs} epochs with {num_batches} batches per epoch")
    
    for epoch in range(num_epochs):
        print(f"    Epoch {epoch + 1}/{num_epochs}")
        
        # Standard model training
        standard_model.train()
        epoch_losses_std = []
        epoch_accuracies_std = []
        
        start_time = time.time()
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch_inputs = input_ids[start_idx:end_idx]
            batch_labels = labels_tensor[start_idx:end_idx]
            
            # Forward pass
            standard_optimizer.zero_grad()
            
            # Simple forward pass for standard model
            emb = standard_model[0](batch_inputs)
            hidden = standard_model[1](emb)
            outputs = standard_model[2](hidden[:, 0, :])  # Use first token
            
            loss = F.cross_entropy(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            standard_optimizer.step()
            
            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == batch_labels).float().mean()
            
            epoch_losses_std.append(loss.item())
            epoch_accuracies_std.append(accuracy.item())
        
        std_time = time.time() - start_time
        
        # Improved model training
        improved_model.train()
        epoch_losses_imp = []
        epoch_accuracies_imp = []
        epoch_health_metrics = []
        
        start_time = time.time()
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch_inputs = input_ids[start_idx:end_idx]
            batch_labels = labels_tensor[start_idx:end_idx]
            
            # Forward pass
            improved_optimizer.zero_grad()
            outputs = improved_model(batch_inputs, task='classification')
            
            # Get embeddings for geometric analysis
            embeddings = improved_model.get_embeddings(batch_inputs)
            
            # Compute losses
            standard_loss = F.cross_entropy(outputs, batch_labels)
            geometric_losses = ultra_minimal_loss(embeddings, outputs, batch_labels)
            total_loss = geometric_losses['total_loss']
            
            # Monitor geometric health
            health_metrics = monitor.monitor_training(embeddings, step=epoch * num_batches + batch_idx)
            
            # Backward pass
            total_loss.backward()
            improved_optimizer.step()
            
            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == batch_labels).float().mean()
            
            epoch_losses_imp.append(total_loss.item())
            epoch_accuracies_imp.append(accuracy.item())
            epoch_health_metrics.append(health_metrics['overall_health'])
        
        imp_time = time.time() - start_time
        
        # Store results
        results['standard']['losses'].append(np.mean(epoch_losses_std))
        results['standard']['accuracies'].append(np.mean(epoch_accuracies_std))
        results['standard']['times'].append(std_time)
        
        results['improved']['losses'].append(np.mean(epoch_losses_imp))
        results['improved']['accuracies'].append(np.mean(epoch_accuracies_imp))
        results['improved']['times'].append(imp_time)
        results['improved']['health_metrics'].append(np.mean(epoch_health_metrics))
        
        print(f"      Standard - Loss: {np.mean(epoch_losses_std):.4f}, Acc: {np.mean(epoch_accuracies_std):.4f}, Time: {std_time:.2f}s")
        print(f"      Improved - Loss: {np.mean(epoch_losses_imp):.4f}, Acc: {np.mean(epoch_accuracies_imp):.4f}, Time: {imp_time:.2f}s")
        print(f"      Geometric Health: {np.mean(epoch_health_metrics):.3f}")
    
    return results

def create_ultra_minimal_visualizations(results):
    """
    Create visualizations for ultra-minimal regularization results
    """
    print("\n🎨 Creating Ultra-Minimal Visualizations...")
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # 1. Training Loss Comparison
    epochs = range(1, len(results['standard']['losses']) + 1)
    
    axes[0].plot(epochs, results['standard']['losses'], 'o-', 
                label='Standard Model', linewidth=2, markersize=8)
    axes[0].plot(epochs, results['improved']['losses'], 's-', 
                label='Ultra-Minimal Improved Model', linewidth=2, markersize=8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Comparison (Ultra-Minimal Regularization)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Accuracy Comparison
    axes[1].plot(epochs, results['standard']['accuracies'], 'o-', 
                label='Standard Model', linewidth=2, markersize=8)
    axes[1].plot(epochs, results['improved']['accuracies'], 's-', 
                label='Ultra-Minimal Improved Model', linewidth=2, markersize=8)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Comparison (Ultra-Minimal Regularization)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Training Time Comparison
    std_times = results['standard']['times']
    imp_times = results['improved']['times']
    
    axes[2].bar(['Standard Model', 'Ultra-Minimal Improved Model'], 
               [np.mean(std_times), np.mean(imp_times)], 
               alpha=0.8, color=['#FF6B6B', '#4ECDC4'])
    axes[2].set_ylabel('Training Time (seconds)')
    axes[2].set_title('Training Time Comparison')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Geometric Health Monitoring
    health_metrics = results['improved']['health_metrics']
    
    axes[3].plot(epochs, health_metrics, 'o-', color='green', linewidth=2, markersize=8)
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Geometric Health Score')
    axes[3].set_title('Geometric Health During Training')
    axes[3].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Health Threshold')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/images/ultra_minimal_regularization_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Ultra-minimal visualizations created!")

def generate_ultra_minimal_report(results):
    """
    Generate report for ultra-minimal regularization results
    """
    print("\n📝 Generating Ultra-Minimal Report...")
    
    report = []
    report.append("# 🪶 Ultra-Minimal Regularization Report")
    report.append("## Extremely Light Geometric Regularization for Small Models")
    report.append("")
    report.append("**Based on Previous Findings:**")
    report.append("- Even λ=0.01 is too aggressive for small models")
    report.append("- Need ultra-minimal regularization (λ=0.001-0.005)")
    report.append("- Focus on essential components only")
    report.append("")
    report.append("---")
    report.append("")
    
    # Executive Summary
    report.append("## 📊 Executive Summary")
    report.append("")
    
    # Calculate improvements
    final_loss_std = results['standard']['losses'][-1]
    final_loss_imp = results['improved']['losses'][-1]
    loss_improvement = (final_loss_std - final_loss_imp) / final_loss_std * 100
    
    final_acc_std = results['standard']['accuracies'][-1]
    final_acc_imp = results['improved']['accuracies'][-1]
    acc_improvement = (final_acc_imp - final_acc_std) / final_acc_std * 100
    
    avg_time_std = np.mean(results['standard']['times'])
    avg_time_imp = np.mean(results['improved']['times'])
    time_ratio = avg_time_imp / avg_time_std
    
    avg_health = np.mean(results['improved']['health_metrics'])
    
    report.append(f"- **Final Loss Improvement**: {loss_improvement:.1f}%")
    report.append(f"- **Final Accuracy Improvement**: {acc_improvement:.1f}%")
    report.append(f"- **Training Time Ratio**: {time_ratio:.2f}x")
    report.append(f"- **Average Geometric Health**: {avg_health:.3f}")
    report.append("")
    
    # Detailed Results
    report.append("## 🔍 Detailed Results")
    report.append("")
    
    report.append("### Final Performance Comparison")
    report.append(f"- **Standard Model**: Loss = {final_loss_std:.4f}, Accuracy = {final_acc_std:.4f}")
    report.append(f"- **Ultra-Minimal Improved Model**: Loss = {final_loss_imp:.4f}, Accuracy = {final_acc_imp:.4f}")
    report.append("")
    
    report.append("### Training Efficiency")
    report.append(f"- **Standard Model**: {avg_time_std:.2f}s per epoch")
    report.append(f"- **Ultra-Minimal Improved Model**: {avg_time_imp:.2f}s per epoch")
    report.append(f"- **Time Overhead**: {time_ratio:.2f}x")
    report.append("")
    
    report.append("### Geometric Health")
    report.append(f"- **Average Health Score**: {avg_health:.3f}")
    report.append(f"- **Health Status**: {'✅ Healthy' if avg_health > 0.5 else '⚠️ Needs Attention'}")
    report.append("")
    
    # Key Findings
    report.append("## 🔍 Key Findings")
    report.append("")
    
    if loss_improvement > 0 and acc_improvement > 0:
        report.append("### ✅ Ultra-Minimal Regularization Works!")
        report.append(f"- **Better Loss**: {loss_improvement:.1f}% improvement in final loss")
        report.append(f"- **Better Accuracy**: {acc_improvement:.1f}% improvement in accuracy")
        report.append(f"- **Computational Overhead**: {time_ratio:.2f}x slower training")
        report.append(f"- **Geometric Health**: {avg_health:.3f} (monitored successfully)")
    else:
        report.append("### ⚠️ Still Needs Further Tuning")
        if loss_improvement < 0:
            report.append(f"- **Loss**: {abs(loss_improvement):.1f}% worse performance")
        if acc_improvement < 0:
            report.append(f"- **Accuracy**: {abs(acc_improvement):.1f}% worse performance")
        report.append(f"- **Computational Overhead**: {time_ratio:.2f}x slower training")
        report.append(f"- **Geometric Health**: {avg_health:.3f} (monitored successfully)")
    
    report.append("")
    
    # Recommendations
    report.append("## 💡 Recommendations")
    report.append("")
    
    if loss_improvement > 0 and acc_improvement > 0:
        report.append("### ✅ Ultra-Minimal Regularization Success!")
        report.append("- Continue with λ_strata=0.001, λ_curvature=0.001")
        report.append("- Monitor geometric health during training")
        report.append("- Consider slightly increasing regularization if performance improves")
        report.append("- This level works well for very small models")
    else:
        report.append("### ⚠️ Even Lighter Regularization Needed")
        report.append("- Try λ_strata=0.0005, λ_curvature=0.0005")
        report.append("- Consider removing some geometric components entirely")
        report.append("- Focus only on most essential geometric improvements")
        report.append("- May need task-specific tuning")
    
    report.append("")
    report.append("### For Very Small Models (< 128D):")
    report.append("- Use λ_strata=0.001, λ_curvature=0.001, λ_manifold=0.0005")
    report.append("- Implement only essential geometric components")
    report.append("- Monitor computational overhead closely")
    report.append("- Focus on minimal geometric improvements")
    report.append("")
    
    # Save report
    with open('results/analysis/ultra_minimal_regularization_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    print("✅ Ultra-minimal report generated!")

def run_ultra_minimal_regularization_experiment():
    """
    Run ultra-minimal regularization experiment
    """
    print("🪶 Starting Ultra-Minimal Regularization Experiment")
    print("=" * 60)
    print("Testing extremely light geometric regularization for small models")
    print("=" * 60)
    
    # Test ultra-minimal regularization
    print("\n1. Testing Ultra-Minimal Regularization...")
    results = test_ultra_minimal_regularization()
    
    # Create visualizations
    print("\n2. Creating Visualizations...")
    create_ultra_minimal_visualizations(results)
    
    # Generate report
    print("\n3. Generating Report...")
    generate_ultra_minimal_report(results)
    
    print("\n✅ Ultra-Minimal Regularization Experiment Complete!")
    print("📊 Results saved to:")
    print("- results/analysis/ultra_minimal_regularization_report.md")
    print("- results/images/ultra_minimal_regularization_results.png")
    
    return results

if __name__ == "__main__":
    run_ultra_minimal_regularization_experiment()
