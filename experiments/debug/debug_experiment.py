"""
Debug Experiment - Why No Improvements?
Investigating why geometric regularization isn't showing improvements

This experiment will:
1. Test with very simple models
2. Verify geometric regularization is actually working
3. Check if the loss components are meaningful
4. Test with known good configurations
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def create_simple_test_data():
    """Create very simple test data"""
    print("📊 Creating Simple Test Data...")
    
    # Very simple binary classification
    texts = [
        "positive good great excellent amazing wonderful",
        "negative bad terrible awful horrible disgusting",
        "positive nice lovely beautiful fantastic superb",
        "negative ugly horrible terrible awful disgusting",
        "positive happy joyful cheerful delightful pleasant",
        "negative sad angry frustrated disappointed upset"
    ] * 50  # Repeat to get more data
    
    labels = [1, 0, 1, 0, 1, 0] * 50
    
    # Simple tokenizer
    vocab = {
        'positive': 0, 'good': 1, 'great': 2, 'excellent': 3, 'amazing': 4, 'wonderful': 5,
        'negative': 6, 'bad': 7, 'terrible': 8, 'awful': 9, 'horrible': 10, 'disgusting': 11,
        'nice': 12, 'lovely': 13, 'beautiful': 14, 'fantastic': 15, 'superb': 16,
        'ugly': 17, 'sad': 18, 'angry': 19, 'frustrated': 20, 'disappointed': 21, 'upset': 22,
        'happy': 23, 'joyful': 24, 'cheerful': 25, 'delightful': 26, 'pleasant': 27,
        '<PAD>': 28, '<UNK>': 29
    }
    
    def tokenize(text, max_len=10):
        words = text.split()[:max_len]
        ids = [vocab.get(word, vocab['<UNK>']) for word in words]
        while len(ids) < max_len:
            ids.append(vocab['<PAD>'])
        return torch.tensor(ids[:max_len])
    
    input_ids = torch.stack([tokenize(text) for text in texts])
    labels_tensor = torch.tensor(labels)
    
    print(f"  ✅ Created {len(texts)} samples")
    print(f"  ✅ Vocabulary size: {len(vocab)}")
    print(f"  ✅ Input shape: {input_ids.shape}")
    print(f"  ✅ Labels: {labels_tensor[:10].tolist()}")
    
    return input_ids, labels_tensor, len(vocab)

class SimpleStandardModel(nn.Module):
    """Very simple standard model"""
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True),
            num_layers=2
        )
        self.classifier = nn.Linear(d_model, 2)
        
    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        hidden = self.transformer(emb)
        cls_output = hidden[:, 0, :]  # Use first token
        return self.classifier(cls_output)

class SimpleImprovedModel(nn.Module):
    """Very simple improved model with geometric regularization"""
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True),
            num_layers=2
        )
        self.classifier = nn.Linear(d_model, 2)
        
        # Add a simple geometric component
        self.geometric_layer = nn.Linear(d_model, d_model)
        
    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        
        # Apply geometric transformation
        geometric_emb = self.geometric_layer(emb)
        emb = emb + 0.1 * geometric_emb  # Small geometric influence
        
        hidden = self.transformer(emb)
        cls_output = hidden[:, 0, :]
        return self.classifier(cls_output)
    
    def get_embeddings(self, input_ids):
        emb = self.embedding(input_ids)
        geometric_emb = self.geometric_layer(emb)
        return emb + 0.1 * geometric_emb

class SimpleGeometricLoss(nn.Module):
    """Very simple geometric loss for debugging"""
    def __init__(self, lambda_geo=0.1):
        super().__init__()
        self.lambda_geo = lambda_geo
        
    def forward(self, embeddings, predictions=None, targets=None):
        losses = {}
        
        # Simple geometric loss: encourage embeddings to be well-separated
        batch_size, seq_len, d_model = embeddings.shape
        
        # Flatten embeddings
        flat_emb = embeddings.view(-1, d_model)
        
        # Compute pairwise distances
        distances = torch.cdist(flat_emb, flat_emb, p=2)
        
        # Geometric loss: encourage some structure
        # This should be meaningful for the simple data
        geometric_loss = torch.mean(torch.exp(-distances)) * 0.01
        
        losses['geometric_loss'] = geometric_loss
        losses['total_geometric'] = self.lambda_geo * geometric_loss
        
        if predictions is not None and targets is not None:
            standard_loss = F.cross_entropy(predictions, targets)
            losses['standard_loss'] = standard_loss
            losses['total_loss'] = standard_loss + losses['total_geometric']
        
        return losses

def test_simple_models():
    """Test with very simple models"""
    print("\n🔍 Testing Simple Models...")
    
    # Create test data
    input_ids, labels, vocab_size = create_simple_test_data()
    
    # Create models
    standard_model = SimpleStandardModel(vocab_size, d_model=64)
    improved_model = SimpleImprovedModel(vocab_size, d_model=64)
    geometric_loss = SimpleGeometricLoss(lambda_geo=0.1)
    
    print(f"\n📊 Model Parameters:")
    print(f"  Standard model: {sum(p.numel() for p in standard_model.parameters())} params")
    print(f"  Improved model: {sum(p.numel() for p in improved_model.parameters())} params")
    
    # Test without training first
    print(f"\n🧪 Testing Without Training:")
    
    standard_model.eval()
    improved_model.eval()
    
    with torch.no_grad():
        # Standard model
        standard_outputs = standard_model(input_ids)
        standard_loss = F.cross_entropy(standard_outputs, labels)
        standard_acc = (torch.argmax(standard_outputs, dim=1) == labels).float().mean()
        
        # Improved model
        improved_outputs = improved_model(input_ids)
        improved_loss = F.cross_entropy(improved_outputs, labels)
        improved_acc = (torch.argmax(improved_outputs, dim=1) == labels).float().mean()
        
        # Geometric loss
        embeddings = improved_model.get_embeddings(input_ids)
        geo_losses = geometric_loss(embeddings, improved_outputs, labels)
    
    print(f"  Standard: Loss={standard_loss.item():.4f}, Acc={standard_acc.item():.4f}")
    print(f"  Improved: Loss={improved_loss.item():.4f}, Acc={improved_acc.item():.4f}")
    print(f"  Geometric Loss: {geo_losses['geometric_loss'].item():.6f}")
    print(f"  Total Geometric: {geo_losses['total_geometric'].item():.6f}")
    
    # Now test with training
    print(f"\n🏋️ Testing With Training:")
    
    # Training setup
    optimizer_std = torch.optim.Adam(standard_model.parameters(), lr=0.001)
    optimizer_imp = torch.optim.Adam(improved_model.parameters(), lr=0.001)
    
    # Train both models
    for epoch in range(10):
        # Train standard model
        standard_model.train()
        optimizer_std.zero_grad()
        std_outputs = standard_model(input_ids)
        std_loss = F.cross_entropy(std_outputs, labels)
        std_loss.backward()
        optimizer_std.step()
        
        # Train improved model
        improved_model.train()
        optimizer_imp.zero_grad()
        imp_outputs = improved_model(input_ids)
        imp_embeddings = improved_model.get_embeddings(input_ids)
        imp_losses = geometric_loss(imp_embeddings, imp_outputs, labels)
        imp_losses['total_loss'].backward()
        optimizer_imp.step()
        
        if epoch % 2 == 0:
            print(f"  Epoch {epoch}: Std Loss={std_loss.item():.4f}, Imp Loss={imp_losses['total_loss'].item():.4f}")
    
    # Test after training
    print(f"\n📈 Testing After Training:")
    
    standard_model.eval()
    improved_model.eval()
    
    with torch.no_grad():
        # Standard model
        standard_outputs = standard_model(input_ids)
        standard_loss = F.cross_entropy(standard_outputs, labels)
        standard_acc = (torch.argmax(standard_outputs, dim=1) == labels).float().mean()
        
        # Improved model
        improved_outputs = improved_model(input_ids)
        improved_loss = F.cross_entropy(improved_outputs, labels)
        improved_acc = (torch.argmax(improved_outputs, dim=1) == labels).float().mean()
        
        # Geometric loss
        embeddings = improved_model.get_embeddings(input_ids)
        geo_losses = geometric_loss(embeddings, improved_outputs, labels)
    
    print(f"  Standard: Loss={standard_loss.item():.4f}, Acc={standard_acc.item():.4f}")
    print(f"  Improved: Loss={improved_loss.item():.4f}, Acc={improved_acc.item():.4f}")
    print(f"  Geometric Loss: {geo_losses['geometric_loss'].item():.6f}")
    print(f"  Total Geometric: {geo_losses['total_geometric'].item():.6f}")
    
    # Calculate improvement
    acc_improvement = (improved_acc.item() - standard_acc.item()) / standard_acc.item() * 100 if standard_acc.item() > 0 else 0
    print(f"  Accuracy Improvement: {acc_improvement:.1f}%")
    
    return {
        'standard_loss': standard_loss.item(),
        'standard_acc': standard_acc.item(),
        'improved_loss': improved_loss.item(),
        'improved_acc': improved_acc.item(),
        'acc_improvement': acc_improvement,
        'geometric_loss': geo_losses['geometric_loss'].item()
    }

def test_geometric_loss_components():
    """Test if geometric loss components are meaningful"""
    print("\n🔬 Testing Geometric Loss Components...")
    
    # Create test embeddings
    batch_size, seq_len, d_model = 4, 8, 32
    
    # Test 1: Random embeddings
    print("  Test 1: Random embeddings")
    random_embeddings = torch.randn(batch_size, seq_len, d_model)
    
    # Test 2: Structured embeddings (should have lower geometric loss)
    print("  Test 2: Structured embeddings")
    structured_embeddings = torch.zeros(batch_size, seq_len, d_model)
    for i in range(batch_size):
        for j in range(seq_len):
            structured_embeddings[i, j, :] = torch.tensor([i, j] + [0] * (d_model - 2))
    
    # Test 3: Identical embeddings (should have high geometric loss)
    print("  Test 3: Identical embeddings")
    identical_embeddings = torch.ones(batch_size, seq_len, d_model)
    
    geometric_loss = SimpleGeometricLoss(lambda_geo=0.1)
    
    for name, embeddings in [("Random", random_embeddings), 
                            ("Structured", structured_embeddings), 
                            ("Identical", identical_embeddings)]:
        losses = geometric_loss(embeddings)
        print(f"    {name}: Geometric Loss = {losses['geometric_loss'].item():.6f}")
    
    # Test if the loss is actually differentiable
    print("  Test 4: Gradient test")
    test_embeddings = torch.randn(2, 4, 16, requires_grad=True)
    losses = geometric_loss(test_embeddings)
    losses['total_geometric'].backward()
    print(f"    Gradient norm: {test_embeddings.grad.norm().item():.6f}")
    print(f"    ✅ Geometric loss is differentiable")

def test_known_good_configuration():
    """Test with a configuration that should definitely work"""
    print("\n✅ Testing Known Good Configuration...")
    
    # Use the ultra-minimal regularization that worked before
    from geometric_tools.immediate_improvements import GeometricRegularizationLoss
    
    input_ids, labels, vocab_size = create_simple_test_data()
    
    # Create a very simple model
    class MinimalModel(nn.Module):
        def __init__(self, vocab_size, d_model=32):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.classifier = nn.Linear(d_model, 2)
            
        def forward(self, input_ids):
            emb = self.embedding(input_ids)
            # Use mean pooling
            pooled = torch.mean(emb, dim=1)
            return self.classifier(pooled)
        
        def get_embeddings(self, input_ids):
            return self.embedding(input_ids)
    
    # Test with ultra-minimal regularization
    model = MinimalModel(vocab_size, d_model=32)
    geometric_loss = GeometricRegularizationLoss(
        lambda_strata=0.001, 
        lambda_curvature=0.001, 
        lambda_manifold=0.0005
    )
    
    print("  Testing ultra-minimal regularization...")
    
    # Test geometric loss
    embeddings = model.get_embeddings(input_ids)
    losses = geometric_loss(embeddings)
    
    print(f"    Strata Loss: {losses['strata_loss'].item():.6f}")
    print(f"    Curvature Loss: {losses['curvature_loss'].item():.6f}")
    print(f"    Manifold Loss: {losses['manifold_loss'].item():.6f}")
    print(f"    Total Geometric: {losses['total_geometric'].item():.6f}")
    
    # Test if it's actually different from zero
    if losses['total_geometric'].item() > 1e-8:
        print("    ✅ Geometric loss is non-zero")
    else:
        print("    ❌ Geometric loss is essentially zero - this might be the problem!")

def run_debug_experiment():
    """Run the debug experiment"""
    print("🐛 Starting Debug Experiment")
    print("=" * 50)
    print("Investigating why geometric regularization isn't working")
    print("=" * 50)
    
    # Test 1: Simple models
    print("\n1. Testing Simple Models...")
    simple_results = test_simple_models()
    
    # Test 2: Geometric loss components
    print("\n2. Testing Geometric Loss Components...")
    test_geometric_loss_components()
    
    # Test 3: Known good configuration
    print("\n3. Testing Known Good Configuration...")
    test_known_good_configuration()
    
    print("\n🔍 Debug Analysis:")
    print(f"Simple model improvement: {simple_results['acc_improvement']:.1f}%")
    
    if simple_results['acc_improvement'] > 0:
        print("✅ Simple models show improvement - the idea works!")
        print("❌ The problem is likely in the complex model implementations")
    else:
        print("❌ Even simple models don't show improvement")
        print("❌ The geometric regularization might be fundamentally flawed")
    
    print("\n✅ Debug experiment complete!")
    return simple_results

if __name__ == "__main__":
    run_debug_experiment()
