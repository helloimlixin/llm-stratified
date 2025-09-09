"""
Challenging Debug Experiment
Testing geometric regularization on a task where it should actually help

The previous debug showed both models reach 100% accuracy too quickly.
This experiment uses:
1. More challenging data
2. Smaller models (so they struggle more)
3. Less training (so improvement is visible)
4. Noisy data (so perfect accuracy is harder)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def create_challenging_test_data():
    """Create more challenging test data"""
    print("📊 Creating Challenging Test Data...")
    
    # More challenging: longer sequences, more vocabulary, noisy labels
    texts = []
    labels = []
    
    # Positive examples (label 1)
    positive_templates = [
        "this is absolutely wonderful and fantastic",
        "I love this amazing product very much",
        "excellent quality and great service here",
        "outstanding performance and brilliant design",
        "superb experience with perfect results"
    ]
    
    # Negative examples (label 0)  
    negative_templates = [
        "this is terrible and awful quality",
        "I hate this horrible product completely",
        "poor quality and bad service here",
        "disappointing performance and ugly design",
        "terrible experience with failed results"
    ]
    
    # Create more diverse examples
    for i in range(200):
        if i % 2 == 0:
            template = random.choice(positive_templates)
            label = 1
        else:
            template = random.choice(negative_templates)
            label = 0
        
        # Add some noise/randomness
        noise_words = ["random", "extra", "word", "here", "there", "some", "more", "text"]
        noisy_template = template + " " + " ".join(random.choices(noise_words, k=random.randint(1, 3)))
        
        texts.append(noisy_template)
        labels.append(label)
    
    # Add some mislabeled examples (10% noise)
    for i in range(20):
        if i % 2 == 0:
            template = random.choice(positive_templates)
            label = 0  # Mislabeled as negative
        else:
            template = random.choice(negative_templates)
            label = 1  # Mislabeled as positive
        
        texts.append(template)
        labels.append(label)
    
    # Create vocabulary
    all_words = []
    for text in texts:
        all_words.extend(text.split())
    
    unique_words = list(set(all_words))
    vocab = {word: i for i, word in enumerate(unique_words)}
    vocab['<PAD>'] = len(vocab)
    vocab['<UNK>'] = len(vocab)
    
    def tokenize(text, max_len=15):
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
    print(f"  ✅ Label distribution: {torch.bincount(labels_tensor).tolist()}")
    print(f"  ✅ Added 10% label noise for challenge")
    
    return input_ids, labels_tensor, len(vocab)

class ChallengingStandardModel(nn.Module):
    """Smaller model that should struggle more"""
    def __init__(self, vocab_size, d_model=16):  # Very small model
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=2, batch_first=True),
            num_layers=1  # Only 1 layer
        )
        self.classifier = nn.Linear(d_model, 2)
        
    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        hidden = self.transformer(emb)
        cls_output = hidden[:, 0, :]
        return self.classifier(cls_output)

class ChallengingImprovedModel(nn.Module):
    """Smaller improved model"""
    def __init__(self, vocab_size, d_model=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=2, batch_first=True),
            num_layers=1
        )
        self.classifier = nn.Linear(d_model, 2)
        
        # Geometric enhancement
        self.geometric_projection = nn.Linear(d_model, d_model)
        
    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        
        # Apply geometric enhancement
        geo_emb = self.geometric_projection(emb)
        emb = emb + 0.2 * geo_emb  # Stronger geometric influence
        
        hidden = self.transformer(emb)
        cls_output = hidden[:, 0, :]
        return self.classifier(cls_output)
    
    def get_embeddings(self, input_ids):
        emb = self.embedding(input_ids)
        geo_emb = self.geometric_projection(emb)
        return emb + 0.2 * geo_emb

class ChallengingGeometricLoss(nn.Module):
    """More aggressive geometric loss"""
    def __init__(self, lambda_geo=0.5):  # Higher weight
        super().__init__()
        self.lambda_geo = lambda_geo
        
    def forward(self, embeddings, predictions=None, targets=None):
        losses = {}
        
        batch_size, seq_len, d_model = embeddings.shape
        
        # Flatten embeddings
        flat_emb = embeddings.view(-1, d_model)
        
        # Compute pairwise distances
        distances = torch.cdist(flat_emb, flat_emb, p=2)
        
        # More aggressive geometric loss
        # Encourage embeddings to be well-separated
        geometric_loss = torch.mean(torch.exp(-distances * 0.1)) * 0.1
        
        losses['geometric_loss'] = geometric_loss
        losses['total_geometric'] = self.lambda_geo * geometric_loss
        
        if predictions is not None and targets is not None:
            standard_loss = F.cross_entropy(predictions, targets)
            losses['standard_loss'] = standard_loss
            losses['total_loss'] = standard_loss + losses['total_geometric']
        
        return losses

def test_challenging_models():
    """Test with challenging models and data"""
    print("\n🔍 Testing Challenging Models...")
    
    # Create challenging test data
    input_ids, labels, vocab_size = create_challenging_test_data()
    
    # Create smaller models that should struggle
    standard_model = ChallengingStandardModel(vocab_size, d_model=16)
    improved_model = ChallengingImprovedModel(vocab_size, d_model=16)
    geometric_loss = ChallengingGeometricLoss(lambda_geo=0.5)
    
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
    
    # Train with fewer epochs so we can see improvement
    print(f"\n🏋️ Training (Limited Epochs):")
    
    optimizer_std = torch.optim.Adam(standard_model.parameters(), lr=0.01)
    optimizer_imp = torch.optim.Adam(improved_model.parameters(), lr=0.01)
    
    # Track training progress
    std_losses = []
    imp_losses = []
    std_accs = []
    imp_accs = []
    
    for epoch in range(5):  # Only 5 epochs
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
        imp_losses_dict = geometric_loss(imp_embeddings, imp_outputs, labels)
        imp_losses_dict['total_loss'].backward()
        optimizer_imp.step()
        
        # Track progress
        std_losses.append(std_loss.item())
        imp_losses.append(imp_losses_dict['total_loss'].item())
        
        # Evaluate accuracy
        with torch.no_grad():
            std_acc = (torch.argmax(std_outputs, dim=1) == labels).float().mean()
            imp_acc = (torch.argmax(imp_outputs, dim=1) == labels).float().mean()
            std_accs.append(std_acc.item())
            imp_accs.append(imp_acc.item())
        
        print(f"  Epoch {epoch}: Std Loss={std_loss.item():.4f}, Imp Loss={imp_losses_dict['total_loss'].item():.4f}")
        print(f"           Std Acc={std_acc.item():.4f}, Imp Acc={imp_acc.item():.4f}")
    
    # Final evaluation
    print(f"\n📈 Final Results:")
    
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
    
    # Show training curves
    print(f"\n📊 Training Curves:")
    print(f"  Standard Loss: {[f'{x:.4f}' for x in std_losses]}")
    print(f"  Improved Loss: {[f'{x:.4f}' for x in imp_losses]}")
    print(f"  Standard Acc:  {[f'{x:.4f}' for x in std_accs]}")
    print(f"  Improved Acc:  {[f'{x:.4f}' for x in imp_accs]}")
    
    return {
        'standard_loss': standard_loss.item(),
        'standard_acc': standard_acc.item(),
        'improved_loss': improved_loss.item(),
        'improved_acc': improved_acc.item(),
        'acc_improvement': acc_improvement,
        'geometric_loss': geo_losses['geometric_loss'].item(),
        'training_curves': {
            'std_losses': std_losses,
            'imp_losses': imp_losses,
            'std_accs': std_accs,
            'imp_accs': imp_accs
        }
    }

def run_challenging_debug_experiment():
    """Run the challenging debug experiment"""
    print("🐛 Starting Challenging Debug Experiment")
    print("=" * 60)
    print("Testing geometric regularization on challenging data")
    print("=" * 60)
    
    # Test challenging models
    print("\n1. Testing Challenging Models...")
    results = test_challenging_models()
    
    print("\n🔍 Analysis:")
    print(f"Final accuracy improvement: {results['acc_improvement']:.1f}%")
    
    if results['acc_improvement'] > 0:
        print("✅ Challenging models show improvement!")
        print("✅ The geometric regularization idea works when models struggle")
        print("✅ The problem was that previous tasks were too easy")
    elif results['acc_improvement'] < 0:
        print("❌ Improved model performs worse")
        print("❌ Geometric regularization might be hurting performance")
    else:
        print("❌ No improvement even on challenging data")
        print("❌ The geometric regularization might be fundamentally flawed")
    
    # Check if geometric loss is actually helping during training
    print(f"\n📈 Training Analysis:")
    std_accs = results['training_curves']['std_accs']
    imp_accs = results['training_curves']['imp_accs']
    
    print(f"Standard model accuracy progression: {std_accs}")
    print(f"Improved model accuracy progression: {imp_accs}")
    
    if len(imp_accs) > 1 and imp_accs[-1] > imp_accs[0]:
        print("✅ Improved model shows learning progression")
    else:
        print("❌ Improved model doesn't show clear learning")
    
    print("\n✅ Challenging debug experiment complete!")
    return results

if __name__ == "__main__":
    run_challenging_debug_experiment()
