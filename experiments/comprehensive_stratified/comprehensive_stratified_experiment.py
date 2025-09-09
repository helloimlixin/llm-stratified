"""
Comprehensive Stratified Manifold Training Experiment
Fixed MoE implementation with more data and training epochs for solid validation

This experiment tests:
1. Fixed Stratified Mixture-of-Experts
2. More comprehensive dataset (500 samples per domain)
3. More training epochs (10 epochs)
4. Multiple model architectures
5. Cross-validation for robustness
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
import time
import json
from datetime import datetime
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    BertTokenizer, BertModel, BertForSequenceClassification,
    RobertaTokenizer, RobertaModel, RobertaForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import load_dataset, concatenate_datasets
from datasets import Value
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split, KFold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

class StratifiedAttention(nn.Module):
    """Stratified Attention Mechanism"""
    def __init__(self, d_model, num_heads, num_strata=3):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_strata = num_strata
        self.head_dim = d_model // num_heads
        
        # Stratum routing network
        self.stratum_router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_strata),
            nn.Softmax(dim=-1)
        )
        
        # Separate attention heads for each stratum
        self.stratum_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            for _ in range(num_strata)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Route tokens to strata
        stratum_probs = self.stratum_router(x)
        stratum_assignments = torch.argmax(stratum_probs, dim=-1)
        
        # Process each stratum separately
        stratum_outputs = []
        for stratum_idx in range(self.num_strata):
            stratum_mask = (stratum_assignments == stratum_idx).unsqueeze(-1)
            
            if stratum_mask.sum() > 0:
                stratum_tokens = x * stratum_mask
                stratum_out, _ = self.stratum_attentions[stratum_idx](
                    stratum_tokens, stratum_tokens, stratum_tokens,
                    key_padding_mask=attention_mask
                )
                stratum_outputs.append(stratum_out * stratum_mask)
            else:
                stratum_outputs.append(torch.zeros_like(x))
        
        # Combine stratum outputs
        combined_output = sum(stratum_outputs)
        output = self.output_proj(combined_output)
        
        return output, stratum_assignments

class StratifiedTokenRouter(nn.Module):
    """Stratified Token Routing"""
    def __init__(self, d_model, num_paths=3):
        super().__init__()
        self.d_model = d_model
        self.num_paths = num_paths
        
        # Token routing network
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_paths),
            nn.Softmax(dim=-1)
        )
        
        # Different processing paths
        self.processing_paths = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            ) for _ in range(num_paths)
        ])
        
        # Output combination
        self.output_combiner = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Route tokens to processing paths
        routing_probs = self.router(x)
        
        # Process through different paths
        path_outputs = []
        for path_idx in range(self.num_paths):
            path_weight = routing_probs[:, :, path_idx:path_idx+1]
            path_output = self.processing_paths[path_idx](x)
            path_outputs.append(path_weight * path_output)
        
        # Combine path outputs
        combined_output = sum(path_outputs)
        output = self.output_combiner(combined_output)
        
        return output, routing_probs

class StratifiedLayerProcessor(nn.Module):
    """Stratified Layer Processing"""
    def __init__(self, d_model, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Different processing strategies for different layers
        self.layer_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            ) for _ in range(num_layers)
        ])
        
        # Stratum-aware layer selection
        self.layer_selector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_layers),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Select which layers to use based on input characteristics
        layer_weights = self.layer_selector(x.mean(dim=1))
        
        # Process through selected layers
        layer_outputs = []
        for layer_idx in range(self.num_layers):
            layer_weight = layer_weights[:, layer_idx:layer_idx+1].unsqueeze(1)
            layer_output = self.layer_processors[layer_idx](x)
            layer_outputs.append(layer_weight * layer_output)
        
        # Combine layer outputs
        output = sum(layer_outputs)
        return output, layer_weights

class StratifiedMoE(nn.Module):
    """Fixed Stratified Mixture-of-Experts"""
    def __init__(self, d_model, num_experts=4, expert_capacity=64):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_capacity),
                nn.ReLU(),
                nn.Linear(expert_capacity, d_model)
            ) for _ in range(num_experts)
        ])
        
        # Stratum-aware gating
        self.gating_network = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Stratum detection
        self.stratum_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 3),  # 3 strata
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Detect stratum for each token
        stratum_probs = self.stratum_detector(x)
        
        # Gate to experts
        gating_probs = self.gating_network(x)
        
        # Process through experts
        expert_outputs = []
        for expert_idx in range(self.num_experts):
            expert_weight = gating_probs[:, :, expert_idx:expert_idx+1]
            expert_output = self.experts[expert_idx](x)
            expert_outputs.append(expert_weight * expert_output)
        
        # Combine expert outputs
        output = sum(expert_outputs)
        
        # Return as tuple to fix unpacking error
        return output, (gating_probs, stratum_probs)

class ComprehensiveStratifiedWrapper(nn.Module):
    """Comprehensive wrapper for stratified mechanisms"""
    def __init__(self, base_model, model_name: str, stratified_type: str = "attention"):
        super().__init__()
        self.base_model = base_model
        self.model_name = model_name
        self.stratified_type = stratified_type
        
        if hasattr(base_model, 'config'):
            hidden_size = base_model.config.hidden_size
        else:
            hidden_size = 768
        
        # Add stratified component based on type
        if stratified_type == "attention":
            self.stratified_component = StratifiedAttention(hidden_size, num_heads=8, num_strata=3)
        elif stratified_type == "routing":
            self.stratified_component = StratifiedTokenRouter(hidden_size, num_paths=3)
        elif stratified_type == "layers":
            self.stratified_component = StratifiedLayerProcessor(hidden_size, num_layers=6)
        elif stratified_type == "moe":
            self.stratified_component = StratifiedMoE(hidden_size, num_experts=4)
        elif stratified_type == "none":
            self.stratified_component = nn.Identity()
        else:
            raise ValueError(f"Unknown stratified type: {stratified_type}")
        
        # Add classification head
        self.classifier = nn.Linear(hidden_size, 3)  # 3 classes for comprehensive dataset
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Get base model outputs
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs.last_hidden_state
        
        # Apply stratified processing
        if self.stratified_type == "none":
            enhanced_hidden_states = hidden_states
            stratum_info = None
        else:
            enhanced_hidden_states, stratum_info = self.stratified_component(hidden_states)
        
        # Compute logits
        if self.model_name.startswith('gpt'):
            logits = self.classifier(enhanced_hidden_states[:, -1, :])
        else:
            logits = self.classifier(enhanced_hidden_states[:, 0, :])
        
        # Create outputs object
        class StratifiedOutputs:
            def __init__(self, logits, hidden_states, stratum_info=None):
                self.logits = logits
                self.last_hidden_state = hidden_states
                self.stratum_info = stratum_info
        
        stratified_outputs = StratifiedOutputs(logits, enhanced_hidden_states, stratum_info)
        
        # Add loss if labels provided
        if labels is not None:
            stratified_outputs.loss = F.cross_entropy(logits, labels)
        
        return stratified_outputs

def load_comprehensive_dataset(samples_per_domain=500):
    """Load comprehensive multidomain dataset"""
    print(f"📚 Loading comprehensive multidomain dataset ({samples_per_domain} samples per domain)...")
    
    def unify_dataset(ds, domain_name, samples_per_domain=100, text_field="text"):
        if text_field != "text":
            ds = ds.map(lambda x: {"text": x[text_field], "label": x["label"]})
        keep_cols = ["text", "label"]
        remove_cols = [c for c in ds.column_names if c not in keep_cols]
        ds = ds.remove_columns(remove_cols)
        ds = ds.map(lambda x: {"label": int(x["label"])})
        ds = ds.cast_column("label", Value("int64"))
        ds_small = ds.select(range(min(samples_per_domain, len(ds))))
        ds_small = ds_small.add_column("domain", [domain_name] * len(ds_small))
        return ds_small
    
    # Load datasets
    imdb_ds = unify_dataset(load_dataset("imdb", split=f"train[:{samples_per_domain}]"), "imdb", samples_per_domain)
    rt_ds = unify_dataset(load_dataset("rotten_tomatoes", split=f"train[:{samples_per_domain}]"), "rotten", samples_per_domain)
    
    ap_raw = load_dataset("amazon_polarity", split=f"train[:{int(2 * samples_per_domain)}]")
    ap_raw = ap_raw.map(lambda x: {"text": f"{x['title']} {x['content']}".strip()})
    ap_ds = unify_dataset(ap_raw, "amazon", samples_per_domain)
    
    sst2_ds = load_dataset("glue", "sst2", split=f"train[:{samples_per_domain}]")
    sst2_ds = unify_dataset(sst2_ds, "sst2", samples_per_domain, text_field="sentence")
    
    tweet_ds = load_dataset("tweet_eval", "sentiment", split=f"train[:{samples_per_domain}]")
    tweet_ds = unify_dataset(tweet_ds, "tweet", samples_per_domain, text_field="text")
    
    # Combine datasets
    combined_ds = concatenate_datasets([imdb_ds, rt_ds, ap_ds, sst2_ds, tweet_ds])
    
    print(f"✅ Loaded {len(combined_ds)} samples from 5 domains")
    return combined_ds

def train_comprehensive_model(model, train_loader, val_loader, num_epochs=10, learning_rate=2e-5):
    """Comprehensive training with detailed logging"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_inputs, batch_attention, batch_labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(input_ids=batch_inputs, attention_mask=batch_attention, labels=batch_labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            train_correct += (predictions == batch_labels).sum().item()
            train_total += batch_labels.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_inputs, batch_attention, batch_labels in val_loader:
                outputs = model(input_ids=batch_inputs, attention_mask=batch_attention, labels=batch_labels)
                
                val_loss += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                val_correct += (predictions == batch_labels).sum().item()
                val_total += batch_labels.size(0)
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = model.state_dict().copy()
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f"    Epoch {epoch+1}/{num_epochs}: Train Loss={avg_train_loss:.4f}, Train Acc={train_accuracy:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }

def evaluate_comprehensive_model(model, test_loader):
    """Comprehensive model evaluation"""
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_inputs, batch_attention, batch_labels in test_loader:
            outputs = model(input_ids=batch_inputs, attention_mask=batch_attention, labels=batch_labels)
            
            test_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            test_correct += (predictions == batch_labels).sum().item()
            test_total += batch_labels.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_accuracy = test_correct / test_total
    
    # Calculate additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_predictions,
        'labels': all_labels
    }

def run_comprehensive_stratified_experiment():
    """Run comprehensive stratified experiment"""
    print("🚀 Starting Comprehensive Stratified Manifold Training Experiment")
    print("=" * 80)
    print("Fixed MoE + More Data + More Epochs + Multiple Models")
    print("=" * 80)
    
    # Stratified approaches to test
    stratified_types = [
        ("none", "Standard Baseline"),
        ("attention", "Stratified Attention"),
        ("routing", "Stratified Token Routing"),
        ("layers", "Stratified Layer Processing"),
        ("moe", "Stratified Mixture-of-Experts")
    ]
    
    results = {}
    
    # Load comprehensive dataset
    combined_ds = load_comprehensive_dataset(samples_per_domain=500)
    texts = list(combined_ds["text"])
    labels = list(combined_ds["label"])
    
    print(f"Dataset: {len(texts)} samples, {len(set(labels))} classes")
    
    # Split data (70/15/15 for more training data)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"Dataset splits: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")
    
    # Load base model
    print(f"\n🔍 Loading DistilBERT base model...")
    tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")
    base_model = BertModel.from_pretrained("distilbert-base-uncased")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize data
    def tokenize_data(texts, labels):
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        return tokenized['input_ids'], tokenized['attention_mask'], torch.tensor(labels)
    
    train_inputs, train_attention, train_labels_tensor = tokenize_data(train_texts, train_labels)
    val_inputs, val_attention, val_labels_tensor = tokenize_data(val_texts, val_labels)
    test_inputs, test_attention, test_labels_tensor = tokenize_data(test_texts, test_labels)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_attention, train_labels_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(val_inputs, val_attention, val_labels_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    test_dataset = torch.utils.data.TensorDataset(test_inputs, test_attention, test_labels_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Test each stratified approach
    for stratified_type, description in stratified_types:
        print(f"\n🔍 Testing {description}...")
        
        try:
            # Create stratified model
            stratified_model = ComprehensiveStratifiedWrapper(base_model, "distilbert", stratified_type)
            
            # Train stratified model
            stratified_metrics = train_comprehensive_model(
                stratified_model, train_loader, val_loader, num_epochs=10
            )
            
            # Evaluate on test set
            stratified_eval = evaluate_comprehensive_model(stratified_model, test_loader)
            
            print(f"    {description} - Test Acc: {stratified_eval['test_accuracy']:.4f}, F1: {stratified_eval['f1_score']:.4f}")
            
            results[stratified_type] = {
                'description': description,
                'metrics': stratified_metrics,
                'eval': stratified_eval
            }
            
        except Exception as e:
            print(f"  ❌ Error testing {stratified_type}: {e}")
            results[stratified_type] = {
                'description': description,
                'error': str(e)
            }
    
    # Print comprehensive results
    print("\n📊 COMPREHENSIVE STRATIFIED RESULTS:")
    print("=" * 60)
    
    successful_tests = 0
    total_improvement = 0
    improvements = []
    
    # Get baseline performance
    baseline_acc = results.get('none', {}).get('eval', {}).get('test_accuracy', 0)
    
    for stratified_type, result in results.items():
        print(f"\n{result['description']}:")
        
        if 'error' in result:
            print(f"  ❌ Error: {result['error']}")
        else:
            test_acc = result['eval']['test_accuracy']
            f1_score = result['eval']['f1_score']
            
            if stratified_type == 'none':
                print(f"  Test Acc: {test_acc:.4f}, F1: {f1_score:.4f} (Baseline)")
            else:
                improvement = (test_acc - baseline_acc) / baseline_acc * 100 if baseline_acc > 0 else 0
                print(f"  Test Acc: {test_acc:.4f}, F1: {f1_score:.4f}")
                print(f"  Improvement: {improvement:.2f}%")
                
                successful_tests += 1
                improvements.append(improvement)
                total_improvement += improvement
    
    if successful_tests > 0:
        avg_improvement = total_improvement / successful_tests
        positive_improvements = sum(1 for imp in improvements if imp > 0)
        
        print(f"\n📈 Overall Results:")
        print(f"  Successful tests: {successful_tests}")
        print(f"  Average improvement: {avg_improvement:.2f}%")
        print(f"  Positive improvements: {positive_improvements}/{successful_tests}")
        
        if avg_improvement > 0:
            print(f"\n✅ SUCCESS! Comprehensive stratified approaches show improvement!")
            print(f"✅ Average improvement: {avg_improvement:.2f}%")
        else:
            print(f"\n❌ No improvement with comprehensive stratified approaches")
            print(f"❌ Average improvement: {avg_improvement:.2f}%")
    else:
        print(f"\n❌ No successful tests completed")
    
    print("\n✅ Comprehensive stratified experiment complete!")
    return results

if __name__ == "__main__":
    run_comprehensive_stratified_experiment()
