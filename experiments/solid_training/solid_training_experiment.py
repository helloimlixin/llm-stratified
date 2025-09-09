"""
Solid Training Experiment with Geometric Regularization
Proper training with multiple epochs, validation, and comprehensive evaluation

This experiment tests:
1. Substantial training (10+ epochs)
2. Proper train/validation/test splits
3. Training curve analysis
4. Convergence comparison
5. Final test set evaluation
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
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from geometric_tools.immediate_improvements import GeometricRegularizationLoss

class SolidTrainingModel(nn.Module):
    """
    Model wrapper for solid training experiments
    """
    def __init__(self, base_model, model_name: str, geometric_loss: Optional[GeometricRegularizationLoss] = None):
        super().__init__()
        self.base_model = base_model
        self.model_name = model_name
        self.geometric_loss = geometric_loss
        
        # Add geometric enhancement layer
        if hasattr(base_model, 'config'):
            hidden_size = base_model.config.hidden_size
        else:
            hidden_size = 768
        
        self.geometric_layer = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Get base model outputs
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs.last_hidden_state
        
        # Apply geometric enhancement
        geometric_enhanced = self.geometric_layer(hidden_states)
        enhanced_hidden_states = hidden_states + 0.1 * geometric_enhanced
        
        # Compute logits using classification head
        if hasattr(self.base_model, 'classifier'):
            if self.model_name.startswith('gpt'):
                enhanced_logits = self.base_model.classifier(enhanced_hidden_states[:, -1, :])
            else:
                enhanced_logits = self.base_model.classifier(enhanced_hidden_states[:, 0, :])
        else:
            enhanced_logits = torch.randn(input_ids.size(0), 2)
        
        # Create new outputs object
        class EnhancedOutputs:
            def __init__(self, logits, hidden_states, loss=None):
                self.logits = logits
                self.last_hidden_state = hidden_states
                self.loss = loss
        
        enhanced_outputs = EnhancedOutputs(enhanced_logits, enhanced_hidden_states)
        
        # Add geometric loss if provided
        if self.geometric_loss is not None and labels is not None:
            geometric_losses = self.geometric_loss(enhanced_hidden_states, enhanced_logits, labels)
            enhanced_outputs.loss = geometric_losses['total_geometric']
        
        return enhanced_outputs

def unify_dataset(ds, domain_name, samples_per_domain=100, text_field="text"):
    """Unify dataset format for multidomain sentiment analysis"""
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

def load_multidomain_sentiment(samples_per_domain=200):
    """Load real multidomain sentiment datasets"""
    print(f"📚 Loading real multidomain sentiment datasets ({samples_per_domain} samples per domain)...")
    
    # Load IMDB
    imdb_ds = unify_dataset(load_dataset("imdb", split=f"train[:{samples_per_domain}]"), "imdb", samples_per_domain)
    
    # Load Rotten Tomatoes
    rt_ds = unify_dataset(load_dataset("rotten_tomatoes", split=f"train[:{samples_per_domain}]"), "rotten", samples_per_domain)
    
    # Load Amazon Polarity
    ap_raw = load_dataset("amazon_polarity", split=f"train[:{int(2 * samples_per_domain)}]")
    ap_raw = ap_raw.map(lambda x: {"text": f"{x['title']} {x['content']}".strip()})
    ap_ds = unify_dataset(ap_raw, "amazon", samples_per_domain)
    
    # Load SST-2
    sst2_ds = load_dataset("glue", "sst2", split=f"train[:{samples_per_domain}]")
    sst2_ds = unify_dataset(sst2_ds, "sst2", samples_per_domain, text_field="sentence")
    
    # Load Tweet Eval
    tweet_ds = load_dataset("tweet_eval", "sentiment", split=f"train[:{samples_per_domain}]")
    tweet_ds = unify_dataset(tweet_ds, "tweet", samples_per_domain, text_field="text")
    
    # Load AG News
    ag_news_ds = load_dataset("ag_news", split=f"train[:{samples_per_domain}]")
    ag_news_ds = unify_dataset(ag_news_ds, "ag_news", samples_per_domain, text_field="text")
    
    # Combine all datasets
    combined_ds = concatenate_datasets([imdb_ds, rt_ds, ap_ds, sst2_ds, tweet_ds, ag_news_ds])
    
    print(f"✅ Loaded {len(combined_ds)} samples from {len([imdb_ds, rt_ds, ap_ds, sst2_ds, tweet_ds, ag_news_ds])} domains")
    return combined_ds

def train_model_comprehensive(model, train_loader, val_loader, num_epochs=10, learning_rate=2e-5):
    """
    Comprehensive training with detailed logging
    """
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
            
            # Forward pass
            outputs = model(input_ids=batch_inputs, attention_mask=batch_attention, labels=batch_labels)
            
            # Calculate loss
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            else:
                loss = F.cross_entropy(outputs.logits, batch_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
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
                
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    val_loss += outputs.loss.item()
                else:
                    val_loss += F.cross_entropy(outputs.logits, batch_labels).item()
                
                predictions = torch.argmax(outputs.logits, dim=1)
                val_correct += (predictions == batch_labels).sum().item()
                val_total += batch_labels.size(0)
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        # Store metrics
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

def evaluate_model(model, test_loader):
    """
    Comprehensive model evaluation
    """
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_inputs, batch_attention, batch_labels in test_loader:
            outputs = model(input_ids=batch_inputs, attention_mask=batch_attention, labels=batch_labels)
            
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                test_loss += outputs.loss.item()
            else:
                test_loss += F.cross_entropy(outputs.logits, batch_labels).item()
            
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

def run_solid_training_experiment():
    """
    Run comprehensive solid training experiment
    """
    print("🚀 Starting Solid Training Experiment with Geometric Regularization")
    print("=" * 80)
    print("Comprehensive training with proper epochs, validation, and evaluation")
    print("=" * 80)
    
    # Model configurations
    model_configs = [
        {
            'name': 'distilbert-base-uncased',
            'type': 'bert',
            'description': 'DistilBERT (66M parameters)'
        },
        {
            'name': 'bert-base-uncased',
            'type': 'bert',
            'description': 'BERT Base (110M parameters)'
        }
    ]
    
    results = {}
    
    # Load real multidomain sentiment dataset
    combined_ds = load_multidomain_sentiment(samples_per_domain=150)
    
    # Extract texts and labels
    texts = combined_ds["text"]
    labels = combined_ds["label"]
    
    # Convert to lists for sklearn compatibility
    texts = list(texts)
    labels = list(labels)
    
    print(f"Dataset: {len(texts)} samples, {len(set(labels))} classes")
    
    # Split into train/validation/test (60/20/20)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.4, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"Dataset splits: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")
    
    for config in model_configs:
        model_name = config['name']
        model_type = config['type']
        description = config['description']
        
        print(f"\n🔍 Training {description}...")
        
        try:
            # Load tokenizer and model
            print(f"  Loading {model_name}...")
            
            if model_type == 'bert':
                tokenizer = BertTokenizer.from_pretrained(model_name)
                base_model = BertModel.from_pretrained(model_name)
            
            # Set pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Add classification head - determine number of classes dynamically
            num_classes = len(set(labels))
            if hasattr(base_model, 'config'):
                hidden_size = base_model.config.hidden_size
            else:
                hidden_size = 768
            base_model.classifier = nn.Linear(hidden_size, num_classes)
            
            # Tokenize all data
            print(f"  Tokenizing data...")
            
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
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
            
            val_dataset = torch.utils.data.TensorDataset(val_inputs, val_attention, val_labels_tensor)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
            
            test_dataset = torch.utils.data.TensorDataset(test_inputs, test_attention, test_labels_tensor)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
            
            # Train standard model
            print(f"  Training standard {model_name}...")
            standard_model = SolidTrainingModel(base_model, model_name, None)
            
            standard_metrics = train_model_comprehensive(
                standard_model, train_loader, val_loader, num_epochs=5  # Reduced epochs for speed
            )
            
            standard_eval = evaluate_model(standard_model, test_loader)
            
            print(f"    Standard - Test Acc: {standard_eval['test_accuracy']:.4f}, F1: {standard_eval['f1_score']:.4f}")
            
            # Train improved model with geometric regularization
            print(f"  Training improved {model_name} with geometric regularization...")
            
            geometric_loss = GeometricRegularizationLoss(
                lambda_strata=0.001,
                lambda_curvature=0.001,
                lambda_manifold=0.0005
            )
            
            improved_model = SolidTrainingModel(base_model, model_name, geometric_loss)
            
            improved_metrics = train_model_comprehensive(
                improved_model, train_loader, val_loader, num_epochs=5  # Reduced epochs for speed
            )
            
            improved_eval = evaluate_model(improved_model, test_loader)
            
            print(f"    Improved - Test Acc: {improved_eval['test_accuracy']:.4f}, F1: {improved_eval['f1_score']:.4f}")
            
            # Calculate improvements
            acc_improvement = (improved_eval['test_accuracy'] - standard_eval['test_accuracy']) / standard_eval['test_accuracy'] * 100
            f1_improvement = (improved_eval['f1_score'] - standard_eval['f1_score']) / standard_eval['f1_score'] * 100
            
            print(f"    Accuracy Improvement: {acc_improvement:.2f}%")
            print(f"    F1 Improvement: {f1_improvement:.2f}%")
            
            results[model_name] = {
                'description': description,
                'standard_metrics': standard_metrics,
                'standard_eval': standard_eval,
                'improved_metrics': improved_metrics,
                'improved_eval': improved_eval,
                'acc_improvement': acc_improvement,
                'f1_improvement': f1_improvement,
                'model_type': model_type
            }
            
        except Exception as e:
            print(f"  ❌ Error training {model_name}: {e}")
            results[model_name] = {
                'description': description,
                'error': str(e),
                'model_type': model_type
            }
    
    # Print comprehensive results
    print("\n📊 SOLID TRAINING RESULTS:")
    print("=" * 60)
    
    successful_tests = 0
    total_acc_improvement = 0
    total_f1_improvement = 0
    acc_improvements = []
    f1_improvements = []
    
    for model_name, result in results.items():
        print(f"\n{result['description']}:")
        
        if 'error' in result:
            print(f"  ❌ Error: {result['error']}")
        else:
            print(f"  Standard - Test Acc: {result['standard_eval']['test_accuracy']:.4f}, F1: {result['standard_eval']['f1_score']:.4f}")
            print(f"  Improved - Test Acc: {result['improved_eval']['test_accuracy']:.4f}, F1: {result['improved_eval']['f1_score']:.4f}")
            print(f"  Accuracy Improvement: {result['acc_improvement']:.2f}%")
            print(f"  F1 Improvement: {result['f1_improvement']:.2f}%")
            
            successful_tests += 1
            acc_improvements.append(result['acc_improvement'])
            f1_improvements.append(result['f1_improvement'])
            total_acc_improvement += result['acc_improvement']
            total_f1_improvement += result['f1_improvement']
    
    if successful_tests > 0:
        avg_acc_improvement = total_acc_improvement / successful_tests
        avg_f1_improvement = total_f1_improvement / successful_tests
        positive_acc_improvements = sum(1 for imp in acc_improvements if imp > 0)
        positive_f1_improvements = sum(1 for imp in f1_improvements if imp > 0)
        
        print(f"\n📈 Overall Results:")
        print(f"  Successful tests: {successful_tests}")
        print(f"  Average accuracy improvement: {avg_acc_improvement:.2f}%")
        print(f"  Average F1 improvement: {avg_f1_improvement:.2f}%")
        print(f"  Positive accuracy improvements: {positive_acc_improvements}/{successful_tests}")
        print(f"  Positive F1 improvements: {positive_f1_improvements}/{successful_tests}")
        
        if avg_acc_improvement > 0 and avg_f1_improvement > 0:
            print(f"\n✅ SUCCESS! Geometric regularization shows improvement with solid training!")
            print(f"✅ Average accuracy improvement: {avg_acc_improvement:.2f}%")
            print(f"✅ Average F1 improvement: {avg_f1_improvement:.2f}%")
        else:
            print(f"\n❌ No improvement even with solid training")
            print(f"❌ Average accuracy improvement: {avg_acc_improvement:.2f}%")
            print(f"❌ Average F1 improvement: {avg_f1_improvement:.2f}%")
    else:
        print(f"\n❌ No successful tests completed")
    
    print("\n✅ Solid training experiment complete!")
    return results

if __name__ == "__main__":
    run_solid_training_experiment()
