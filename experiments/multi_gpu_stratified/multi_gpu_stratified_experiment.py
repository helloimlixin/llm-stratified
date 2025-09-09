"""
Multi-GPU Stratified Manifold Training with Hugging Face Framework
High-performance training with 5000 samples per domain and 100 epochs

This experiment uses:
1. Hugging Face Trainer framework
2. Multi-GPU DataParallel/DistributedDataParallel
3. Large batch sizes for GPU utilization
4. 5000 samples per domain (25,000 total)
5. 100 training epochs
6. Advanced stratified mechanisms
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
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import load_dataset, concatenate_datasets, Dataset
from datasets import Value
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
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
    """Stratified Mixture-of-Experts"""
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
        
        return output, (gating_probs, stratum_probs)

class MultiGPUStratifiedModel(nn.Module):
    """Multi-GPU compatible stratified model"""
    def __init__(self, model_name: str, stratified_type: str = "attention"):
        super().__init__()
        self.model_name = model_name
        self.stratified_type = stratified_type
        
        # Load base model
        if model_name == "distilbert":
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.base_model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", 
                num_labels=3,  # 3 classes for comprehensive dataset
                ignore_mismatched_sizes=True
            )
        elif model_name == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.base_model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=3,
                ignore_mismatched_sizes=True
            )
        elif model_name == "roberta":
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            self.base_model = AutoModelForSequenceClassification.from_pretrained(
                "roberta-base",
                num_labels=3,
                ignore_mismatched_sizes=True
            )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get hidden size
        hidden_size = self.base_model.config.hidden_size
        
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
        
        # Replace classification head with stratified processing
        if stratified_type != "none":
            # Store original classifier
            self.original_classifier = self.base_model.classifier
            # Create new classifier that includes stratified processing
            self.base_model.classifier = nn.Identity()  # Remove original classifier
            self.final_classifier = nn.Linear(hidden_size, 3)
        else:
            self.final_classifier = None
        
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
        if self.final_classifier is not None:
            if self.model_name.startswith('gpt'):
                logits = self.final_classifier(enhanced_hidden_states[:, -1, :])
            else:
                logits = self.final_classifier(enhanced_hidden_states[:, 0, :])
        else:
            # Use original classifier
            if self.model_name.startswith('gpt'):
                logits = self.original_classifier(enhanced_hidden_states[:, -1, :])
            else:
                logits = self.original_classifier(enhanced_hidden_states[:, 0, :])
        
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

def load_large_multidomain_dataset(samples_per_domain=5000):
    """Load large multidomain dataset for multi-GPU training"""
    print(f"📚 Loading large multidomain dataset ({samples_per_domain} samples per domain)...")
    
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

def compute_metrics(eval_pred):
    """Compute metrics for Hugging Face Trainer"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def run_multi_gpu_stratified_experiment():
    """Run multi-GPU stratified experiment"""
    print("🚀 Starting Multi-GPU Stratified Manifold Training")
    print("=" * 80)
    print("Hugging Face Trainer + Multi-GPU + Large Dataset + 100 Epochs")
    print("=" * 80)
    
    # Check GPU availability
    device_count = torch.cuda.device_count()
    print(f"🔍 Available GPUs: {device_count}")
    
    if device_count == 0:
        print("❌ No GPUs available, falling back to CPU")
        device = "cpu"
    else:
        device = "cuda"
        print(f"✅ Using {device_count} GPUs")
    
    # Stratified approaches to test
    stratified_types = [
        ("none", "Standard Baseline"),
        ("attention", "Stratified Attention"),
        ("routing", "Stratified Token Routing"),
        ("layers", "Stratified Layer Processing"),
        ("moe", "Stratified Mixture-of-Experts")
    ]
    
    results = {}
    
    # Load large dataset
    combined_ds = load_large_multidomain_dataset(samples_per_domain=5000)
    
    # Convert to Hugging Face Dataset format
    texts = list(combined_ds["text"])
    labels = list(combined_ds["label"])
    
    # Create Hugging Face dataset
    dataset = Dataset.from_dict({
        "text": texts,
        "label": labels
    })
    
    print(f"Dataset: {len(dataset)} samples, {len(set(labels))} classes")
    
    # Split dataset (80/10/10 for large dataset)
    train_test_split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split_dataset["train"]
    test_dataset = train_test_split_dataset["test"]
    
    val_test_split_dataset = test_dataset.train_test_split(test_size=0.5, seed=42)
    val_dataset = val_test_split_dataset["train"]
    test_dataset = val_test_split_dataset["test"]
    
    print(f"Dataset splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Test each stratified approach
    for stratified_type, description in stratified_types:
        print(f"\n🔍 Testing {description}...")
        
        try:
            # Create stratified model
            model = MultiGPUStratifiedModel("distilbert", stratified_type)
            
            # Move to GPU if available
            if device == "cuda":
                model = model.cuda()
                if device_count > 1:
                    model = nn.DataParallel(model)
                    print(f"    Using DataParallel with {device_count} GPUs")
            
            # Get tokenizer from model (handle DataParallel wrapper)
            tokenizer = model.tokenizer if not isinstance(model, nn.DataParallel) else model.module.tokenizer
            
            # Tokenize datasets
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=128
                )
            
            train_tokenized = train_dataset.map(tokenize_function, batched=True)
            val_tokenized = val_dataset.map(tokenize_function, batched=True)
            test_tokenized = test_dataset.map(tokenize_function, batched=True)
            
            # Set format for PyTorch
            train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
            val_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
            test_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"./results/multi_gpu_{stratified_type}",
                num_train_epochs=100,
                per_device_train_batch_size=32,  # Large batch size
                per_device_eval_batch_size=64,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=f"./logs/multi_gpu_{stratified_type}",
                logging_steps=100,
                eval_strategy="steps",
                eval_steps=500,
                save_strategy="steps",
                save_steps=1000,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                greater_is_better=True,
                dataloader_num_workers=4,
                fp16=True,  # Mixed precision for speed
                gradient_accumulation_steps=2,  # Effective batch size = 32 * 2 * num_gpus
                learning_rate=2e-5,
                lr_scheduler_type="cosine",
                report_to=None,  # Disable wandb/tensorboard
            )
            
            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_tokenized,
                eval_dataset=val_tokenized,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
            )
            
            # Train model
            print(f"    Starting training for {description}...")
            start_time = time.time()
            
            trainer.train()
            
            training_time = time.time() - start_time
            print(f"    Training completed in {training_time:.2f} seconds")
            
            # Evaluate on test set
            test_results = trainer.evaluate(test_tokenized)
            
            print(f"    {description} - Test Acc: {test_results['eval_accuracy']:.4f}, F1: {test_results['eval_f1']:.4f}")
            
            results[stratified_type] = {
                'description': description,
                'test_results': test_results,
                'training_time': training_time
            }
            
        except Exception as e:
            print(f"  ❌ Error testing {stratified_type}: {e}")
            results[stratified_type] = {
                'description': description,
                'error': str(e)
            }
    
    # Print comprehensive results
    print("\n📊 MULTI-GPU STRATIFIED RESULTS:")
    print("=" * 60)
    
    successful_tests = 0
    total_improvement = 0
    improvements = []
    
    # Get baseline performance
    baseline_acc = results.get('none', {}).get('test_results', {}).get('eval_accuracy', 0)
    
    for stratified_type, result in results.items():
        print(f"\n{result['description']}:")
        
        if 'error' in result:
            print(f"  ❌ Error: {result['error']}")
        else:
            test_acc = result['test_results']['eval_accuracy']
            f1_score = result['test_results']['eval_f1']
            training_time = result['training_time']
            
            if stratified_type == 'none':
                print(f"  Test Acc: {test_acc:.4f}, F1: {f1_score:.4f}, Time: {training_time:.2f}s (Baseline)")
            else:
                improvement = (test_acc - baseline_acc) / baseline_acc * 100 if baseline_acc > 0 else 0
                print(f"  Test Acc: {test_acc:.4f}, F1: {f1_score:.4f}, Time: {training_time:.2f}s")
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
            print(f"\n✅ SUCCESS! Multi-GPU stratified approaches show improvement!")
            print(f"✅ Average improvement: {avg_improvement:.2f}%")
        else:
            print(f"\n❌ No improvement with multi-GPU stratified approaches")
            print(f"❌ Average improvement: {avg_improvement:.2f}%")
    else:
        print(f"\n❌ No successful tests completed")
    
    print("\n✅ Multi-GPU stratified experiment complete!")
    return results

if __name__ == "__main__":
    run_multi_gpu_stratified_experiment()
