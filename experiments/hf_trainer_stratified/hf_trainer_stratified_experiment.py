"""
Hugging Face Trainer + Accelerate Multi-GPU Stratified Training Experiment
================================================================================
Using Hugging Face Trainer with Accelerate for proper multi-GPU training
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, BertTokenizer, BertModel,
    RobertaTokenizer, RobertaModel, GPT2Tokenizer, GPT2Model,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from accelerate import Accelerator
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class StratifiedAttention(nn.Module):
    """Stratified attention mechanism"""
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Stratum-specific projections
        self.stratum_q = nn.Linear(d_model, d_model)
        self.stratum_k = nn.Linear(d_model, d_model)
        self.stratum_v = nn.Linear(d_model, d_model)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Standard attention
        Q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Stratum-specific attention
        Q_s = self.stratum_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K_s = self.stratum_k(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V_s = self.stratum_v(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Combine standard and stratum attention
        Q_combined = Q + 0.1 * Q_s
        K_combined = K + 0.1 * K_s
        V_combined = V + 0.1 * V_s
        
        # Scaled dot-product attention
        scores = torch.matmul(Q_combined, K_combined.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V_combined)
        
        # Reshape and project
        attn_output = attn_output.view(batch_size, seq_len, d_model)
        output = self.out_proj(attn_output)
        
        stratum_info = {
            'attention_weights': attn_weights.mean(dim=1),  # Average over heads
            'stratum_contribution': 0.1
        }
        
        return output, stratum_info

class StratifiedTokenRouter(nn.Module):
    """Stratified token routing mechanism"""
    def __init__(self, d_model: int, num_strata: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_strata = num_strata
        
        self.router = nn.Linear(d_model, num_strata)
        self.stratum_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_strata)
        ])
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Route tokens to strata
        routing_logits = self.router(hidden_states)
        routing_probs = F.softmax(routing_logits, dim=-1)
        
        # Apply stratum-specific projections
        stratum_outputs = []
        for i, proj in enumerate(self.stratum_projections):
            stratum_output = proj(hidden_states) * routing_probs[:, :, i:i+1]
            stratum_outputs.append(stratum_output)
        
        # Combine stratum outputs
        output = sum(stratum_outputs)
        
        stratum_info = {
            'routing_probs': routing_probs,
            'dominant_strata': torch.argmax(routing_probs, dim=-1)
        }
        
        return output, stratum_info

class StratifiedLayerProcessor(nn.Module):
    """Stratified layer processing"""
    def __init__(self, d_model: int, num_strata: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_strata = num_strata
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_strata)
        ])
        self.feedforward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(num_strata)
        ])
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Process through different strata
        stratum_outputs = []
        for i in range(self.num_strata):
            # Layer norm
            normed = self.layer_norms[i](hidden_states)
            # Feedforward
            ff_output = self.feedforward[i](normed)
            stratum_outputs.append(ff_output)
        
        # Combine outputs (simple average for now)
        output = sum(stratum_outputs) / len(stratum_outputs)
        
        stratum_info = {
            'stratum_contributions': [torch.norm(out, dim=-1).mean() for out in stratum_outputs]
        }
        
        return output, stratum_info

class StratifiedMoE(nn.Module):
    """Stratified Mixture-of-Experts"""
    def __init__(self, d_model: int, num_experts: int = 4, num_strata: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_strata = num_strata
        
        # Gating network
        self.gate = nn.Linear(d_model, num_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model)
            ) for _ in range(num_experts)
        ])
        
        # Stratum-specific gating
        self.stratum_gate = nn.Linear(d_model, num_strata)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Standard gating
        gating_logits = self.gate(hidden_states)
        gating_probs = F.softmax(gating_logits, dim=-1)
        
        # Stratum gating
        stratum_logits = self.stratum_gate(hidden_states)
        stratum_probs = F.softmax(stratum_logits, dim=-1)
        
        # Expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(hidden_states)
            expert_outputs.append(expert_output)
        
        # Weighted combination
        output = torch.zeros_like(hidden_states)
        for i, expert_out in enumerate(expert_outputs):
            output += expert_out * gating_probs[:, :, i:i+1]
        
        # Apply stratum weighting
        stratum_weighted_output = output * stratum_probs.mean(dim=-1, keepdim=True)
        
        stratum_info = {
            'gating_probs': gating_probs,
            'stratum_probs': stratum_probs,
            'expert_usage': gating_probs.mean(dim=(0, 1))
        }
        
        return stratum_weighted_output, stratum_info

class StratifiedTransformerWrapper(nn.Module):
    """Wrapper for transformer models with stratified components"""
    def __init__(self, model_name: str, stratified_type: str = "none"):
        super().__init__()
        self.model_name = model_name
        self.stratified_type = stratified_type
        
        # Load tokenizer and base model
        if model_name.startswith('distilbert'):
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.base_model = BertModel.from_pretrained(model_name)
            hidden_size = self.base_model.config.hidden_size
        elif model_name.startswith('bert'):
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.base_model = BertModel.from_pretrained(model_name)
            hidden_size = self.base_model.config.hidden_size
        elif model_name.startswith('roberta'):
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.base_model = RobertaModel.from_pretrained(model_name)
            hidden_size = self.base_model.config.hidden_size
        elif model_name.startswith('gpt'):
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.base_model = GPT2Model.from_pretrained(model_name)
            hidden_size = self.base_model.config.hidden_size
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.base_model = AutoModel.from_pretrained(model_name)
            hidden_size = self.base_model.config.hidden_size
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.base_model.resize_token_embeddings(len(self.tokenizer))
        
        # Stratified component
        if stratified_type == "attention":
            self.stratified_component = StratifiedAttention(hidden_size)
        elif stratified_type == "routing":
            self.stratified_component = StratifiedTokenRouter(hidden_size)
        elif stratified_type == "layers":
            self.stratified_component = StratifiedLayerProcessor(hidden_size)
        elif stratified_type == "moe":
            self.stratified_component = StratifiedMoE(hidden_size)
        elif stratified_type == "none":
            self.stratified_component = nn.Identity()
        else:
            raise ValueError(f"Unknown stratified type: {stratified_type}")
        
        # Classification head
        self.classifier = nn.Linear(hidden_size, 3)
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Filter kwargs to only include valid arguments for the base model
        valid_kwargs = {}
        for key, value in kwargs.items():
            if key in ['output_hidden_states', 'output_attentions', 'return_dict']:
                valid_kwargs[key] = value
        
        # Get base model outputs
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **valid_kwargs)
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
        
        # Return logits directly for Hugging Face Trainer compatibility
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return (loss, logits)
        else:
            return logits

class MultidomainDataset(Dataset):
    """Multidomain sentiment dataset"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_multidomain_sentiment(samples_per_domain=200):
    """Load multidomain sentiment dataset"""
    print(f"📚 Loading multidomain sentiment dataset ({samples_per_domain} samples per domain)...")
    
    datasets = {}
    
    # Load datasets
    try:
        datasets['imdb'] = load_dataset('imdb', split='train').select(range(samples_per_domain))
        datasets['rotten_tomatoes'] = load_dataset('rotten_tomatoes', split='train').select(range(samples_per_domain))
        datasets['amazon_polarity'] = load_dataset('amazon_polarity', split='train').select(range(samples_per_domain))
        datasets['sst2'] = load_dataset('glue', 'sst2', split='train').select(range(samples_per_domain))
        datasets['tweet_eval'] = load_dataset('tweet_eval', 'sentiment', split='train').select(range(samples_per_domain))
    except Exception as e:
        print(f"⚠️ Error loading some datasets: {e}")
        # Fallback to synthetic data
        return create_synthetic_multidomain_dataset(samples_per_domain * 5)
    
    # Unify datasets
    def unify_dataset(ds, domain_name, text_field="text"):
        texts = []
        labels = []
        for item in ds:
            if text_field in item:
                texts.append(item[text_field])
                # Convert to 3-class: 0=negative, 1=neutral, 2=positive
                if 'label' in item:
                    if item['label'] == 0:
                        labels.append(0)  # negative
                    elif item['label'] == 1:
                        labels.append(2)  # positive
                    else:
                        labels.append(1)  # neutral
                else:
                    labels.append(1)  # default to neutral
        return texts, labels
    
    all_texts = []
    all_labels = []
    
    for domain_name, ds in datasets.items():
        if domain_name == 'sst2':
            texts, labels = unify_dataset(ds, domain_name, "sentence")
        elif domain_name == 'tweet_eval':
            texts, labels = unify_dataset(ds, domain_name, "text")
        else:
            texts, labels = unify_dataset(ds, domain_name, "text")
        
        all_texts.extend(texts)
        all_labels.extend(labels)
        print(f"  ✅ {domain_name}: {len(texts)} samples")
    
    print(f"✅ Total samples: {len(all_texts)}")
    return {'text': all_texts, 'label': all_labels}

def create_synthetic_multidomain_dataset(num_samples=1000):
    """Create synthetic multidomain dataset"""
    print(f"📝 Creating synthetic multidomain dataset ({num_samples} samples)...")
    
    domains = ['movies', 'products', 'news', 'social', 'reviews']
    texts = []
    labels = []
    
    for i in range(num_samples):
        domain = domains[i % len(domains)]
        label = i % 3  # 0=negative, 1=neutral, 2=positive
        
        if domain == 'movies':
            if label == 0:
                text = f"This movie was terrible and boring. Waste of time."
            elif label == 1:
                text = f"The movie was okay, nothing special."
            else:
                text = f"Amazing movie! Highly recommended."
        elif domain == 'products':
            if label == 0:
                text = f"Poor quality product, broke after one day."
            elif label == 1:
                text = f"Product works as expected, decent quality."
            else:
                text = f"Excellent product, exceeded my expectations!"
        elif domain == 'news':
            if label == 0:
                text = f"Disappointing news about the economy."
            elif label == 1:
                text = f"Neutral report on current events."
            else:
                text = f"Great news! Positive developments ahead."
        elif domain == 'social':
            if label == 0:
                text = f"Having a really bad day today."
            elif label == 1:
                text = f"Just another regular day."
            else:
                text = f"Feeling amazing and grateful today!"
        else:  # reviews
            if label == 0:
                text = f"Would not recommend this to anyone."
            elif label == 1:
                text = f"Average experience, could be better."
            else:
                text = f"Outstanding service, will definitely return!"
        
        texts.append(text)
        labels.append(label)
    
    print(f"✅ Created {len(texts)} synthetic samples")
    return {'text': texts, 'label': labels}

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).astype(np.float32).mean().item()
    return {"accuracy": accuracy}

def run_hf_trainer_stratified_experiment():
    """Run Hugging Face Trainer stratified experiment"""
    print("🚀 Starting Hugging Face Trainer + Accelerate Multi-GPU Stratified Training")
    print("=" * 80)
    print("Hugging Face Trainer + Accelerate + Large Dataset (5000 samples/domain) + 50 Epochs")
    print("=" * 80)
    
    # Initialize accelerator
    accelerator = Accelerator()
    print(f"🔍 Available devices: {torch.cuda.device_count()}")
    print(f"✅ Using {torch.cuda.device_count()} devices")
    
    # Load dataset
    dataset = load_multidomain_sentiment(samples_per_domain=5000)
    texts = dataset['text']
    labels = dataset['label']
    
    print(f"Dataset: {len(texts)} samples, 3 classes")
    
    # Split dataset
    train_size = int(0.8 * len(texts))
    val_size = int(0.1 * len(texts))
    
    train_texts = texts[:train_size]
    train_labels = labels[:train_size]
    val_texts = texts[train_size:train_size + val_size]
    val_labels = labels[train_size:train_size + val_size]
    test_texts = texts[train_size + val_size:]
    test_labels = labels[train_size + val_size:]
    
    print(f"Dataset splits: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")
    
    # Test different stratified approaches
    stratified_types = ["none", "attention", "routing", "layers", "moe"]
    results = {}
    
    for stratified_type in stratified_types:
        print(f"\n🔍 Testing {stratified_type}...")
        
        try:
            # Create model
            model = StratifiedTransformerWrapper(
                model_name="distilbert-base-uncased",
                stratified_type=stratified_type
            )
            
            # Create datasets
            train_dataset = MultidomainDataset(train_texts, train_labels, model.tokenizer)
            val_dataset = MultidomainDataset(val_texts, val_labels, model.tokenizer)
            
            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=model.tokenizer)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"./results/hf_trainer_stratified/{stratified_type}",
                num_train_epochs=50,
                per_device_train_batch_size=32,  # Increased batch size
                per_device_eval_batch_size=64,    # Increased batch size
                warmup_steps=1000,               # Increased warmup for larger dataset
                weight_decay=0.01,
                logging_dir=f"./logs/hf_trainer_stratified/{stratified_type}",
                logging_steps=200,               # Less frequent logging
                eval_strategy="steps",
                eval_steps=1000,                # Less frequent evaluation
                save_strategy="steps",
                save_steps=2000,                # Less frequent saving
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                greater_is_better=True,
                report_to=None,  # Disable wandb/tensorboard
                dataloader_pin_memory=False,
                dataloader_num_workers=2,       # Increased workers for larger dataset
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
            
            # Train model
            print(f"    Starting training for {stratified_type}...")
            trainer.train()
            
            # Evaluate model
            eval_results = trainer.evaluate()
            
            # Test accuracy
            test_dataset = MultidomainDataset(test_texts, test_labels, model.tokenizer)
            test_results = trainer.evaluate(test_dataset)
            
            results[stratified_type] = {
                'eval_loss': eval_results['eval_loss'],
                'test_loss': test_results['eval_loss'],
                'eval_accuracy': eval_results.get('eval_accuracy', 0.0),
                'test_accuracy': test_results.get('eval_accuracy', 0.0),
                'training_time': 0.0,  # Could be tracked if needed
                'stratum_info': 'Available' if stratified_type != "none" else 'None'
            }
            
            print(f"    ✅ {stratified_type}: Test Acc={test_results.get('eval_accuracy', 0.0):.3f}")
            
        except Exception as e:
            print(f"    ❌ Error testing {stratified_type}: {e}")
            results[stratified_type] = {'error': str(e)}
    
    # Print results
    print(f"\n📊 HUGGING FACE TRAINER STRATIFIED RESULTS:")
    print("=" * 60)
    
    baseline_acc = results.get('none', {}).get('test_accuracy', 0.0)
    
    for stratified_type, result in results.items():
        if 'error' in result:
            print(f"{stratified_type}: ❌ Error: {result['error']}")
        else:
            acc = result['test_accuracy']
            if baseline_acc > 0:
                improvement = (acc - baseline_acc) / baseline_acc * 100
                print(f"{stratified_type}:")
                print(f"  Test Accuracy: {acc:.3f}")
                print(f"  Improvement: {improvement:+.1f}%")
                print(f"  Stratum Info: {result['stratum_info']}")
            else:
                print(f"{stratified_type}: Test Accuracy: {acc:.3f}")
    
    # Save results
    os.makedirs("./results/hf_trainer_stratified", exist_ok=True)
    with open("./results/hf_trainer_stratified/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Hugging Face Trainer stratified experiment complete!")
    print(f"📁 Results saved to: ./results/hf_trainer_stratified/results.json")

if __name__ == "__main__":
    run_hf_trainer_stratified_experiment()
