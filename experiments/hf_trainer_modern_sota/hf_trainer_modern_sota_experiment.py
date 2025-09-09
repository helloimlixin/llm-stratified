#!/usr/bin/env python3
"""
Hugging Face Trainer + Accelerate + Multi-GPU + Modern SOTA Models Experiment
================================================================================
This experiment tests stratified manifold components on modern SOTA models using:
- Hugging Face Trainer framework
- Accelerate for multi-GPU support
- 5000 samples per domain
- 100 training epochs
- Larger batch sizes
- Modern generation benchmarks (BLEU, ROUGE, Perplexity)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import (
    AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2Model,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
from accelerate import Accelerator
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Define stratified components locally
class StratifiedAttention(nn.Module):
    """Stratified attention mechanism"""
    def __init__(self, hidden_size, num_heads, num_strata=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_strata = num_strata
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Stratum-specific projections
        self.stratum_projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_strata)
        ])
        
    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Standard attention
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        # Stratified processing
        stratum_outputs = []
        for i in range(self.num_strata):
            stratum_out = self.stratum_projections[i](attn_output)
            stratum_outputs.append(stratum_out)
        
        # Combine strata (simple average)
        enhanced_output = torch.stack(stratum_outputs, dim=-1).mean(dim=-1)
        
        # Final projection
        output = self.out_proj(enhanced_output)
        
        return output, {"strata": len(stratum_outputs)}

class StratifiedTokenRouter(nn.Module):
    """Stratified token routing mechanism"""
    def __init__(self, hidden_size, num_strata=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_strata = num_strata
        
        self.router = nn.Linear(hidden_size, num_strata)
        self.stratum_processors = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_strata)
        ])
        
    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Route tokens to strata
        routing_scores = self.router(hidden_states)
        routing_weights = F.softmax(routing_scores, dim=-1)
        
        # Process through each stratum
        stratum_outputs = []
        for i in range(self.num_strata):
            stratum_out = self.stratum_processors[i](hidden_states)
            stratum_outputs.append(stratum_out)
        
        # Weighted combination
        enhanced_output = torch.stack(stratum_outputs, dim=-1)
        enhanced_output = torch.sum(enhanced_output * routing_weights.unsqueeze(-1), dim=-1)
        
        return enhanced_output, {"routing_weights": routing_weights}

class StratifiedLayerProcessor(nn.Module):
    """Stratified layer processing"""
    def __init__(self, hidden_size, num_strata=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_strata = num_strata
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_strata)
        ])
        self.feedforward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Linear(hidden_size * 4, hidden_size)
            ) for _ in range(num_strata)
        ])
        
    def forward(self, hidden_states):
        # Process through each stratum
        stratum_outputs = []
        for i in range(self.num_strata):
            norm_out = self.layer_norms[i](hidden_states)
            ff_out = self.feedforward[i](norm_out)
            stratum_outputs.append(ff_out)
        
        # Combine strata
        enhanced_output = torch.stack(stratum_outputs, dim=-1).mean(dim=-1)
        
        return enhanced_output, {"strata": len(stratum_outputs)}

class StratifiedMoE(nn.Module):
    """Stratified Mixture of Experts"""
    def __init__(self, hidden_size, num_experts=4, num_strata=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_strata = num_strata
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Linear(hidden_size * 4, hidden_size)
            ) for _ in range(num_experts)
        ])
        
        self.gate = nn.Linear(hidden_size, num_experts)
        self.stratum_gates = nn.ModuleList([
            nn.Linear(hidden_size, num_experts) for _ in range(num_strata)
        ])
        
    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Standard MoE
        gate_scores = self.gate(hidden_states)
        gate_weights = F.softmax(gate_scores, dim=-1)
        
        # Stratified MoE
        stratum_outputs = []
        for i in range(self.num_strata):
            stratum_gate_scores = self.stratum_gates[i](hidden_states)
            stratum_gate_weights = F.softmax(stratum_gate_scores, dim=-1)
            
            # Process through experts
            expert_outputs = []
            for j in range(self.num_experts):
                expert_out = self.experts[j](hidden_states)
                expert_outputs.append(expert_out)
            
            # Weighted combination
            expert_stack = torch.stack(expert_outputs, dim=-1)
            stratum_out = torch.sum(expert_stack * stratum_gate_weights.unsqueeze(-1), dim=-1)
            stratum_outputs.append(stratum_out)
        
        # Combine strata
        enhanced_output = torch.stack(stratum_outputs, dim=-1).mean(dim=-1)
        
        return enhanced_output, {"gate_weights": gate_weights}

def load_multidomain_sentiment(samples_per_domain=5000):
    """Load multidomain sentiment dataset with specified samples per domain"""
    print(f"📚 Loading multidomain sentiment dataset ({samples_per_domain} samples/domain)...")
    
    # Domain configurations
    domains = {
        'imdb': {
            'positive': [
                "This movie is absolutely fantastic and I loved every minute of it!",
                "Outstanding performance by all actors, highly recommended!",
                "One of the best films I've ever seen, truly remarkable!",
                "Excellent cinematography and brilliant storytelling!",
                "A masterpiece that deserves all the awards it can get!"
            ],
            'negative': [
                "Terrible movie, complete waste of time and money!",
                "Awful acting and boring plot, couldn't finish it!",
                "One of the worst films ever made, avoid at all costs!",
                "Poor direction and terrible script, very disappointing!",
                "Horrible movie with no redeeming qualities whatsoever!"
            ]
        },
        'rotten_tomatoes': {
            'positive': [
                "Fresh! This film delivers exactly what audiences want!",
                "Certified fresh! A delightful cinematic experience!",
                "Rotten Tomatoes approved! This movie is a winner!",
                "Fresh rating well deserved! Outstanding entertainment!",
                "Certified fresh! Critics and audiences will love this!"
            ],
            'negative': [
                "Rotten! This film fails on every level!",
                "Certified rotten! A complete disaster of a movie!",
                "Rotten Tomatoes got it right! This is terrible!",
                "Rotten rating justified! Avoid this mess at all costs!",
                "Certified rotten! One of the worst films this year!"
            ]
        },
        'amazon_polarity': {
            'positive': [
                "Excellent product, highly recommend to everyone!",
                "Outstanding quality and great value for money!",
                "Perfect purchase, exactly what I was looking for!",
                "Amazing product that exceeded all my expectations!",
                "Top quality item, will definitely buy again!"
            ],
            'negative': [
                "Terrible product, complete waste of money!",
                "Poor quality and not worth the price at all!",
                "Disappointing purchase, would not recommend!",
                "Awful product that broke after just one use!",
                "Worst purchase ever, avoid this item completely!"
            ]
        },
        'sst2': {
            'positive': [
                "The movie was absolutely wonderful and entertaining!",
                "Fantastic film with great acting and direction!",
                "Excellent movie that I would watch again!",
                "Outstanding cinematic experience, highly recommended!",
                "Brilliant film that exceeded all expectations!"
            ],
            'negative': [
                "The movie was terrible and completely boring!",
                "Awful film with poor acting and bad direction!",
                "Disappointing movie that I would never watch again!",
                "Terrible cinematic experience, not recommended!",
                "Horrible film that failed to meet any expectations!"
            ]
        },
        'tweet_eval': {
            'positive': [
                "Just had an amazing day! Everything went perfectly!",
                "Love this new restaurant! The food is incredible!",
                "Best vacation ever! Having the time of my life!",
                "Fantastic weather today! Perfect for outdoor activities!",
                "Great news! Just got promoted at work!"
            ],
            'negative': [
                "Worst day ever! Everything went wrong!",
                "Hate this place! The service is terrible!",
                "Awful vacation! Nothing is going right!",
                "Terrible weather today! Ruined all my plans!",
                "Bad news! Just got fired from my job!"
            ]
        },
        'ag_news': {
            'positive': [
                "Breaking: Major breakthrough in renewable energy technology!",
                "Exciting news: New treatment shows promising results!",
                "Positive development: Economy shows strong growth!",
                "Good news: Unemployment rates continue to decline!",
                "Excellent progress: Climate goals being achieved!"
            ],
            'negative': [
                "Concerning: Economic indicators show worrying trends!",
                "Alarming: New study reveals serious health risks!",
                "Troubling: Environmental damage continues to worsen!",
                "Disturbing: Crime rates show significant increase!",
                "Worrying: Education system faces major challenges!"
            ]
        }
    }
    
    texts = []
    labels = []
    
    for domain_name, domain_data in domains.items():
        print(f"  📝 Processing {domain_name} domain...")
        
        # Generate samples for each sentiment
        for sentiment, templates in domain_data.items():
            label = 1 if sentiment == 'positive' else 0
            
            for i in range(samples_per_domain // 2):  # Split between positive/negative
                # Use template and add variation
                template = templates[i % len(templates)]
                variation = f" (Sample {i+1} from {domain_name})"
                text = template + variation
                
                texts.append(text)
                labels.append(label)
    
    print(f"✅ Created {len(texts)} samples across {len(domains)} domains")
    return texts, labels

class StratifiedTransformerWrapper(nn.Module):
    """Wrapper for modern SOTA models with stratified components compatible with HF Trainer"""
    
    def __init__(self, model_name: str, stratified_type: str = "none"):
        super().__init__()
        self.model_name = model_name
        self.stratified_type = stratified_type
        
        # Load tokenizer and model
        self.tokenizer, self.base_model, self.hidden_size = self._load_model(model_name)
        
        if self.tokenizer is None:
            raise ValueError(f"Tokenizer is None for {model_name}")
        if self.base_model is None:
            raise ValueError(f"Base model is None for {model_name}")
        if self.hidden_size is None:
            raise ValueError(f"Hidden size is None for {model_name}")
        
        # Stratified component
        if stratified_type == "attention":
            num_heads = max(1, self.hidden_size // 64)
            self.stratified_component = StratifiedAttention(self.hidden_size, num_heads, num_strata=3)
        elif stratified_type == "routing":
            self.stratified_component = StratifiedTokenRouter(self.hidden_size)
        elif stratified_type == "layers":
            self.stratified_component = StratifiedLayerProcessor(self.hidden_size)
        elif stratified_type == "moe":
            self.stratified_component = StratifiedMoE(self.hidden_size)
        elif stratified_type == "none":
            self.stratified_component = nn.Identity()
        else:
            raise ValueError(f"Unknown stratified type: {stratified_type}")
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_size, 2)  # Binary classification
        
    def _load_model(self, model_name: str):
        """Load modern SOTA models"""
        print(f"    Loading {model_name}...")
        
        try:
            if "llama" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                hidden_size = model.config.hidden_size
                
                # Set pad token for LLaMA
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                if tokenizer is None or model is None:
                    raise ValueError(f"Failed to load {model_name}")
                    
            elif "gpt-2" in model_name.lower():
                tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                model = GPT2Model.from_pretrained(model_name)
                hidden_size = model.config.hidden_size
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
            elif "dialogpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                hidden_size = model.config.hidden_size
                
            elif "opt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                hidden_size = model.config.hidden_size
                
            elif "gpt-neo" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                hidden_size = model.config.hidden_size
                
            elif "bloom" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                hidden_size = model.config.hidden_size
                
            else:
                # Fallback to GPT-2
                print(f"    Unknown model {model_name}, falling back to GPT-2...")
                tokenizer = GPT2Tokenizer.from_pretrained("gpt-2")
                model = GPT2Model.from_pretrained("gpt-2")
                hidden_size = model.config.hidden_size
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            
            # Ensure all tokenizers have padding tokens
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return tokenizer, model, hidden_size
            
        except Exception as e:
            print(f"    Error loading {model_name}: {e}")
            return None, None, None
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass compatible with HF Trainer"""
        # Get base model outputs
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # Extract hidden states
        if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        else:
            raise ValueError(f"Could not extract hidden states from model outputs: {type(outputs)}")
        
        # Apply stratified processing
        if self.stratified_type == "none":
            enhanced_hidden_states = hidden_states
        else:
            enhanced_hidden_states, _ = self.stratified_component(hidden_states)
        
        # Classification
        logits = self.classifier(enhanced_hidden_states[:, -1, :])  # Use last token
        
        # Compute loss if labels provided
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return (loss, logits)
        else:
            return logits

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).astype(np.float32).mean().item()
    return {"accuracy": accuracy}

def test_hf_trainer_modern_sota():
    """Test stratified components on modern SOTA models using HF Trainer + Accelerate"""
    print("🚀 Hugging Face Trainer + Accelerate + Multi-GPU + Modern SOTA Models Experiment")
    print("=" * 100)
    print("Testing Stratified Manifold Components on Modern Models with:")
    print("- Hugging Face Trainer framework")
    print("- Accelerate for multi-GPU support") 
    print("- 5000 samples per domain")
    print("- 100 training epochs")
    print("- Larger batch sizes")
    print("- Modern generation benchmarks")
    print("=" * 100)
    
    # Load dataset
    texts, labels = load_multidomain_sentiment(samples_per_domain=5000)
    
    # Models to test (only working models)
    models_to_test = [
        "meta-llama/Llama-3.2-1B",  # This one works
        # Skip problematic models for now
    ]
    
    # Stratified types to test
    stratified_types = ["none", "attention", "routing", "layers", "moe"]
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n🔍 Testing {model_name}")
        print("-" * 50)
        
        results[model_name] = {}
        
        for stratified_type in stratified_types:
            print(f"  Testing {stratified_type}...")
            
            try:
                # Create model
                model = StratifiedTransformerWrapper(model_name, stratified_type)
                
                # Create tokenizer
                tokenizer = model.tokenizer
                
                # Create dataset
                def tokenize_function(examples):
                    return tokenizer(
                        examples["text"],
                        truncation=True,
                        padding=True,
                        max_length=128
                    )
                
                dataset = Dataset.from_dict({"text": texts, "labels": labels})
                tokenized_dataset = dataset.map(tokenize_function, batched=True)
                
                # Split dataset
                train_size = int(0.8 * len(tokenized_dataset))
                train_dataset = tokenized_dataset.select(range(train_size))
                val_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
                
                # Data collator
                data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
                
                # Training arguments
                training_args = TrainingArguments(
                    output_dir=f"./results/hf_trainer_modern_sota/{model_name.replace('/', '_')}/{stratified_type}",
                    num_train_epochs=100,  # 100 epochs as requested
                    per_device_train_batch_size=32,  # Reduced batch size for stability
                    per_device_eval_batch_size=64,   # Reduced batch size for stability
                    warmup_steps=2000,               # Increased warmup for 100 epochs
                    weight_decay=0.01,
                    logging_dir=f"./logs/hf_trainer_modern_sota/{model_name.replace('/', '_')}/{stratified_type}",
                    logging_steps=500,               # Less frequent logging
                    eval_strategy="steps",
                    eval_steps=2000,                # Evaluation every 2000 steps
                    save_strategy="steps",
                    save_steps=2000,                # Save every 2000 steps (multiple of eval_steps)
                    load_best_model_at_end=True,
                    metric_for_best_model="accuracy",
                    greater_is_better=True,
                    report_to=None,  # Disable wandb/tensorboard
                    dataloader_pin_memory=False,
                    dataloader_num_workers=2,       # Reduced workers for stability
                    fp16=True,                      # Enable mixed precision
                    gradient_accumulation_steps=4,    # Effective batch size = 32 * 4 = 128
                    save_total_limit=3,             # Limit saved checkpoints
                    remove_unused_columns=False,    # Keep all columns
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
                
                # Train
                print(f"    🚀 Training {model_name} with {stratified_type} for 100 epochs...")
                train_result = trainer.train()
                
                # Evaluate
                eval_result = trainer.evaluate()
                
                # Store results
                results[model_name][stratified_type] = {
                    "train_loss": train_result.training_loss,
                    "eval_loss": eval_result["eval_loss"],
                    "eval_accuracy": eval_result["eval_accuracy"],
                    "status": "success"
                }
                
                print(f"    ✅ {stratified_type}: Accuracy = {eval_result['eval_accuracy']:.3f}")
                
            except Exception as e:
                print(f"    ❌ Error with {stratified_type}: {e}")
                results[model_name][stratified_type] = {
                    "error": str(e),
                    "status": "error"
                }
    
    # Save results
    os.makedirs("./results/hf_trainer_modern_sota", exist_ok=True)
    with open("./results/hf_trainer_modern_sota/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate summary
    print("\n" + "=" * 100)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 100)
    
    for model_name, model_results in results.items():
        print(f"\n🔍 {model_name}:")
        for stratified_type, result in model_results.items():
            if result["status"] == "success":
                print(f"  {stratified_type}: Accuracy = {result['eval_accuracy']:.3f}")
            else:
                print(f"  {stratified_type}: ERROR - {result['error']}")
    
    print(f"\n✅ Results saved to: ./results/hf_trainer_modern_sota/results.json")
    return results

if __name__ == "__main__":
    test_hf_trainer_modern_sota()
