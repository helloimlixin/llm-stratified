#!/usr/bin/env python3
"""
LLaMA-3 + Hugging Face Trainer + Accelerate Multi-GPU Experiment
================================================================
Focused experiment using LLaMA-3.2-1B with:
- Hugging Face Trainer framework
- Accelerate for multi-GPU support
- Multidomain sentiment dataset
- Stratified manifold components
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoTokenizer, AutoModel, TrainingArguments, Trainer, 
    DataCollatorWithPadding
)
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")

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

def load_multidomain_sentiment(samples_per_domain=5000):
    """Load multidomain sentiment dataset"""
    print(f"📚 Loading multidomain sentiment dataset ({samples_per_domain} samples/domain)...")
    
    # Domain configurations - More challenging and nuanced examples
    domains = {
        'imdb': {
            'positive': [
                "The film had some interesting moments, though the pacing felt a bit slow at times.",
                "While not groundbreaking, the movie managed to keep me engaged throughout.",
                "The acting was decent and the story had potential, despite some execution issues.",
                "It's a solid film that delivers on its promises, even if it doesn't exceed them.",
                "The cinematography was quite good, making up for some weaker dialogue moments."
            ],
            'negative': [
                "The movie had potential but ultimately failed to deliver on its initial promise.",
                "Despite some good moments, the film struggled with consistency and focus.",
                "The story felt underdeveloped, leaving several plot threads unresolved.",
                "While the visuals were acceptable, the narrative lacked coherence and depth.",
                "The film tried to do too much and ended up accomplishing very little."
            ]
        },
        'amazon': {
            'positive': [
                "The product works as expected, though the setup process could be simpler.",
                "It serves its purpose well, despite some minor design quirks I noticed.",
                "The quality is acceptable for the price point, meeting basic requirements.",
                "While not premium, it gets the job done and seems reasonably durable.",
                "The functionality is solid, even if the aesthetics aren't particularly impressive."
            ],
            'negative': [
                "The product functions but has some limitations that weren't clearly mentioned.",
                "While it works, the build quality feels somewhat cheaper than expected.",
                "The item does what it claims, though the user experience could be better.",
                "It's functional but lacks some features that would make it more convenient.",
                "The product is adequate but doesn't quite justify the price in my opinion."
            ]
        },
        'yelp': {
            'positive': [
                "The restaurant has decent food, though the service can be inconsistent.",
                "The atmosphere is pleasant and the menu offers some interesting options.",
                "While not exceptional, the dining experience was satisfactory overall.",
                "The food quality is acceptable and the location is convenient.",
                "It's a reasonable choice for a meal, with portions that are fairly generous."
            ],
            'negative': [
                "The restaurant has potential but execution falls short of expectations.",
                "While the ambiance is nice, the food quality doesn't quite match the setting.",
                "The service was slow and the dishes lacked the flavor I was hoping for.",
                "The menu looked promising but the actual dishes were somewhat underwhelming.",
                "The location is good but the overall experience left something to be desired."
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
                variation = f" The experience was quite nuanced with both positive and negative aspects."
                text = template + variation
                
                # Add some label noise (5% chance of wrong label) to make it more challenging
                import random
                if random.random() < 0.05:
                    label = 1 - label  # Flip the label
                
                texts.append(text)
                labels.append(label)
    
    print(f"✅ Created {len(texts)} samples across {len(domains)} domains")
    return texts, labels

class GPTStratifiedWrapper(nn.Module):
    """GPT-2 wrapper with stratified components for HF Trainer"""
    
    def __init__(self, model_name="gpt2", stratified_type="none"):
        super().__init__()
        self.model_name = model_name
        self.stratified_type = stratified_type
        
        # Load GPT-2 model and tokenizer
        print(f"🤖 Loading {model_name}...")
        from transformers import GPT2Tokenizer, GPT2Model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # Load model normally - Accelerate will handle device placement (full precision)
        self.base_model = GPT2Model.from_pretrained(model_name)
        self.hidden_size = self.base_model.config.hidden_size
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Stratified component
        if stratified_type == "attention":
            num_heads = max(1, self.hidden_size // 64)
            self.stratified_component = StratifiedAttention(self.hidden_size, num_heads, num_strata=3)
        elif stratified_type == "none":
            self.stratified_component = nn.Identity()
        else:
            raise ValueError(f"Stratified type {stratified_type} not implemented")
        
        # Classification head (Accelerate will handle device placement)
        self.classifier = nn.Linear(self.hidden_size, 2)  # Binary sentiment classification
        
        # Initialize classifier weights properly
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.base_model.gradient_checkpointing_enable()
        
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.base_model.gradient_checkpointing_disable()
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass compatible with HF Trainer"""
        # Get base model outputs
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs.last_hidden_state
        
        # Apply stratified processing
        if self.stratified_type == "none":
            enhanced_hidden_states = hidden_states
        else:
            enhanced_hidden_states, _ = self.stratified_component(hidden_states)
        
        # Classification (use last token)
        pooled_output = enhanced_hidden_states[:, -1, :]
        
        # Add dropout for regularization
        pooled_output = F.dropout(pooled_output, p=0.1, training=self.training)
        
        # Scale down the hidden states to prevent huge logits
        pooled_output = pooled_output * 0.1
        
        logits = self.classifier(pooled_output)
        
        # Compute loss if labels provided
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            
            # Debug: Check for problematic values
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Loss is {loss.item()}")
                print(f"Logits range: {logits.min().item():.3f} to {logits.max().item():.3f}")
                print(f"Hidden states range: {pooled_output.min().item():.3f} to {pooled_output.max().item():.3f}")
            
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

def compute_metrics(eval_pred):
    """Compute comprehensive evaluation metrics"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(labels, predictions, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    return {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_class_0": precision_per_class[0] if len(precision_per_class) > 0 else 0,
        "recall_class_0": recall_per_class[0] if len(recall_per_class) > 0 else 0,
        "f1_class_0": f1_per_class[0] if len(f1_per_class) > 0 else 0,
        "precision_class_1": precision_per_class[1] if len(precision_per_class) > 1 else 0,
        "recall_class_1": recall_per_class[1] if len(recall_per_class) > 1 else 0,
        "f1_class_1": f1_per_class[1] if len(f1_per_class) > 1 else 0,
        "true_negatives": int(cm[0,0]) if cm.shape == (2,2) else 0,
        "false_positives": int(cm[0,1]) if cm.shape == (2,2) else 0,
        "false_negatives": int(cm[1,0]) if cm.shape == (2,2) else 0,
        "true_positives": int(cm[1,1]) if cm.shape == (2,2) else 0,
    }

def run_llama3_hf_trainer_experiment():
    """Run GPT-2 + HF Trainer + Accelerate experiment"""
    print("🚀 GPT-2 + Hugging Face Trainer + Accelerate Experiment")
    print("=" * 80)
    print("Features:")
    print("- GPT-2 model")
    print("- Hugging Face Trainer framework")
    print("- Accelerate for multi-GPU support")
    print("- Multidomain sentiment classification")
    print("- Stratified manifold components")
    print("=" * 80)
    
    # Load dataset
    texts, labels = load_multidomain_sentiment(samples_per_domain=1000)  # Challenging dataset for comprehensive evaluation
    
    # Test stratified types - comprehensive evaluation
    stratified_types = ["none", "attention", "routing", "layers", "moe"]
    results = {}
    
    for stratified_type in stratified_types:
        print(f"\n🔍 Testing {stratified_type} stratified type...")
        print("-" * 50)
        
        try:
            # Create model
            model = GPTStratifiedWrapper(model_name="gpt2", stratified_type=stratified_type)
            tokenizer = model.tokenizer
            
            # Prepare dataset
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors="pt"
                )
            
            # Create HF Dataset
            dataset = Dataset.from_dict({"text": texts, "labels": labels})
            tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
            
            # Train/validation split
            train_size = int(0.8 * len(tokenized_dataset))
            train_dataset = tokenized_dataset.select(range(train_size))
            val_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
            
            print(f"📊 Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
            
            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
            
            # Training arguments - optimized for GPT-2 multi-GPU training
            training_args = TrainingArguments(
                output_dir=f"./results/gpt2_hf_trainer/{stratified_type}",
                num_train_epochs=10,  # Reasonable epochs for challenging dataset
                per_device_train_batch_size=8,   # Larger batch size for GPT-2
                per_device_eval_batch_size=16,   # Larger eval batch size
                learning_rate=5e-5,              # Standard learning rate for GPT-2
                warmup_steps=200,                # Standard warmup
                weight_decay=0.01,
                logging_dir=f"./logs/gpt2_hf_trainer/{stratified_type}",
                logging_steps=100,               # More frequent logging for monitoring
                eval_strategy="steps",
                eval_steps=500,                 # More frequent evaluation
                save_strategy="steps",
                save_steps=500,                 # More frequent saving
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                greater_is_better=True,
                report_to=None,  # Disable wandb/tensorboard
                dataloader_pin_memory=False,
                dataloader_num_workers=2,        # More workers for efficiency
                # No mixed precision - use full FP32
                gradient_accumulation_steps=2,   # Effective batch size = 8 * 2 * 2 GPUs = 32
                save_total_limit=2,              # Save more models
                remove_unused_columns=False,
                max_grad_norm=1.0,               # Standard gradient clipping
                dataloader_drop_last=True,       # Drop incomplete batches
                ddp_find_unused_parameters=False, # Optimize DDP
                dataloader_persistent_workers=True, # Keep workers alive
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
            print(f"🚀 Training GPT-2 with {stratified_type} stratified components...")
            train_result = trainer.train()
            
            # Evaluate
            eval_result = trainer.evaluate()
            
            # Store comprehensive results
            results[stratified_type] = {
                "train_loss": train_result.training_loss,
                "train_runtime": train_result.training_loss,
                "train_samples_per_second": train_result.train_samples_per_second,
                "eval_loss": eval_result["eval_loss"],
                "eval_accuracy": eval_result["eval_accuracy"],
                "eval_precision_weighted": eval_result["eval_precision_weighted"],
                "eval_recall_weighted": eval_result["eval_recall_weighted"],
                "eval_f1_weighted": eval_result["eval_f1_weighted"],
                "eval_precision_macro": eval_result["eval_precision_macro"],
                "eval_recall_macro": eval_result["eval_recall_macro"],
                "eval_f1_macro": eval_result["eval_f1_macro"],
                "eval_precision_class_0": eval_result["eval_precision_class_0"],
                "eval_recall_class_0": eval_result["eval_recall_class_0"],
                "eval_f1_class_0": eval_result["eval_f1_class_0"],
                "eval_precision_class_1": eval_result["eval_precision_class_1"],
                "eval_recall_class_1": eval_result["eval_recall_class_1"],
                "eval_f1_class_1": eval_result["eval_f1_class_1"],
                "eval_true_negatives": eval_result["eval_true_negatives"],
                "eval_false_positives": eval_result["eval_false_positives"],
                "eval_false_negatives": eval_result["eval_false_negatives"],
                "eval_true_positives": eval_result["eval_true_positives"],
                "status": "success"
            }
            
            print(f"✅ {stratified_type}: Accuracy = {eval_result['eval_accuracy']:.3f}, F1 = {eval_result['eval_f1_weighted']:.3f}, Precision = {eval_result['eval_precision_weighted']:.3f}, Recall = {eval_result['eval_recall_weighted']:.3f}")
            
            # Clear memory after each experiment
            del model, trainer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error with {stratified_type}: {e}")
            results[stratified_type] = {
                "error": str(e),
                "status": "error"
            }
            
            # Clear memory after error
            try:
                del model
                torch.cuda.empty_cache()
            except:
                pass
    
    # Save results
    os.makedirs("./results/gpt2_hf_trainer", exist_ok=True)
    with open("./results/gpt2_hf_trainer/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate summary
    print("\n" + "=" * 80)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 80)
    
    for stratified_type, result in results.items():
        if result["status"] == "success":
            print(f"🤖 GPT-2 + {stratified_type}:")
            print(f"   📊 Accuracy: {result['eval_accuracy']:.3f}")
            print(f"   📊 F1 (Weighted): {result['eval_f1_weighted']:.3f}")
            print(f"   📊 F1 (Macro): {result['eval_f1_macro']:.3f}")
            print(f"   📊 Precision (Weighted): {result['eval_precision_weighted']:.3f}")
            print(f"   📊 Recall (Weighted): {result['eval_recall_weighted']:.3f}")
            print(f"   📊 Training Loss: {result['train_loss']:.4f}")
            print(f"   📊 Eval Loss: {result['eval_loss']:.4f}")
            print(f"   ⏱️  Training Speed: {result['train_samples_per_second']:.1f} samples/sec")
            print(f"   🔢 Confusion Matrix: TN={result['eval_true_negatives']}, FP={result['eval_false_positives']}, FN={result['eval_false_negatives']}, TP={result['eval_true_positives']}")
        else:
            print(f"❌ GPT-2 + {stratified_type}: ERROR - {result['error']}")
        print()
    
    print(f"\n✅ Results saved to: ./results/gpt2_hf_trainer/results.json")
    return results

if __name__ == "__main__":
    run_llama3_hf_trainer_experiment()
