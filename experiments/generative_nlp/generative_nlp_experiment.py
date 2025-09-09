"""
Modern Generative NLP Tasks with Geometric Regularization
Testing geometric regularization on text generation, language modeling, and completion tasks

This experiment tests:
1. Text generation with GPT-2
2. Language modeling tasks
3. Text completion and continuation
4. Comparison with standard generative models
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
    AutoTokenizer, AutoModelForCausalLM, AutoModel,
    GPT2Tokenizer, GPT2LMHeadModel, GPT2Model,
    T5Tokenizer, T5ForConditionalGeneration, T5Model,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from geometric_tools.immediate_improvements import GeometricRegularizationLoss

class GenerativeModelGeometricWrapper(nn.Module):
    """
    Wrapper for generative models with geometric regularization
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
            hidden_size = 768  # Default for BERT-like models
        
        self.geometric_layer = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Get base model outputs
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        
        # Apply geometric enhancement to hidden states
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # Use last hidden state
            hidden_states = outputs.hidden_states[-1]
            
            # Apply geometric enhancement
            geometric_enhanced = self.geometric_layer(hidden_states)
            enhanced_hidden_states = hidden_states + 0.1 * geometric_enhanced
            
            # Recompute logits with enhanced hidden states
            if hasattr(self.base_model, 'lm_head'):
                enhanced_logits = self.base_model.lm_head(enhanced_hidden_states)
            else:
                enhanced_logits = outputs.logits
            
            # Create new outputs object
            class EnhancedOutputs:
                def __init__(self, logits, hidden_states, loss=None):
                    self.logits = logits
                    self.hidden_states = hidden_states
                    self.loss = loss
            
            enhanced_outputs = EnhancedOutputs(enhanced_logits, enhanced_hidden_states)
            
            # Add geometric loss if provided
            if self.geometric_loss is not None and labels is not None:
                geometric_losses = self.geometric_loss(enhanced_hidden_states, enhanced_logits, labels)
                if outputs.loss is not None:
                    enhanced_outputs.loss = outputs.loss + geometric_losses['total_geometric']
                else:
                    enhanced_outputs.loss = geometric_losses['total_geometric']
            
            return enhanced_outputs
        else:
            # Fallback to original outputs
            return outputs
    
    def generate(self, input_ids, attention_mask=None, max_length=50, **kwargs):
        """Generate text using the enhanced model"""
        with torch.no_grad():
            return self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                **kwargs
            )

def create_generative_datasets():
    """
    Create datasets for generative tasks
    """
    print("📚 Creating generative NLP datasets...")
    
    datasets = {}
    
    # 1. Text Completion Dataset
    completion_prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "The most important skill for the 21st century is",
        "Climate change represents one of the greatest challenges",
        "The key to success in life is",
        "Education should focus on",
        "The role of government in society is",
        "Technology has transformed the way we",
        "The most significant invention of the modern era is",
        "Human creativity combined with artificial intelligence can"
    ]
    
    completion_targets = [
        "revolutionary and will transform every aspect of human life",
        "we must adapt quickly to new paradigms and opportunities",
        "critical thinking and adaptability",
        "facing humanity today and requires immediate action",
        "perseverance, learning, and helping others",
        "developing critical thinking and problem-solving abilities",
        "protecting citizens and promoting the common good",
        "communicate, work, and learn",
        "the internet and digital connectivity",
        "solve complex problems and create innovative solutions"
    ]
    
    datasets['text_completion'] = {
        'prompts': completion_prompts,
        'targets': completion_targets
    }
    
    # 2. Language Modeling Dataset (next token prediction)
    lm_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Natural language processing enables computers to understand human language",
        "Deep learning uses neural networks with multiple layers",
        "Computer vision allows machines to interpret visual information",
        "Data science combines statistics, programming, and domain expertise",
        "Cloud computing provides scalable computing resources over the internet",
        "Cybersecurity protects digital systems from malicious attacks",
        "Blockchain technology enables secure and transparent transactions",
        "Quantum computing promises exponential improvements in computational power"
    ]
    
    datasets['language_modeling'] = {
        'texts': lm_texts
    }
    
    # 3. Text Generation Dataset (creative writing)
    creative_prompts = [
        "Write a short story about a robot who learns to dream",
        "Describe a day in the life of an AI researcher",
        "Create a dialogue between a human and an advanced AI",
        "Write about the impact of technology on future society",
        "Describe a world where humans and AI work together"
    ]
    
    datasets['creative_writing'] = {
        'prompts': creative_prompts
    }
    
    print(f"✅ Created {len(datasets)} generative datasets")
    return datasets

def evaluate_text_generation(model, tokenizer, prompts, targets, max_length=50):
    """
    Evaluate text generation quality
    """
    model.eval()
    generated_texts = []
    
    with torch.no_grad():
        for prompt in prompts:
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
            
            # Generate text
            generated_ids = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode generated text
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
    
    return generated_texts

def evaluate_language_modeling(model, tokenizer, texts):
    """
    Evaluate language modeling performance
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            # Tokenize text
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
            
            # Create labels (shifted by 1 for next token prediction)
            labels = inputs['input_ids'].clone()
            labels[:, :-1] = labels[:, 1:].clone()
            labels[:, -1] = tokenizer.eos_token_id
            
            # Forward pass
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)
            
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                total_loss += outputs.loss.item() * labels.size(1)
                total_tokens += labels.size(1)
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity

def test_generative_models():
    """
    Test geometric regularization on generative models
    """
    print("🚀 Testing Geometric Regularization on Generative NLP Models")
    print("=" * 70)
    
    # Model configurations for generative tasks
    model_configs = [
        {
            'name': 'gpt2',
            'type': 'causal_lm',
            'description': 'GPT-2 Small (117M parameters)'
        },
        {
            'name': 'gpt2-medium',
            'type': 'causal_lm', 
            'description': 'GPT-2 Medium (345M parameters)'
        }
    ]
    
    results = {}
    
    # Create generative datasets
    datasets = create_generative_datasets()
    
    for config in model_configs:
        model_name = config['name']
        model_type = config['type']
        description = config['description']
        
        print(f"\n🔍 Testing {description}...")
        
        try:
            # Load tokenizer and model
            print(f"  Loading {model_name}...")
            
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            base_model = GPT2LMHeadModel.from_pretrained(model_name)
            
            # Set pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print(f"  Testing standard {model_name}...")
            
            # Test standard model on text completion
            standard_completions = evaluate_text_generation(
                base_model, tokenizer, 
                datasets['text_completion']['prompts'][:5],  # Use first 5 for speed
                datasets['text_completion']['targets'][:5]
            )
            
            # Test standard model on language modeling
            standard_loss, standard_perplexity = evaluate_language_modeling(
                base_model, tokenizer,
                datasets['language_modeling']['texts'][:5]  # Use first 5 for speed
            )
            
            print(f"    Standard - Loss: {standard_loss:.4f}, Perplexity: {standard_perplexity:.2f}")
            
            # Test improved model with geometric regularization
            print(f"  Testing improved {model_name} with geometric regularization...")
            
            geometric_loss = GeometricRegularizationLoss(
                lambda_strata=0.001,
                lambda_curvature=0.001,
                lambda_manifold=0.0005
            )
            
            improved_model = GenerativeModelGeometricWrapper(base_model, model_name, geometric_loss)
            
            # Test improved model on text completion
            improved_completions = evaluate_text_generation(
                improved_model, tokenizer,
                datasets['text_completion']['prompts'][:5],
                datasets['text_completion']['targets'][:5]
            )
            
            # Test improved model on language modeling
            improved_loss, improved_perplexity = evaluate_language_modeling(
                improved_model, tokenizer,
                datasets['language_modeling']['texts'][:5]
            )
            
            print(f"    Improved - Loss: {improved_loss:.4f}, Perplexity: {improved_perplexity:.2f}")
            
            # Calculate improvement
            loss_improvement = (improved_loss - standard_loss) / standard_loss * 100 if standard_loss > 0 else 0
            perplexity_improvement = (improved_perplexity - standard_perplexity) / standard_perplexity * 100 if standard_perplexity > 0 else 0
            
            print(f"    Loss Improvement: {loss_improvement:.2f}%")
            print(f"    Perplexity Improvement: {perplexity_improvement:.2f}%")
            
            # Sample generated text comparison
            print(f"    Sample Generation Comparison:")
            print(f"      Prompt: '{datasets['text_completion']['prompts'][0]}'")
            print(f"      Standard: '{standard_completions[0][:100]}...'")
            print(f"      Improved: '{improved_completions[0][:100]}...'")
            
            results[model_name] = {
                'description': description,
                'standard_loss': standard_loss,
                'standard_perplexity': standard_perplexity,
                'improved_loss': improved_loss,
                'improved_perplexity': improved_perplexity,
                'loss_improvement': loss_improvement,
                'perplexity_improvement': perplexity_improvement,
                'standard_completions': standard_completions,
                'improved_completions': improved_completions,
                'model_type': model_type
            }
            
        except Exception as e:
            print(f"  ❌ Error testing {model_name}: {e}")
            results[model_name] = {
                'description': description,
                'error': str(e),
                'model_type': model_type
            }
    
    return results

def run_generative_nlp_experiment():
    """
    Run the generative NLP experiment
    """
    print("🚀 Starting Generative NLP Tasks with Geometric Regularization")
    print("=" * 80)
    print("Testing geometric regularization on text generation and language modeling")
    print("=" * 80)
    
    # Test generative models
    results = test_generative_models()
    
    # Print results
    print("\n📊 GENERATIVE NLP RESULTS:")
    print("=" * 60)
    
    successful_tests = 0
    total_loss_improvement = 0
    total_perplexity_improvement = 0
    loss_improvements = []
    perplexity_improvements = []
    
    for model_name, result in results.items():
        print(f"\n{result['description']}:")
        
        if 'error' in result:
            print(f"  ❌ Error: {result['error']}")
        else:
            print(f"  Standard: Loss={result['standard_loss']:.4f}, Perplexity={result['standard_perplexity']:.2f}")
            print(f"  Improved: Loss={result['improved_loss']:.4f}, Perplexity={result['improved_perplexity']:.2f}")
            print(f"  Loss Improvement: {result['loss_improvement']:.2f}%")
            print(f"  Perplexity Improvement: {result['perplexity_improvement']:.2f}%")
            
            successful_tests += 1
            loss_improvements.append(result['loss_improvement'])
            perplexity_improvements.append(result['perplexity_improvement'])
            total_loss_improvement += result['loss_improvement']
            total_perplexity_improvement += result['perplexity_improvement']
    
    if successful_tests > 0:
        avg_loss_improvement = total_loss_improvement / successful_tests
        avg_perplexity_improvement = total_perplexity_improvement / successful_tests
        positive_loss_improvements = sum(1 for imp in loss_improvements if imp < 0)  # Lower loss is better
        positive_perplexity_improvements = sum(1 for imp in perplexity_improvements if imp < 0)  # Lower perplexity is better
        
        print(f"\n📈 Overall Results:")
        print(f"  Successful tests: {successful_tests}")
        print(f"  Average loss improvement: {avg_loss_improvement:.2f}%")
        print(f"  Average perplexity improvement: {avg_perplexity_improvement:.2f}%")
        print(f"  Better loss: {positive_loss_improvements}/{successful_tests}")
        print(f"  Better perplexity: {positive_perplexity_improvements}/{successful_tests}")
        
        if avg_loss_improvement < 0 and avg_perplexity_improvement < 0:
            print(f"\n✅ SUCCESS! Geometric regularization improves generative performance!")
            print(f"✅ Average loss improvement: {avg_loss_improvement:.2f}%")
            print(f"✅ Average perplexity improvement: {avg_perplexity_improvement:.2f}%")
        else:
            print(f"\n❌ No improvement on generative tasks")
            print(f"❌ Average loss improvement: {avg_loss_improvement:.2f}%")
            print(f"❌ Average perplexity improvement: {avg_perplexity_improvement:.2f}%")
    else:
        print(f"\n❌ No successful tests completed")
    
    print("\n✅ Generative NLP experiment complete!")
    return results

if __name__ == "__main__":
    run_generative_nlp_experiment()
