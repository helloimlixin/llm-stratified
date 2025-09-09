#!/usr/bin/env python3
"""
Comprehensive Generation Benchmarks for Stratified Manifold Models
================================================================
This experiment evaluates generation capacity across multiple dimensions:
- Fidelity (perplexity, coherence)
- Diversity (n-gram diversity, semantic diversity)
- Controllability (instruction following)
- Long-range coherence (story completion)

Benchmarks implemented:
- Language Modeling (perplexity on validation sets)
- Story Generation (WritingPrompts-style)
- Dialogue Generation (multi-turn consistency)
- Instruction Following (simple commands)
- Code Generation (basic programming tasks)
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
    AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)
import torch.distributed as dist
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Define stratified components locally (same as before)
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

class GPTGenerativeWrapper(nn.Module):
    """GPT-2 wrapper optimized for generation tasks"""
    
    def __init__(self, model_name="gpt2", stratified_type="none"):
        super().__init__()
        self.model_name = model_name
        self.stratified_type = stratified_type
        
        # Load GPT-2 LMHead model and tokenizer (pretrained head for sensible perplexity)
        print(f"🤖 Loading {model_name} for generation benchmarks...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.base_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.hidden_size = self.base_model.config.hidden_size

        # Set pad token to eos to avoid resizing; ensure model uses it consistently
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model.config.pad_token_id = self.tokenizer.pad_token_id

        # Stratified component (post-attention modulation on hidden states)
        if stratified_type == "attention":
            num_heads = max(1, self.hidden_size // 64)
            self.stratified_component = StratifiedAttention(self.hidden_size, num_heads, num_strata=3)
        elif stratified_type == "none":
            self.stratified_component = nn.Identity()
        else:
            raise ValueError(f"Stratified type {stratified_type} not implemented")

        # Use pretrained LM head from GPT2LMHeadModel
        self.lm_head = self.base_model.lm_head
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass for language modeling"""
        # Ensure device consistency
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Get base transformer hidden states
        transformer_outputs = self.base_model.transformer(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        hidden_states = transformer_outputs.last_hidden_state
        
        # Apply stratified processing
        if self.stratified_type == "none":
            enhanced_hidden_states = hidden_states
        else:
            enhanced_hidden_states, _ = self.stratified_component(hidden_states)
        
        # Language modeling logits
        logits = self.lm_head(enhanced_hidden_states)
        
        # Compute loss if labels provided
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}
    
    def generate_text(self, prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.9):
        """Generate text continuation from prompt"""
        self.eval()
        with torch.no_grad():
            # Tokenize prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(next(self.parameters()).device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(next(self.parameters()).device)

            gen_ids = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            generated_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            # Remove prompt prefix if present
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
            return generated_text.strip()

class GenerationBenchmarks:
    """Comprehensive generation benchmarking suite"""
    
    def __init__(self, model):
        self.model = model
        self.tokenizer = model.tokenizer
        
    def evaluate_perplexity(self, texts):
        """Evaluate perplexity on text samples"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask")

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs["loss"]
                
                if torch.isfinite(loss):
                    total_loss += loss.item() * input_ids.size(1)
                    total_tokens += input_ids.size(1)
        
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            return perplexity
        return float('inf')
    
    def evaluate_diversity(self, generated_texts):
        """Evaluate n-gram diversity of generated texts"""
        def get_ngrams(text, n):
            tokens = text.split()
            return set([' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
        
        # Calculate distinct n-gram ratios
        all_unigrams = []
        all_bigrams = []
        all_trigrams = []
        
        for text in generated_texts:
            all_unigrams.extend(list(get_ngrams(text, 1)))
            all_bigrams.extend(list(get_ngrams(text, 2)))
            all_trigrams.extend(list(get_ngrams(text, 3)))
        
        distinct_1 = len(set(all_unigrams)) / len(all_unigrams) if all_unigrams else 0
        distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0
        distinct_3 = len(set(all_trigrams)) / len(all_trigrams) if all_trigrams else 0
        
        return {
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
            "distinct_3": distinct_3,
            "avg_length": np.mean([len(text.split()) for text in generated_texts])
        }
    
    def evaluate_story_generation(self, prompts):
        """Evaluate story generation quality"""
        generated_stories = []
        
        for prompt in prompts:
            story = self.model.generate_text(
                prompt, 
                max_length=150, 
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
            generated_stories.append(story)
        
        # Evaluate diversity and basic quality
        diversity_metrics = self.evaluate_diversity(generated_stories)
        
        # Simple coherence check (sentences should be reasonably connected)
        coherence_scores = []
        for story in generated_stories:
            sentences = story.split('.')
            # Basic coherence: check if story has multiple sentences and reasonable length
            coherence = min(1.0, len(sentences) / 3.0) * min(1.0, len(story.split()) / 50.0)
            coherence_scores.append(coherence)
        
        return {
            "generated_stories": generated_stories,
            "diversity": diversity_metrics,
            "avg_coherence": np.mean(coherence_scores),
            "avg_story_length": np.mean([len(story.split()) for story in generated_stories])
        }
    
    def evaluate_dialogue_generation(self, dialogue_contexts):
        """Evaluate dialogue generation"""
        generated_responses = []
        
        for context in dialogue_contexts:
            response = self.model.generate_text(
                context + "\nResponse:",
                max_length=50,
                temperature=0.7,
                top_k=40,
                top_p=0.9
            )
            generated_responses.append(response)
        
        # Evaluate response quality
        diversity_metrics = self.evaluate_diversity(generated_responses)
        
        # Check if responses are contextually appropriate (basic length and content checks)
        appropriateness_scores = []
        for response in generated_responses:
            # Basic appropriateness: not too short, not too long, contains words
            length_score = min(1.0, len(response.split()) / 10.0) * (1.0 if len(response.split()) < 30 else 0.5)
            appropriateness_scores.append(length_score)
        
        return {
            "generated_responses": generated_responses,
            "diversity": diversity_metrics,
            "avg_appropriateness": np.mean(appropriateness_scores),
            "avg_response_length": np.mean([len(resp.split()) for resp in generated_responses])
        }
    
    def evaluate_instruction_following(self, instructions):
        """Evaluate basic instruction following"""
        generated_outputs = []
        
        for instruction in instructions:
            output = self.model.generate_text(
                f"Instruction: {instruction}\nResponse:",
                max_length=80,
                temperature=0.5,  # Lower temperature for more focused responses
                top_k=30,
                top_p=0.8
            )
            generated_outputs.append(output)
        
        # Basic instruction following evaluation
        following_scores = []
        for instruction, output in zip(instructions, generated_outputs):
            # Simple heuristic: check if output contains relevant keywords from instruction
            instruction_words = set(instruction.lower().split())
            output_words = set(output.lower().split())
            overlap = len(instruction_words.intersection(output_words)) / len(instruction_words)
            following_scores.append(overlap)
        
        return {
            "generated_outputs": generated_outputs,
            "avg_instruction_following": np.mean(following_scores),
            "avg_output_length": np.mean([len(out.split()) for out in generated_outputs])
        }

def create_benchmark_datasets():
    """Create datasets for different generation benchmarks"""
    
    # Language modeling validation texts
    lm_texts = [
        "The quick brown fox jumps over the lazy dog and continues running through the forest.",
        "In a world where technology advances rapidly, humans must adapt to new challenges.",
        "The ancient castle stood majestically on the hill, overlooking the peaceful valley below.",
        "Scientists have discovered a new species of butterfly in the Amazon rainforest.",
        "The chef prepared an exquisite meal using fresh ingredients from the local market."
    ]
    
    # Story generation prompts
    story_prompts = [
        "Once upon a time, in a small village nestled between mountains,",
        "The detective examined the mysterious letter that had arrived that morning,",
        "As the spaceship approached the unknown planet,",
        "The old lighthouse keeper had seen many storms, but this one was different,",
        "In the year 2050, robots had become commonplace, but one robot was special,"
    ]
    
    # Dialogue contexts
    dialogue_contexts = [
        "Person A: What do you think about the weather today?\nPerson B: It's quite nice, but a bit cloudy.\nPerson A:",
        "Customer: I'm looking for a good restaurant nearby.\nAssistant: There are several great options. What type of cuisine do you prefer?\nCustomer:",
        "Student: I'm having trouble understanding this math problem.\nTeacher: Let me help you break it down step by step.\nStudent:",
        "Friend 1: Did you see the movie last night?\nFriend 2: Yes, it was amazing! The special effects were incredible.\nFriend 1:",
        "Doctor: Your test results look good overall.\nPatient: That's great to hear! Are there any areas I should focus on?\nDoctor:"
    ]
    
    # Instruction following tasks
    instructions = [
        "Write a short poem about nature.",
        "Explain how to make a paper airplane.",
        "List three benefits of regular exercise.",
        "Describe the process of photosynthesis in simple terms.",
        "Give advice on how to study effectively for exams."
    ]
    
    return {
        "lm_texts": lm_texts,
        "story_prompts": story_prompts,
        "dialogue_contexts": dialogue_contexts,
        "instructions": instructions
    }

def _is_rank_zero():
    try:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    # Fallback to env vars used by accelerate
    return os.environ.get("RANK", "0") in ("0", 0)


def run_generation_benchmarks_experiment():
    """Run comprehensive generation benchmarks experiment"""
    print("🚀 Comprehensive Generation Benchmarks Experiment")
    print("=" * 80)
    print("Evaluating generation capacity across multiple dimensions:")
    print("- Language Modeling (Perplexity)")
    print("- Story Generation (Coherence & Diversity)")
    print("- Dialogue Generation (Appropriateness)")
    print("- Instruction Following (Task Completion)")
    print("=" * 80)
    
    # Create benchmark datasets
    datasets = create_benchmark_datasets()
    
    # Test stratified types
    stratified_types = ["none", "attention"]
    results = {}
    
    for stratified_type in stratified_types:
        print(f"\n🔍 Benchmarking {stratified_type} stratified type...")
        print("-" * 50)
        
        try:
            # Create model
            model = GPTGenerativeWrapper(model_name="gpt2", stratified_type=stratified_type)
            benchmarks = GenerationBenchmarks(model)
            
            print(f"🤖 Running benchmarks for GPT-2 + {stratified_type}...")
            
            # 1. Language Modeling Perplexity
            print("📊 Evaluating perplexity...")
            perplexity = benchmarks.evaluate_perplexity(datasets["lm_texts"])
            
            # 2. Story Generation
            print("📖 Evaluating story generation...")
            story_results = benchmarks.evaluate_story_generation(datasets["story_prompts"])
            
            # 3. Dialogue Generation
            print("💬 Evaluating dialogue generation...")
            dialogue_results = benchmarks.evaluate_dialogue_generation(datasets["dialogue_contexts"])
            
            # 4. Instruction Following
            print("📝 Evaluating instruction following...")
            instruction_results = benchmarks.evaluate_instruction_following(datasets["instructions"])
            
            # Store comprehensive results
            results[stratified_type] = {
                "perplexity": perplexity,
                "story_generation": {
                    "diversity": story_results["diversity"],
                    "avg_coherence": story_results["avg_coherence"],
                    "avg_length": story_results["avg_story_length"],
                    "sample_stories": story_results["generated_stories"][:2]  # Save first 2 as examples
                },
                "dialogue_generation": {
                    "diversity": dialogue_results["diversity"],
                    "avg_appropriateness": dialogue_results["avg_appropriateness"],
                    "avg_length": dialogue_results["avg_response_length"],
                    "sample_responses": dialogue_results["generated_responses"][:2]
                },
                "instruction_following": {
                    "avg_following_score": instruction_results["avg_instruction_following"],
                    "avg_length": instruction_results["avg_output_length"],
                    "sample_outputs": instruction_results["generated_outputs"][:2]
                },
                "status": "success"
            }
            
            print(f"✅ {stratified_type}: Perplexity = {perplexity:.2f}")
            print(f"   📖 Story Coherence = {story_results['avg_coherence']:.3f}")
            print(f"   💬 Dialogue Appropriateness = {dialogue_results['avg_appropriateness']:.3f}")
            print(f"   📝 Instruction Following = {instruction_results['avg_instruction_following']:.3f}")
            
        except Exception as e:
            print(f"❌ Error with {stratified_type}: {e}")
            results[stratified_type] = {
                "error": str(e),
                "status": "error"
            }
    
    # Save results from rank 0 only
    if _is_rank_zero():
        os.makedirs("./results/generation_benchmarks", exist_ok=True)
        with open("./results/generation_benchmarks/results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Generate comprehensive summary
    print("\n" + "=" * 80)
    print("📊 GENERATION BENCHMARKS SUMMARY")
    print("=" * 80)
    
    if _is_rank_zero():
        for stratified_type, result in results.items():
            if result["status"] == "success":
                print(f"🤖 GPT-2 + {stratified_type}:")
                print(f"   🔤 Perplexity: {result['perplexity']:.2f}")
                print(f"   📖 Story Generation:")
                print(f"      - Coherence: {result['story_generation']['avg_coherence']:.3f}")
                print(f"      - Diversity (distinct-1): {result['story_generation']['diversity']['distinct_1']:.3f}")
                print(f"      - Avg Length: {result['story_generation']['avg_length']:.1f} words")
                print(f"   💬 Dialogue Generation:")
                print(f"      - Appropriateness: {result['dialogue_generation']['avg_appropriateness']:.3f}")
                print(f"      - Diversity (distinct-1): {result['dialogue_generation']['diversity']['distinct_1']:.3f}")
                print(f"   📝 Instruction Following:")
                print(f"      - Following Score: {result['instruction_following']['avg_following_score']:.3f}")
                print(f"      - Avg Length: {result['instruction_following']['avg_length']:.1f} words")
            else:
                print(f"❌ GPT-2 + {stratified_type}: ERROR - {result['error']}")
            print()
        
        print(f"✅ Detailed results saved to: ./results/generation_benchmarks/results.json")
    return results

if __name__ == "__main__":
    run_generation_benchmarks_experiment()
