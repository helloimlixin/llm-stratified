#!/usr/bin/env python3
"""
Fiber Bundle Hypothesis Test Framework - Main Entry Point

This is the unified interface for all fiber bundle analyses on LLM embeddings.
Supports basic analysis, multi-domain analysis, LLaMA models, and model comparisons.
"""

import sys
import os
import argparse
import logging
import time
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List

# Aggressive warning suppression at the very start
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Setup clean environment
from fiber_bundle_test.utils.warning_suppression import setup_clean_environment
setup_clean_environment()

# Additional transformers-specific suppression
try:
    import transformers
    transformers.logging.set_verbosity_error()
    
    # Suppress specific transformers warnings
    import transformers.modeling_utils
    transformers.modeling_utils.logger.setLevel(logging.ERROR)
    
    # Suppress parameter renaming warnings at the source
    import transformers.models.bert.modeling_bert
    import transformers.models.roberta.modeling_roberta
    
    # Set all transformers model loggers to ERROR level
    for module_name in dir(transformers.models):
        try:
            module = getattr(transformers.models, module_name)
            if hasattr(module, 'logger'):
                module.logger.setLevel(logging.ERROR)
        except:
            pass
            
except:
    pass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Import framework components
from fiber_bundle_test import (
    FiberBundleTest, BERTEmbeddingExtractor, RoBERTaEmbeddingExtractor, 
    ModernLLMExtractor, ResultsVisualizer, load_multidomain_sentiment,
    ProcessingConfig, ScalableEmbeddingProcessor, MixtureOfDictionaryExperts,
    ContrastiveTrainer, TrainingConfig, StratificationAnalyzer,
    ClusteringAnalyzer, DimensionalityAnalyzer
)
from fiber_bundle_test.embeddings.bert_embeddings import create_sample_dataset
from fiber_bundle_test.utils import DataUtils
from fiber_bundle_test.visualization.advanced_visualizations import create_notebook_visualizations

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fiber Bundle Hypothesis Test Framework',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Analysis type (required)
    parser.add_argument('analysis', 
                       choices=['basic', 'multi-domain', 'llama', 'comparison', 'advanced', 'notebook'],
                       help='Type of analysis to run')
    
    # Model options
    parser.add_argument('--model', type=str, default='auto',
                       help='Model to use (auto selects based on analysis type)')
    parser.add_argument('--models', nargs='+', 
                       default=['bert-base', 'roberta-base'],
                       help='Models for comparison analysis')
    
    # Data options
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of samples per domain/token')
    parser.add_argument('--domains', nargs='+',
                       default=['imdb', 'amazon', 'rotten', 'sst2'],
                       help='Domains for multi-domain analysis')
    
    # Processing options
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs for MoE models')
    
    # Test parameters
    parser.add_argument('--r-min', type=float, default=0.01,
                       help='Minimum radius value')
    parser.add_argument('--r-max', type=float, default=20.0,
                       help='Maximum radius value')
    parser.add_argument('--alpha', type=float, default=0.001,
                       help='Significance level (conservative for reliable results)')
    parser.add_argument('--window-size', type=int, default=20,
                       help='Window size for slope detection (larger = more stable)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory')
    parser.add_argument('--save-embeddings', action='store_true',
                       help='Save extracted embeddings')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save visualization plots')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip MoE training for faster execution')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> str:
    """Setup and return the appropriate device."""
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg
    
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = 'cpu'
    
    return device


def print_header(title: str, device: str, quiet: bool = False):
    """Print analysis header."""
    if not quiet:
        print(f"\n🚀 {title}")
        print("=" * (len(title) + 4))
        print(f"Device: {device}")


def run_basic_analysis(args, device: str) -> Dict[str, Any]:
    """Run basic fiber bundle analysis with BERT."""
    print_header("Basic Fiber Bundle Analysis", device, args.quiet)
    
    # Load sample dataset
    sentences, tokens = create_sample_dataset()
    if not args.quiet:
        print(f"📊 Loaded {len(sentences)} sentences with target tokens")
    
    # Extract BERT embeddings
    model_name = args.model if args.model != 'auto' else 'bert-base-uncased'
    extractor = BERTEmbeddingExtractor(model_name)
    embeddings = extractor.get_embeddings(sentences, tokens)
    
    if not args.quiet:
        print(f"🔄 Extracted embeddings: {embeddings.shape}")
        DataUtils.print_summary_statistics(embeddings)
    
    # Run fiber bundle test with conservative parameters
    test = FiberBundleTest(
        r_min=args.r_min,
        r_max=args.r_max,
        n_r=150,
        alpha=args.alpha,  # Now defaults to 0.01 (more conservative)
        window_size=args.window_size  # Now defaults to 15
    )
    
    results = test.run_test(embeddings, verbose=not args.quiet)
    
    # Print results
    print(f"\n📊 Results:")
    print(f"  Rejection rate: {results['rejection_rate']:.1%}")
    print(f"  Total rejections: {results['total_rejections']}/{results['total_tokens']}")
    
    # Token-level summary
    if not args.quiet:
        token_labels = DataUtils.extract_token_labels(sentences, tokens)
        token_results = {}
        
        for (token_idx, decision), label, dimensions in zip(
            results['results'], token_labels, results['dimensions']
        ):
            token = label.split(' ')[0]
            if token not in token_results:
                token_results[token] = {'reject': 0, 'total': 0, 'base_dims': [], 'fiber_dims': []}
            
            token_results[token]['total'] += 1
            if decision == 'Reject':
                token_results[token]['reject'] += 1
                if dimensions[0] is not None and dimensions[1] is not None:
                    token_results[token]['base_dims'].append(dimensions[0])
                    token_results[token]['fiber_dims'].append(dimensions[1])
        
        print(f"\n📋 Results by Token:")
        for token, stats in token_results.items():
            rejection_rate = stats['reject'] / stats['total']
            avg_base = sum(stats['base_dims']) / len(stats['base_dims']) if stats['base_dims'] else 0
            avg_fiber = sum(stats['fiber_dims']) / len(stats['fiber_dims']) if stats['fiber_dims'] else 0
            print(f"  {token:<8} {stats['reject']:2d}/{stats['total']:2d} ({rejection_rate:.0%}) "
                  f"Base: {avg_base:.1f}, Fiber: {avg_fiber:.1f}")
    
    return results


def run_multidomain_analysis(args, device: str) -> Dict[str, Any]:
    """Run multi-domain analysis with RoBERTa."""
    print_header("Multi-Domain Analysis", device, args.quiet)
    
    # Load multi-domain dataset
    dataset = load_multidomain_sentiment(samples_per_domain=args.samples)
    texts = dataset["text"]
    domains = dataset["domain"]
    labels = torch.tensor(dataset["label"], dtype=torch.long)
    
    if not args.quiet:
        print(f"📊 Loaded {len(texts)} samples from {len(set(domains))} domains")
        domain_counts = {domain: domains.count(domain) for domain in set(domains)}
        for domain, count in domain_counts.items():
            print(f"  {domain}: {count} samples")
    
    # Extract RoBERTa embeddings
    model_name = args.model if args.model != 'auto' else 'roberta-base'
    extractor = RoBERTaEmbeddingExtractor(model_name, device)
    embeddings = extractor.embed_texts(texts, batch_size=args.batch_size)
    
    if not args.quiet:
        print(f"🔄 Extracted embeddings: {embeddings.shape}")
    
    # Run comprehensive stratification analysis
    if not args.quiet:
        print(f"🔬 Running comprehensive stratification analysis...")
        print(f"   This may take a few minutes for {len(embeddings)} samples...")
    
    analyzer = StratificationAnalyzer()
    results = analyzer.analyze_stratification(embeddings, domains, labels.numpy())
    
    # Print results
    print(f"\n📊 Results:")
    print(f"  Fiber bundle rejection rate: {results['fiber_bundle']['rejection_rate']:.1%}")
    print(f"  Overall intrinsic dimension: {results['dimensionality']['overall_intrinsic_dimension']}")
    print(f"  Clustering quality: {results['clustering']['metrics']['silhouette_score']:.3f}")
    
    if not args.quiet and 'stratum_dimensions' in results['dimensionality']:
        print(f"  Stratum dimensions: {list(results['dimensionality']['stratum_dimensions'].values())}")
    
    return results


def run_llama_analysis(args, device: str) -> Dict[str, Any]:
    """Run LLaMA-specific analysis."""
    print_header("LLaMA Analysis", device, args.quiet)
    
    # Load dataset
    dataset = load_multidomain_sentiment(samples_per_domain=args.samples)
    texts = dataset["text"]
    domains = dataset["domain"]
    
    if not args.quiet:
        print(f"📊 Loaded {len(texts)} samples from {len(set(domains))} domains")
    
    # Extract LLaMA embeddings
    model_name = args.model if args.model != 'auto' else 'llama-1b'
    
    try:
        extractor = ModernLLMExtractor.create_extractor(
            model_name, 
            device=device,
            batch_size=args.batch_size
        )
        embeddings = extractor.get_embeddings(texts)
        
        if not args.quiet:
            print(f"🔄 Extracted embeddings: {embeddings.shape}")
        
    except Exception as e:
        error_msg = str(e)
        if "access" in error_msg.lower() or "gated" in error_msg.lower():
            print(f"⚠️ {model_name} requires access approval")
            print("1. Visit: https://huggingface.co/meta-llama/Llama-3.2-1B")
            print("2. Request access from Meta")
            print("3. Login: huggingface-cli login")
            print("4. Try again once approved")
        elif "not a string" in error_msg or "tokenizer" in error_msg.lower():
            print(f"⚠️ {model_name} has compatibility issues")
            print("Falling back to BERT analysis...")
            # Fallback to BERT analysis
            return run_basic_analysis(args, device)
        else:
            print(f"❌ Failed to load {model_name}: {e}")
            print("Troubleshooting:")
            print("• Check model name spelling")
            print("• Verify internet connection") 
            print("• Try with --device cpu")
        return None
    
    # Run fiber bundle test with LLaMA-optimized parameters
    # LLaMA embeddings have very different scale and require ultra-conservative parameters
    test = FiberBundleTest(
        r_min=0.5,      # Larger minimum radius for LLaMA scale
        r_max=100.0,    # Much larger maximum radius
        n_r=200,
        alpha=0.0001,   # Ultra-conservative significance level
        window_size=40  # Large window for stable detection
    )
    
    results = test.run_test(embeddings, verbose=False)
    
    # Print results
    print(f"\n📊 Results:")
    print(f"  Model: {model_name}")
    print(f"  Rejection rate: {results['rejection_rate']:.1%}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Total rejections: {results['total_rejections']}/{results['total_tokens']}")
    
    # Additional LLaMA-specific analysis
    if results['rejection_rate'] < 0.1:  # Only show if very low (< 10%)
        print(f"  ℹ️ Low rejection rate may indicate:")
        print(f"     • LLaMA embeddings have different geometric properties")
        print(f"     • May need different test parameters for LLaMA models")
        print(f"     • Consider using advanced analysis for more insights")
    elif results['rejection_rate'] > 0.9:  # High rejection rate
        print(f"  ✅ High rejection rate indicates strong fiber bundle violations")
        print(f"  📊 LLaMA embeddings show clear stratified manifold structure")
    
    # Domain-level analysis
    if not args.quiet:
        domain_results = {}
        for (idx, decision), domain in zip(results['results'], domains):
            if domain not in domain_results:
                domain_results[domain] = {'reject': 0, 'total': 0}
            domain_results[domain]['total'] += 1
            if decision == 'Reject':
                domain_results[domain]['reject'] += 1
        
        print(f"\n📋 Results by Domain:")
        for domain, stats in domain_results.items():
            rate = stats['reject'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {domain:<12} {stats['reject']:3d}/{stats['total']:3d} ({rate:.1%})")
    
    return results


def run_comparison_analysis(args, device: str) -> Dict[str, Any]:
    """Run model comparison analysis."""
    print_header("Model Comparison Analysis", device, args.quiet)
    
    # Load dataset
    sentences, tokens = create_sample_dataset()
    if not args.quiet:
        print(f"📊 Loaded {len(sentences)} sentences for comparison")
    
    results = {}
    
    # Test each model
    for model in args.models:
        try:
            if not args.quiet:
                print(f"\n🔄 Analyzing {model}...")
            
            # Create appropriate extractor
            if model.startswith('bert'):
                model_full = model + '-uncased' if 'uncased' not in model else model
                extractor = BERTEmbeddingExtractor(model_full)
                embeddings = extractor.get_embeddings(sentences, tokens)
            else:
                extractor = ModernLLMExtractor.create_extractor(model, device=device, batch_size=args.batch_size)
                embeddings = extractor.get_embeddings(sentences, tokens)
            
            # Run test with model-specific parameters
            if 'llama' in model.lower():
                # LLaMA-optimized parameters (ultra-conservative)
                test = FiberBundleTest(
                    r_min=0.5, r_max=100.0, n_r=200, alpha=0.0001, window_size=40
                )
            elif 'gpt' in model.lower():
                # GPT-optimized parameters (larger radius range needed)
                test = FiberBundleTest(
                    r_min=1.0, r_max=120.0, n_r=150, alpha=0.001, window_size=20
                )
            else:
                # Conservative parameters for BERT/RoBERTa
                test = FiberBundleTest(
                    r_min=args.r_min,
                    r_max=args.r_max,
                    n_r=150,
                    alpha=args.alpha,  # Now 0.001 by default
                    window_size=args.window_size  # Now 20 by default
                )
            
            result = test.run_test(embeddings, verbose=False)
            results[model] = result
            
            if not args.quiet:
                print(f"  ✅ {model}: {result['rejection_rate']:.1%} rejection rate")
            
        except Exception as e:
            error_msg = str(e)
            if "access" in error_msg.lower() or "login" in error_msg.lower() or "token" in error_msg.lower():
                if not args.quiet:
                    print(f"  ⚠️ {model}: Access required - try 'huggingface-cli login'")
            elif "not a string" in error_msg or "tokenizer" in error_msg.lower():
                if not args.quiet:
                    print(f"  ⚠️ {model}: Model compatibility issue - skipping")
            else:
                if not args.quiet:
                    print(f"  ❌ {model}: Failed - {e}")
            logger.warning(f"Failed to analyze {model}: {e}")
    
    # Print comparison
    print(f"\n📊 Model Comparison:")
    print("-" * 50)
    print(f"{'Model':<20} {'Rejection Rate':<15} {'Rejections'}")
    print("-" * 50)
    
    for model, result in results.items():
        print(f"{model:<20} {result['rejection_rate']:>6.1%} {result['total_rejections']:>12d}/{result['total_tokens']}")
    
    return results


def run_advanced_analysis(args, device: str) -> Dict[str, Any]:
    """Run advanced analysis with LLaMA and MoE training."""
    print_header("Advanced LLaMA Analysis with MoE", device, args.quiet)
    
    # Load multi-domain dataset
    dataset = load_multidomain_sentiment(samples_per_domain=args.samples)
    texts = dataset["text"]
    domains = dataset["domain"] 
    labels = torch.tensor(dataset["label"], dtype=torch.long)
    
    if not args.quiet:
        print(f"📊 Loaded {len(texts)} samples from {len(set(domains))} domains")
    
    # Extract LLaMA embeddings
    model_name = args.model if args.model != 'auto' else 'llama-1b'
    
    try:
        extractor = ModernLLMExtractor.create_extractor(
            model_name, 
            device=device,
            batch_size=args.batch_size
        )
        all_embeddings = extractor.get_embeddings(texts)
        
        if not args.quiet:
            print(f"🔄 Extracted embeddings: {all_embeddings.shape}")
        
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        print("Make sure you have access to LLaMA models and are logged in to HuggingFace")
        return None
    
    # Dimensionality reduction
    dim_analyzer = DimensionalityAnalyzer()
    emb_64d, reduction_info = dim_analyzer.reduce_dimensionality(all_embeddings, target_dim=64)
    
    if not args.quiet:
        print(f"🔧 Reduced to {emb_64d.shape[1]}D (variance: {reduction_info.get('total_explained_variance', 0):.1%})")
    
    # Fiber bundle analysis
    test = FiberBundleTest(
        r_min=args.r_min,
        r_max=args.r_max,
        alpha=args.alpha,
        window_size=args.window_size
    )
    
    fiber_results = test.run_test(emb_64d, verbose=False)
    
    print(f"\n📊 Fiber Bundle Results:")
    print(f"  Rejection rate: {fiber_results['rejection_rate']:.1%}")
    print(f"  Total rejections: {fiber_results['total_rejections']}/{fiber_results['total_tokens']}")
    
    # Clustering analysis
    clustering_analyzer = ClusteringAnalyzer()
    clustering_results = clustering_analyzer.perform_clustering(emb_64d, n_clusters=5)
    strata = clustering_results['labels']
    
    print(f"  Clustering quality: {clustering_results['metrics']['silhouette_score']:.3f}")
    
    # MoE training (if not skipped)
    trained_model = None
    if not args.skip_training:
        if not args.quiet:
            print(f"\n🧠 Training Mixture-of-Experts...")
        
        X_tensor = torch.tensor(emb_64d, dtype=torch.float32)
        dataset_tensor = TensorDataset(X_tensor, labels)
        loader = DataLoader(dataset_tensor, batch_size=8, shuffle=True)
        
        model = MixtureOfDictionaryExperts(
            input_dim=64, query_dim=128, code_dim=32, K=7,
            projection_dim=64, num_lista_layers=5,
            sparsity_levels=[8, 12, 16, 20, 24, 28, 32],
            threshold=0.5
        )
        
        config = TrainingConfig(
            num_epochs=args.epochs,
            learning_rate=1e-3,
            log_interval=max(1, args.epochs // 5)
        )
        
        trainer = ContrastiveTrainer(model, config)
        training_history = trainer.train(loader)
        trained_model = model
        
        print(f"✅ MoE training completed (final loss: {training_history['train_loss'][-1]:.6f})")
    
    # Combine results
    combined_results = {
        'fiber_bundle': fiber_results,
        'clustering': clustering_results,
        'model_name': model_name,
        'embedding_shape': all_embeddings.shape,
        'reduced_shape': emb_64d.shape,
        'domains': list(set(domains)),
        'moe_trained': trained_model is not None
    }
    
    return combined_results


def run_notebook_workflow(args, device: str) -> Dict[str, Any]:
    """Run complete notebook workflow analysis."""
    print_header("Complete Notebook Workflow", device, args.quiet)
    
    # Load multi-domain dataset
    dataset = load_multidomain_sentiment(samples_per_domain=args.samples)
    texts = dataset["text"]
    domains = dataset["domain"]
    labels = torch.tensor(dataset["label"], dtype=torch.long)
    
    if not args.quiet:
        print(f"📊 Loaded {len(texts)} samples from {len(set(domains))} domains")
    
    # Extract RoBERTa embeddings
    model_name = args.model if args.model != 'auto' else 'roberta-base'
    extractor = RoBERTaEmbeddingExtractor(model_name, device)
    all_embeddings = extractor.embed_texts(texts, batch_size=args.batch_size)
    
    if not args.quiet:
        print(f"🔄 Extracted embeddings: {all_embeddings.shape}")
    
    # Dimensionality reduction
    dim_analyzer = DimensionalityAnalyzer()
    emb_64d, reduction_info = dim_analyzer.reduce_dimensionality(all_embeddings, target_dim=64)
    
    if not args.quiet:
        print(f"🔧 Reduced to {emb_64d.shape[1]}D (variance: {reduction_info.get('total_explained_variance', 0):.1%})")
    
    # Comprehensive stratification analysis
    analyzer = StratificationAnalyzer()
    results = analyzer.analyze_stratification(emb_64d, domains, labels.numpy())
    
    print(f"\n📊 Comprehensive Results:")
    print(f"  Fiber bundle rejection rate: {results['fiber_bundle']['rejection_rate']:.1%}")
    print(f"  Overall intrinsic dimension: {results['dimensionality']['overall_intrinsic_dimension']}")
    print(f"  Clustering quality: {results['clustering']['metrics']['silhouette_score']:.3f}")
    
    if not args.quiet and 'stratum_dimensions' in results['dimensionality']:
        stratum_dims = results['dimensionality']['stratum_dimensions']
        print(f"  Stratum dimensions: {list(stratum_dims.values())}")
    
    return results


def save_results(results: Dict[str, Any], args, analysis_type: str):
    """Save analysis results."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main results
    DataUtils.save_results(results, output_dir / f'{analysis_type}_results.json')
    
    # Save embeddings if requested
    if args.save_embeddings and 'embeddings' in locals():
        DataUtils.save_embeddings(
            locals()['embeddings'],
            output_dir / f'{analysis_type}_embeddings.npz',
            {'analysis_type': analysis_type, 'model': args.model}
        )
    
    # Create summary
    summary = {
        'analysis_type': analysis_type,
        'model': args.model,
        'timestamp': time.time(),
        'parameters': vars(args)
    }
    
    # Add type-specific summary info
    if analysis_type in ['basic', 'comparison']:
        if isinstance(results, dict) and 'rejection_rate' in results:
            summary['rejection_rate'] = results['rejection_rate']
        elif isinstance(results, dict):
            # Multi-model results
            summary['model_results'] = {
                model: result.get('rejection_rate', 0) 
                for model, result in results.items()
            }
    elif analysis_type in ['multi-domain', 'notebook']:
        summary['rejection_rate'] = results.get('fiber_bundle', {}).get('rejection_rate', 0)
        summary['clustering_quality'] = results.get('clustering', {}).get('metrics', {}).get('silhouette_score', 0)
    elif analysis_type in ['llama', 'advanced']:
        if 'fiber_bundle' in results:
            summary['rejection_rate'] = results['fiber_bundle']['rejection_rate']
        else:
            summary['rejection_rate'] = results.get('rejection_rate', 0)
    
    # Use DataUtils for proper JSON serialization
    DataUtils.save_results(summary, output_dir / f'{analysis_type}_summary.json')
    
    if not args.quiet:
        print(f"\n💾 Results saved to: {output_dir}")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup device
    device = setup_device(args.device)
    
    if not args.quiet:
        print(f"🚀 Fiber Bundle Hypothesis Test Framework")
        print(f"Analysis: {args.analysis}")
        print(f"Device: {device}")
    
    # Run selected analysis
    try:
        if args.analysis == 'basic':
            results = run_basic_analysis(args, device)
            
        elif args.analysis == 'multi-domain':
            results = run_multidomain_analysis(args, device)
            
        elif args.analysis == 'llama':
            results = run_llama_analysis(args, device)
            if results is None:
                return
                
        elif args.analysis == 'comparison':
            results = run_comparison_analysis(args, device)
            
        elif args.analysis == 'advanced':
            results = run_advanced_analysis(args, device)
            if results is None:
                return
                
        elif args.analysis == 'notebook':
            results = run_notebook_workflow(args, device)
        
        # Save results
        save_results(results, args, args.analysis)
        
        if not args.quiet:
            print(f"\n🎉 {args.analysis.title()} analysis completed successfully!")
    
    except KeyboardInterrupt:
        print(f"\n⏹️ Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\n❌ Analysis failed: {e}")
        print("\nTroubleshooting:")
        print("• Install dependencies: pip install -r requirements.txt")
        print("• Check model access (for LLaMA): huggingface-cli login")
        print("• Try reducing --samples or --batch-size for memory issues")
        print("• Use --device cpu if GPU issues persist")


if __name__ == '__main__':
    main()
