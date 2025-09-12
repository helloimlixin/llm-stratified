#!/usr/bin/env python3
"""
Optimized Analysis for Large Multi-Domain Datasets

This script is specifically optimized for running robust analysis on large datasets
with comprehensive progress tracking and performance optimizations.
"""

import sys
import time
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Setup clean environment
from fiber_bundle_test.utils.warning_suppression import setup_clean_environment
setup_clean_environment()

import numpy as np
import torch
from tqdm import tqdm

from fiber_bundle_test import (
    RoBERTaEmbeddingExtractor, ModernLLMExtractor, FiberBundleTest,
    load_multidomain_sentiment, StratificationAnalyzer
)
from fiber_bundle_test.utils import DataUtils


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Large Dataset Fiber Bundle Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model', type=str, default='roberta-base',
                       choices=['roberta-base', 'roberta-large', 'llama-1b', 'gpt2'],
                       help='Model to analyze')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Samples per domain (total = samples × 6 domains)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for embedding extraction')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use')
    parser.add_argument('--output-dir', type=str, default='./large_dataset_output',
                       help='Output directory')
    parser.add_argument('--save-embeddings', action='store_true',
                       help='Save extracted embeddings')
    parser.add_argument('--fast-mode', action='store_true',
                       help='Use faster but less comprehensive analysis')
    
    return parser.parse_args()


def estimate_time_and_memory(n_samples: int, embedding_dim: int, model_name: str):
    """Estimate processing time and memory requirements."""
    print(f"\n⏱️ Estimated Requirements:")
    
    # Time estimates (rough)
    embedding_time = n_samples * 0.01  # ~10ms per sample
    analysis_time = n_samples * 0.1    # ~100ms per sample for analysis
    total_time = embedding_time + analysis_time
    
    # Memory estimates
    embedding_memory = n_samples * embedding_dim * 4 / (1024**3)  # GB for float32
    analysis_memory = embedding_memory * 2  # Rough estimate for analysis overhead
    
    print(f"   Estimated time: {total_time/60:.1f} minutes")
    print(f"   Estimated memory: {analysis_memory:.1f} GB")
    print(f"   Model: {model_name}")
    
    if total_time > 1800:  # 30 minutes
        print(f"   ⚠️ Long analysis - consider using --fast-mode or reducing --samples")
    
    if analysis_memory > 8:
        print(f"   ⚠️ High memory usage - consider reducing batch size or samples")


def run_large_dataset_analysis(args):
    """Run analysis optimized for large datasets."""
    print(f"🔬 Large Dataset Fiber Bundle Analysis")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Samples per domain: {args.samples}")
    print(f"Total samples: {args.samples * 6} (6 domains)")
    
    # Load large dataset with progress
    print(f"\n📊 Loading large multi-domain dataset...")
    start_time = time.time()
    
    dataset = load_multidomain_sentiment(samples_per_domain=args.samples)
    texts = dataset["text"]
    domains = dataset["domain"]
    labels = torch.tensor(dataset["label"], dtype=torch.long)
    
    load_time = time.time() - start_time
    print(f"✅ Dataset loaded in {load_time:.1f}s: {len(texts)} samples from {len(set(domains))} domains")
    
    # Estimate requirements
    embedding_dim = 2048 if 'llama' in args.model else 768
    estimate_time_and_memory(len(texts), embedding_dim, args.model)
    
    # Extract embeddings with progress
    print(f"\n🔄 Extracting {args.model} embeddings...")
    start_time = time.time()
    
    if args.model.startswith('roberta'):
        extractor = RoBERTaEmbeddingExtractor(args.model, device)
        embeddings = extractor.embed_texts(texts, batch_size=args.batch_size)
    else:
        extractor = ModernLLMExtractor.create_extractor(args.model, device=device, batch_size=args.batch_size)
        embeddings = extractor.get_embeddings(texts)
    
    embedding_time = time.time() - start_time
    print(f"✅ Embeddings extracted in {embedding_time:.1f}s: {embeddings.shape}")
    
    # Print embedding statistics
    print(f"\n📈 Embedding Statistics:")
    DataUtils.print_summary_statistics(embeddings)
    
    # Run analysis based on mode
    if args.fast_mode:
        print(f"\n🚀 Running fast fiber bundle analysis...")
        start_time = time.time()
        
        # Fast mode: just fiber bundle test with optimized parameters
        if 'gpt' in args.model:
            test = FiberBundleTest(r_min=1.0, r_max=120.0, n_r=100, alpha=0.001, window_size=20)
        elif 'llama' in args.model:
            test = FiberBundleTest(r_min=0.1, r_max=80.0, n_r=100, alpha=0.05, window_size=15)
        else:
            test = FiberBundleTest(r_min=0.01, r_max=20.0, n_r=100, alpha=0.001, window_size=20)
        
        results = test.run_test(embeddings, verbose=False)
        analysis_time = time.time() - start_time
        
        print(f"✅ Fast analysis completed in {analysis_time:.1f}s")
        
    else:
        print(f"\n🔬 Running comprehensive stratification analysis...")
        start_time = time.time()
        
        analyzer = StratificationAnalyzer()
        results = analyzer.analyze_stratification(embeddings, domains, labels.numpy())
        analysis_time = time.time() - start_time
        
        print(f"✅ Comprehensive analysis completed in {analysis_time:.1f}s")
    
    # Print results
    print(f"\n📊 Analysis Results:")
    print("=" * 30)
    
    if args.fast_mode:
        print(f"Model: {args.model}")
        print(f"Total samples: {len(embeddings)}")
        print(f"Rejection rate: {results['rejection_rate']:.1%}")
        print(f"Total rejections: {results['total_rejections']}/{results['total_tokens']}")
        
        # Domain-level breakdown
        domain_results = {}
        for (idx, decision), domain in zip(results['results'], domains):
            if domain not in domain_results:
                domain_results[domain] = {'reject': 0, 'total': 0}
            domain_results[domain]['total'] += 1
            if decision == 'Reject':
                domain_results[domain]['reject'] += 1
        
        print(f"\nResults by domain:")
        for domain, stats in domain_results.items():
            rate = stats['reject'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {domain:<12} {stats['reject']:4d}/{stats['total']:4d} ({rate:.1%})")
            
    else:
        print(f"Model: {args.model}")
        print(f"Total samples: {len(embeddings)}")
        print(f"Fiber bundle rejection rate: {results['fiber_bundle']['rejection_rate']:.1%}")
        print(f"Overall intrinsic dimension: {results['dimensionality']['overall_intrinsic_dimension']}")
        print(f"Clustering quality: {results['clustering']['metrics']['silhouette_score']:.3f}")
        
        if 'stratum_dimensions' in results['dimensionality']:
            stratum_dims = results['dimensionality']['stratum_dimensions']
            print(f"Stratum dimensions: {list(stratum_dims.values())}")
    
    # Performance summary
    total_time = load_time + embedding_time + analysis_time
    print(f"\n⏱️ Performance Summary:")
    print(f"  Dataset loading: {load_time:.1f}s")
    print(f"  Embedding extraction: {embedding_time:.1f}s")
    print(f"  Analysis: {analysis_time:.1f}s")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Samples per second: {len(embeddings)/total_time:.1f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving results...")
    DataUtils.save_results(results, output_dir / f'{args.model}_results.json')
    
    if args.save_embeddings:
        DataUtils.save_embeddings(
            embeddings,
            output_dir / f'{args.model}_embeddings.npz',
            {
                'model': args.model,
                'total_samples': len(embeddings),
                'domains': list(set(domains)),
                'analysis_time': total_time
            }
        )
        print(f"✅ Embeddings saved")
    
    # Create summary
    summary = {
        'model': args.model,
        'total_samples': len(embeddings),
        'domains': list(set(domains)),
        'samples_per_domain': args.samples,
        'rejection_rate': results.get('rejection_rate') or results['fiber_bundle']['rejection_rate'],
        'analysis_time': total_time,
        'fast_mode': args.fast_mode
    }
    
    DataUtils.save_results(summary, output_dir / 'analysis_summary.json')
    
    print(f"✅ Results saved to: {output_dir}")
    print(f"\n🎉 Large dataset analysis complete!")
    
    return results


def main():
    """Main function."""
    args = parse_arguments()
    
    print(f"🚀 Large Dataset Analysis Configuration")
    print(f"Samples per domain: {args.samples}")
    print(f"Total samples: {args.samples * 6}")
    print(f"Model: {args.model}")
    print(f"Fast mode: {args.fast_mode}")
    
    if args.samples > 500:
        response = input(f"\n⚠️ Large dataset ({args.samples * 6} samples) - continue? (y/n): ")
        if response.lower() != 'y':
            print("Analysis cancelled.")
            return
    
    try:
        results = run_large_dataset_analysis(args)
        
        print(f"\n🎯 Recommendations for robust results:")
        print(f"  • Use samples ≥ 500 per domain for statistical significance")
        print(f"  • Compare multiple models for architecture insights")
        print(f"  • Save embeddings for further analysis")
        print(f"  • Use comprehensive mode for full geometric analysis")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        print(f"Try reducing --samples or using --fast-mode")


if __name__ == '__main__':
    main()
