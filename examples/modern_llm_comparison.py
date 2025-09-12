#!/usr/bin/env python3
"""
Compare fiber bundle hypothesis results across multiple state-of-the-art LLMs.

This script runs the analysis on multiple modern models to compare how different
architectures and training approaches affect the fiber bundle structure.
"""

import sys
import logging
from pathlib import Path
import time
from typing import List, Dict, Any
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fiber_bundle_test.embeddings.modern_llms import ModernLLMExtractor
from fiber_bundle_test.data.dataset_loaders import create_large_scale_dataset
from fiber_bundle_test.data.scalable_processing import ProcessingConfig, ScalableEmbeddingProcessor
from fiber_bundle_test.core import FiberBundleTest
from fiber_bundle_test.visualization import ResultsVisualizer
from fiber_bundle_test.utils import DataUtils

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparison:
    """Compare fiber bundle results across multiple models."""
    
    def __init__(self, output_dir: str = "./model_comparison"):
        """Initialize model comparison."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def run_comparison(self, 
                      models: List[str],
                      target_tokens: List[str] = None,
                      n_samples_per_token: int = 1000,
                      test_params: Dict[str, Any] = None):
        """
        Run comparison across multiple models.
        
        Args:
            models: List of model names to compare
            target_tokens: Target tokens for analysis
            n_samples_per_token: Number of samples per token
            test_params: Test parameters
        """
        if target_tokens is None:
            target_tokens = ['bank', 'river', 'code', 'model', 'system', 'network']
        
        if test_params is None:
            test_params = {
                'r_min': 0.01,
                'r_max': 25.0,
                'n_r': 250,
                'alpha': 0.01,
                'window_size': 12
            }
        
        # Load dataset once
        logger.info("Loading dataset...")
        sentences, tokens = create_large_scale_dataset(
            target_tokens, n_samples_per_token, sources=['wikipedia']
        )
        
        logger.info(f"Dataset loaded: {len(sentences)} sentences")
        
        # Save dataset info
        with open(self.output_dir / 'dataset_info.json', 'w') as f:
            json.dump({
                'sentences_count': len(sentences),
                'target_tokens': target_tokens,
                'samples_per_token': n_samples_per_token,
                'token_distribution': {token: tokens.count(token) for token in target_tokens}
            }, f, indent=2)
        
        # Process each model
        for model_name in models:
            try:
                logger.info(f"Processing model: {model_name}")
                result = self._process_single_model(
                    model_name, sentences, tokens, test_params
                )
                self.results[model_name] = result
                
                # Save intermediate results
                self._save_model_result(model_name, result)
                
            except Exception as e:
                logger.error(f"Failed to process model {model_name}: {e}")
                continue
        
        # Generate comparison analysis
        self._generate_comparison_analysis()
        self._create_comparison_visualizations()
    
    def _process_single_model(self, 
                             model_name: str,
                             sentences: List[str],
                             tokens: List[str],
                             test_params: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single model."""
        start_time = time.time()
        
        # Create extractor
        try:
            extractor = ModernLLMExtractor.create_extractor(
                model_name,
                device='cuda',
                batch_size=16,
                max_length=512
            )
        except Exception as e:
            logger.warning(f"Failed to create extractor for {model_name}, trying CPU: {e}")
            extractor = ModernLLMExtractor.create_extractor(
                model_name,
                device='cpu',
                batch_size=8,
                max_length=512
            )
        
        # Extract embeddings
        logger.info(f"Extracting embeddings for {model_name}...")
        
        config = ProcessingConfig(
            batch_size=16,
            max_workers=2,
            use_gpu=True,
            memory_limit='6GB'
        )
        
        processor = ScalableEmbeddingProcessor(config)
        
        data_iterator = iter([
            {'sentence': sentence, 'token': token}
            for sentence, token in zip(sentences, tokens)
        ])
        
        embeddings = processor.process_large_dataset(
            extractor, data_iterator, total_items=len(sentences)
        )
        
        embedding_time = time.time() - start_time
        
        # Run hypothesis test
        logger.info(f"Running hypothesis test for {model_name}...")
        test_start = time.time()
        
        test = FiberBundleTest(**test_params)
        results = test.run_test(embeddings, verbose=False)
        
        test_time = time.time() - test_start
        total_time = time.time() - start_time
        
        # Add timing and model info
        results['model_name'] = model_name
        results['embedding_shape'] = embeddings.shape
        results['timing'] = {
            'embedding_extraction': embedding_time,
            'hypothesis_test': test_time,
            'total_time': total_time
        }
        
        # Add token-level analysis
        token_analysis = {}
        for (token_idx, decision), token in zip(results['results'], tokens):
            if token not in token_analysis:
                token_analysis[token] = {'reject': 0, 'total': 0, 'dimensions': []}
            
            token_analysis[token]['total'] += 1
            if decision == 'Reject':
                token_analysis[token]['reject'] += 1
            
            # Add dimensions if available
            dims = results['dimensions'][token_idx]
            if dims[0] is not None and dims[1] is not None:
                token_analysis[token]['dimensions'].append(dims)
        
        # Calculate rejection rates
        for token in token_analysis:
            stats = token_analysis[token]
            stats['rejection_rate'] = stats['reject'] / stats['total']
            
            # Average dimensions
            if stats['dimensions']:
                base_dims = [d[0] for d in stats['dimensions']]
                fiber_dims = [d[1] for d in stats['dimensions']]
                stats['avg_base_dim'] = np.mean(base_dims)
                stats['avg_fiber_dim'] = np.mean(fiber_dims)
                stats['std_base_dim'] = np.std(base_dims)
                stats['std_fiber_dim'] = np.std(fiber_dims)
        
        results['token_analysis'] = token_analysis
        
        logger.info(f"Model {model_name} completed in {total_time:.1f}s")
        logger.info(f"  Rejection rate: {results['rejection_rate']:.1%}")
        
        return results
    
    def _save_model_result(self, model_name: str, result: Dict[str, Any]):
        """Save individual model result."""
        model_dir = self.output_dir / 'models' / model_name.replace('/', '_')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        DataUtils.save_results(result, model_dir / 'results.json')
    
    def _generate_comparison_analysis(self):
        """Generate comparison analysis across models."""
        logger.info("Generating comparison analysis...")
        
        comparison = {
            'models_compared': list(self.results.keys()),
            'comparison_summary': {},
            'token_comparison': {},
            'dimension_comparison': {},
            'timing_comparison': {}
        }
        
        # Overall comparison
        for model_name, result in self.results.items():
            comparison['comparison_summary'][model_name] = {
                'total_tokens': result['total_tokens'],
                'total_rejections': result['total_rejections'],
                'rejection_rate': result['rejection_rate'],
                'embedding_dim': result['embedding_shape'][1] if len(result['embedding_shape']) > 1 else 0,
                'total_time': result['timing']['total_time']
            }
        
        # Token-level comparison
        all_tokens = set()
        for result in self.results.values():
            all_tokens.update(result['token_analysis'].keys())
        
        for token in all_tokens:
            comparison['token_comparison'][token] = {}
            for model_name, result in self.results.items():
                token_stats = result['token_analysis'].get(token, {})
                comparison['token_comparison'][token][model_name] = {
                    'rejection_rate': token_stats.get('rejection_rate', 0),
                    'sample_count': token_stats.get('total', 0),
                    'avg_base_dim': token_stats.get('avg_base_dim'),
                    'avg_fiber_dim': token_stats.get('avg_fiber_dim')
                }
        
        # Dimension comparison
        for model_name, result in self.results.items():
            dimensions = result['dimensions']
            base_dims = [d[0] for d in dimensions if d[0] is not None]
            fiber_dims = [d[1] for d in dimensions if d[1] is not None]
            
            comparison['dimension_comparison'][model_name] = {
                'base_dim_mean': np.mean(base_dims) if base_dims else 0,
                'base_dim_std': np.std(base_dims) if base_dims else 0,
                'fiber_dim_mean': np.mean(fiber_dims) if fiber_dims else 0,
                'fiber_dim_std': np.std(fiber_dims) if fiber_dims else 0,
                'samples_with_dimensions': len(base_dims)
            }
        
        # Timing comparison
        for model_name, result in self.results.items():
            comparison['timing_comparison'][model_name] = result['timing']
        
        # Save comparison
        with open(self.output_dir / 'model_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Print summary
        self._print_comparison_summary(comparison)
        
        return comparison
    
    def _print_comparison_summary(self, comparison: Dict[str, Any]):
        """Print comparison summary."""
        print(f"\n{'='*80}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        print(f"{'Model':<25} {'Rejection Rate':<15} {'Avg Base Dim':<12} {'Avg Fiber Dim':<12} {'Time (s)':<10}")
        print("-" * 80)
        
        for model_name in comparison['models_compared']:
            summary = comparison['comparison_summary'][model_name]
            dimensions = comparison['dimension_comparison'][model_name]
            timing = comparison['timing_comparison'][model_name]
            
            print(f"{model_name:<25} {summary['rejection_rate']:<14.1%} "
                  f"{dimensions['base_dim_mean']:<11.2f} {dimensions['fiber_dim_mean']:<11.2f} "
                  f"{timing['total_time']:<9.1f}")
        
        print(f"\nToken-level Analysis:")
        print("-" * 50)
        
        token_comp = comparison['token_comparison']
        for token in sorted(token_comp.keys()):
            print(f"\nToken: {token}")
            for model_name in comparison['models_compared']:
                stats = token_comp[token].get(model_name, {})
                rate = stats.get('rejection_rate', 0)
                count = stats.get('sample_count', 0)
                print(f"  {model_name:<20} {rate:<8.1%} ({count} samples)")
    
    def _create_comparison_visualizations(self):
        """Create comparison visualizations."""
        logger.info("Creating comparison visualizations...")
        
        if len(self.results) < 2:
            logger.warning("Need at least 2 models for comparison visualizations")
            return
        
        # Set up plotting
        plt.style.use('seaborn-v0_8')
        
        # 1. Rejection rates comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall rejection rates
        models = list(self.results.keys())
        rejection_rates = [self.results[model]['rejection_rate'] for model in models]
        
        axes[0, 0].bar(range(len(models)), rejection_rates)
        axes[0, 0].set_title('Overall Rejection Rates by Model')
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Rejection Rate')
        axes[0, 0].set_xticks(range(len(models)))
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        
        # Token-level rejection rates heatmap
        token_data = []
        all_tokens = set()
        for result in self.results.values():
            all_tokens.update(result['token_analysis'].keys())
        
        all_tokens = sorted(all_tokens)
        
        for model in models:
            model_rates = []
            for token in all_tokens:
                rate = self.results[model]['token_analysis'].get(token, {}).get('rejection_rate', 0)
                model_rates.append(rate)
            token_data.append(model_rates)
        
        im = axes[0, 1].imshow(token_data, cmap='RdYlBu_r', aspect='auto')
        axes[0, 1].set_title('Rejection Rates by Token and Model')
        axes[0, 1].set_xlabel('Token')
        axes[0, 1].set_ylabel('Model')
        axes[0, 1].set_xticks(range(len(all_tokens)))
        axes[0, 1].set_xticklabels(all_tokens, rotation=45, ha='right')
        axes[0, 1].set_yticks(range(len(models)))
        axes[0, 1].set_yticklabels(models)
        plt.colorbar(im, ax=axes[0, 1])
        
        # Base vs Fiber dimensions scatter
        for i, model in enumerate(models):
            dimensions = self.results[model]['dimensions']
            base_dims = [d[0] for d in dimensions if d[0] is not None]
            fiber_dims = [d[1] for d in dimensions if d[1] is not None]
            
            if base_dims and fiber_dims:
                axes[1, 0].scatter(base_dims, fiber_dims, label=model, alpha=0.6, s=20)
        
        axes[1, 0].set_title('Base vs Fiber Dimensions')
        axes[1, 0].set_xlabel('Base Dimension')
        axes[1, 0].set_ylabel('Fiber Dimension')
        axes[1, 0].legend()
        axes[1, 0].plot([0, 10], [0, 10], 'k--', alpha=0.5, label='y=x')
        
        # Processing time comparison
        times = [self.results[model]['timing']['total_time'] for model in models]
        axes[1, 1].bar(range(len(models)), times)
        axes[1, 1].set_title('Processing Time by Model')
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_xticks(range(len(models)))
        axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed dimension analysis
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Base dimensions distribution
        for model in models:
            dimensions = self.results[model]['dimensions']
            base_dims = [d[0] for d in dimensions if d[0] is not None]
            if base_dims:
                axes[0].hist(base_dims, alpha=0.6, label=model, bins=20)
        
        axes[0].set_title('Base Dimension Distributions')
        axes[0].set_xlabel('Base Dimension')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        
        # Fiber dimensions distribution
        for model in models:
            dimensions = self.results[model]['dimensions']
            fiber_dims = [d[1] for d in dimensions if d[1] is not None]
            if fiber_dims:
                axes[1].hist(fiber_dims, alpha=0.6, label=model, bins=20)
        
        axes[1].set_title('Fiber Dimension Distributions')
        axes[1].set_xlabel('Fiber Dimension')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dimension_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {self.output_dir}")


def main():
    """Run model comparison."""
    # Define models to compare
    models_to_compare = [
        'bert-base',           # Classic BERT
        'roberta-large',       # RoBERTa improvement
        'deberta-v3',          # DeBERTa v3
        'gpt2-large',          # GPT-2 Large
        'llama-1b',            # Llama-3.2-1B (efficient)
        'all-mpnet',           # Sentence Transformer
    ]
    
    # You can add more state-of-the-art models:
    # 'llama-7b',          # Llama 7B (requires more memory)
    # 'llama-13b',         # Llama 13B (requires significant memory)
    # 't5-large',          # T5 Large
    
    # Target tokens for analysis
    target_tokens = [
        'bank', 'river', 'code', 'model', 'system', 'network', 
        'process', 'function', 'data', 'information'
    ]
    
    # Test parameters optimized for comparison
    test_params = {
        'r_min': 0.005,
        'r_max': 30.0,
        'n_r': 300,
        'alpha': 0.005,
        'window_size': 15
    }
    
    # Run comparison
    comparison = ModelComparison(output_dir='./model_comparison_results')
    
    try:
        comparison.run_comparison(
            models=models_to_compare,
            target_tokens=target_tokens,
            n_samples_per_token=800,  # Reduced for faster processing
            test_params=test_params
        )
        
        print(f"\nModel comparison completed!")
        print(f"Results saved to: ./model_comparison_results")
        
    except Exception as e:
        logger.error(f"Error during model comparison: {e}", exc_info=True)


if __name__ == '__main__':
    main()
