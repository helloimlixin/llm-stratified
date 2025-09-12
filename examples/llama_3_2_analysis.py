#!/usr/bin/env python3
"""
Fiber Bundle Analysis with Llama-3.2-1B Model

This script demonstrates how to run the fiber bundle hypothesis test on the
new Llama-3.2-1B model, which is more efficient and accessible than larger
Llama models while still providing state-of-the-art performance.
"""

import sys
import logging
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fiber_bundle_test.embeddings.modern_llms import ModernLLMExtractor
from fiber_bundle_test.data.dataset_loaders import create_large_scale_dataset
from fiber_bundle_test.data.scalable_processing import ProcessingConfig, OptimizedHypothesisTest
from fiber_bundle_test.core import FiberBundleTest
from fiber_bundle_test.visualization import ResultsVisualizer
from fiber_bundle_test.utils import DataUtils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_llama_requirements():
    """Check if system can handle Llama-3.2-1B."""
    print("🔍 Checking System Requirements for Llama-3.2-1B")
    print("-" * 50)
    
    # Check GPU memory (1B model needs less than larger models)
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory >= 4:
            print("✅ GPU memory sufficient for Llama-3.2-1B")
            return True, 'cuda'
        else:
            print("⚠️ GPU memory may be tight, but Llama-3.2-1B should work")
            return True, 'cuda'
    else:
        print("⚠️ No GPU available, will use CPU (slower)")
        return True, 'cpu'


def run_llama_3_2_analysis():
    """Run fiber bundle analysis with Llama-3.2-1B."""
    print("🦙 Llama-3.2-1B Fiber Bundle Analysis")
    print("=" * 50)
    
    # Check requirements
    can_run, device = check_llama_requirements()
    if not can_run:
        print("❌ System requirements not met")
        return
    
    # Configuration optimized for Llama-3.2-1B
    config = ProcessingConfig(
        batch_size=8 if device == 'cuda' else 4,  # Smaller batches for safety
        max_workers=2,
        use_gpu=(device == 'cuda'),
        memory_limit="6GB" if device == 'cuda' else "4GB",
        cache_dir="./llama_3_2_cache",
        checkpoint_interval=50,
        resume_from_checkpoint=True
    )
    
    print(f"\n⚙️ Configuration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Memory limit: {config.memory_limit}")
    
    # Target tokens for analysis
    target_tokens = ['model', 'system', 'function', 'process', 'data', 'code']
    n_samples_per_token = 300  # Reasonable size for 1B model
    
    try:
        # 1. Create dataset
        print(f"\n📊 Creating dataset...")
        print(f"Target tokens: {target_tokens}")
        print(f"Samples per token: {n_samples_per_token}")
        
        sentences, tokens = create_large_scale_dataset(
            target_tokens, n_samples_per_token, sources=['wikipedia']
        )
        
        if len(sentences) == 0:
            print("⚠️ No sentences found, creating sample dataset...")
            sentences = []
            tokens = []
            for token in target_tokens:
                for i in range(n_samples_per_token):
                    sentences.append(f"This is a sample sentence containing the word {token} for analysis.")
                    tokens.append(token)
        
        print(f"✅ Dataset created: {len(sentences)} sentences")
        
        # 2. Initialize Llama-3.2-1B extractor
        print(f"\n🦙 Loading Llama-3.2-1B...")
        print("This may take a few minutes on first run...")
        
        try:
            extractor = ModernLLMExtractor.create_extractor(
                'llama-1b',  # Our new alias for Llama-3.2-1B
                device=device,
                batch_size=config.batch_size,
                max_length=512,
                # Additional parameters for Llama models
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                device_map='auto' if device == 'cuda' else None,
                trust_remote_code=True
            )
            print("✅ Llama-3.2-1B loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load Llama-3.2-1B: {e}")
            print("\nTroubleshooting:")
            print("1. Ensure you have access to the model:")
            print("   huggingface-cli login")
            print("2. Install required dependencies:")
            print("   pip install transformers>=4.30.0 torch>=2.0.0")
            print("3. Check if model name is correct: meta-llama/Llama-3.2-1B")
            return
        
        # 3. Extract embeddings
        print(f"\n🔄 Extracting embeddings...")
        
        from fiber_bundle_test.data.scalable_processing import ScalableEmbeddingProcessor
        processor = ScalableEmbeddingProcessor(config)
        
        data_iterator = iter([
            {'sentence': sentence, 'token': token}
            for sentence, token in zip(sentences, tokens)
        ])
        
        embeddings = processor.process_large_dataset(
            extractor, data_iterator, total_items=len(sentences)
        )
        
        print(f"✅ Embeddings extracted: {embeddings.shape}")
        
        # 4. Run hypothesis test
        print(f"\n🔬 Running fiber bundle hypothesis test...")
        
        test_params = {
            'r_min': 0.01,
            'r_max': 25.0,
            'n_r': 200,
            'alpha': 0.01,
            'window_size': 12
        }
        
        # Use optimized test for efficiency
        optimized_test = OptimizedHypothesisTest(config)
        results = optimized_test.run_large_scale_test(embeddings, test_params)
        
        # 5. Analyze and display results
        print(f"\n{'='*60}")
        print(f"LLAMA-3.2-1B FIBER BUNDLE ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"Model: Llama-3.2-1B (1B parameters)")
        print(f"Total tokens analyzed: {results['total_tokens']}")
        print(f"Hypothesis rejected: {results['total_rejections']}")
        print(f"Rejection rate: {results['rejection_rate']:.2%}")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        
        # Token-level analysis
        token_results = {}
        for (token_idx, decision), token in zip(results['results'], tokens):
            if token not in token_results:
                token_results[token] = {'reject': 0, 'total': 0}
            token_results[token]['total'] += 1
            if decision == 'Reject':
                token_results[token]['reject'] += 1
        
        print(f"\n🎯 Results by token:")
        print("-" * 40)
        for token, stats in token_results.items():
            rejection_rate = stats['reject'] / stats['total']
            print(f"{token:12} {stats['reject']:4d}/{stats['total']:4d} ({rejection_rate:.1%})")
        
        # Dimension analysis
        dimensions = results['dimensions']
        base_dims = [d[0] for d in dimensions if d[0] is not None]
        fiber_dims = [d[1] for d in dimensions if d[1] is not None]
        
        if base_dims and fiber_dims:
            import numpy as np
            print(f"\n📏 Dimension Analysis:")
            print(f"Base dimension:  mean={np.mean(base_dims):.2f}, std={np.std(base_dims):.2f}")
            print(f"Fiber dimension: mean={np.mean(fiber_dims):.2f}, std={np.std(fiber_dims):.2f}")
            print(f"Fiber/Base ratio: {np.mean(fiber_dims)/np.mean(base_dims):.2f}")
        
        # 6. Save results
        output_dir = Path('./llama_3_2_output')
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n💾 Saving results...")
        
        # Save detailed results
        DataUtils.save_results(results, output_dir / 'llama_3_2_results.json')
        
        # Save summary
        summary = {
            'model': 'meta-llama/Llama-3.2-1B',
            'model_size': '1B parameters',
            'total_tokens': results['total_tokens'],
            'rejection_rate': results['rejection_rate'],
            'token_results': token_results,
            'test_parameters': test_params,
            'embedding_dimension': embeddings.shape[1]
        }
        
        import json
        with open(output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 7. Create visualizations
        print(f"📊 Creating visualizations...")
        
        visualizer = ResultsVisualizer()
        token_labels = [f"{token}" for token in target_tokens]
        
        # Create plots
        summary_fig = visualizer.plot_results_summary(results, tokens)
        dimension_fig = visualizer.plot_dimension_analysis(results, tokens)
        
        # Save plots
        plots_dir = output_dir / 'plots'
        visualizer.save_plots(
            [summary_fig, dimension_fig],
            ['llama_3_2_summary', 'llama_3_2_dimensions'],
            str(plots_dir)
        )
        
        print(f"✅ Analysis completed!")
        print(f"📁 Results saved to: {output_dir}")
        
        # 8. Model comparison suggestion
        print(f"\n💡 Insights about Llama-3.2-1B:")
        print(f"• Efficient 1B parameter model suitable for research")
        print(f"• Lower memory requirements than larger Llama models")
        print(f"• Good balance of performance and computational efficiency")
        print(f"• Rejection rate of {results['rejection_rate']:.1%} indicates strong fiber bundle violations")
        
        print(f"\n🔬 Next steps:")
        print(f"• Compare with other models: python examples/modern_llm_comparison.py")
        print(f"• Try larger datasets with more samples per token")
        print(f"• Analyze different token types or domains")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        print(f"\n❌ Analysis failed: {e}")
        print(f"\nTroubleshooting tips:")
        print(f"• Check Hugging Face authentication: huggingface-cli login")
        print(f"• Verify model access permissions")
        print(f"• Try reducing batch_size or using CPU")
        print(f"• Check available memory and disk space")


def compare_llama_models():
    """Compare different Llama model sizes."""
    print(f"\n🔄 Llama Model Comparison")
    print("-" * 30)
    
    llama_models = {
        'Llama-3.2-1B': 'llama-1b',
        'Llama-3.2-3B': 'llama-3b',
        'Llama-2-7B': 'llama-7b'
    }
    
    print("Available Llama models for comparison:")
    for name, alias in llama_models.items():
        print(f"  {alias:<12} -> {name}")
    
    print(f"\nTo compare models, use:")
    print(f"python examples/modern_llm_comparison.py")


if __name__ == '__main__':
    print("🦙 Llama-3.2-1B Fiber Bundle Analysis")
    print("This script analyzes embedding geometry using Meta's efficient 1B parameter model")
    
    try:
        results = run_llama_3_2_analysis()
        
        if results:
            compare_llama_models()
            
    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check the error messages above for troubleshooting guidance")
