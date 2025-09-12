#!/usr/bin/env python3
"""
Example showing how to use custom configuration files.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fiber_bundle_test import FiberBundleTest, BERTEmbeddingExtractor
from fiber_bundle_test.embeddings.bert_embeddings import create_sample_dataset
from fiber_bundle_test.utils import ConfigLoader, DataUtils


def create_custom_config():
    """Create a custom configuration file."""
    config = {
        'test_parameters': {
            'r_min': 0.005,
            'r_max': 25.0,
            'n_r': 250,
            'alpha': 0.01,
            'window_size': 15
        },
        'embedding_parameters': {
            'model_name': 'bert-base-uncased'
        },
        'output': {
            'save_embeddings': True,
            'save_results': True,
            'output_dir': './custom_output'
        }
    }
    
    config_path = Path('./config/custom_config.yaml')
    config_path.parent.mkdir(exist_ok=True)
    
    ConfigLoader.save_config(config, str(config_path))
    print(f"Saved custom configuration to: {config_path}")
    
    return str(config_path)


def run_with_config(config_path):
    """Run analysis using configuration file."""
    print(f"Loading configuration from: {config_path}")
    config = ConfigLoader.load_config(config_path)
    
    # Extract parameters
    test_params = config['test_parameters']
    embedding_params = config['embedding_parameters']
    output_config = config['output']
    
    # Load data
    sentences, target_tokens = create_sample_dataset()
    print(f"Loaded {len(sentences)} sentences")
    
    # Extract embeddings
    extractor = BERTEmbeddingExtractor(embedding_params['model_name'])
    embeddings = extractor.get_embeddings(sentences, target_tokens)
    
    # Run test with custom parameters
    test = FiberBundleTest(**test_params)
    results = test.run_test(embeddings)
    
    # Print results
    print(f"\nResults with custom configuration:")
    print(f"  Parameters used: {test_params}")
    print(f"  Total tokens: {results['total_tokens']}")
    print(f"  Rejections: {results['total_rejections']}")
    print(f"  Rejection rate: {results['rejection_rate']:.2%}")
    
    # Save results if configured
    if output_config.get('save_results', False):
        output_dir = Path(output_config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        if output_config.get('save_embeddings', False):
            DataUtils.save_embeddings(
                embeddings,
                output_dir / 'embeddings.npz',
                {'sentences': sentences, 'target_tokens': target_tokens}
            )
        
        DataUtils.save_results(results, output_dir / 'results.json')
        print(f"Results saved to: {output_dir}")


def main():
    """Main function demonstrating configuration usage."""
    print("Custom Configuration Example")
    print("=" * 30)
    
    # Create and use custom config
    config_path = create_custom_config()
    run_with_config(config_path)
    
    print("\nDefault Configuration Example")
    print("=" * 30)
    
    # Show default configuration
    default_config = ConfigLoader.get_default_config()
    print("Default configuration:")
    for section, params in default_config.items():
        print(f"  {section}: {params}")


if __name__ == '__main__':
    main()
