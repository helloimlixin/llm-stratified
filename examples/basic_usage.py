#!/usr/bin/env python3
"""
Basic usage example for the fiber bundle hypothesis test.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fiber_bundle_test import FiberBundleTest, BERTEmbeddingExtractor
from fiber_bundle_test.utils import DataUtils


def basic_example():
    """Basic example using random embeddings."""
    print("Basic Fiber Bundle Test Example")
    print("=" * 35)
    
    # Create sample random embeddings
    print("Creating random embeddings for demonstration...")
    embeddings = DataUtils.create_sample_random_embeddings(
        n_tokens=50, 
        embedding_dim=512, 
        seed=42
    )
    print(f"Created embeddings shape: {embeddings.shape}")
    
    # Initialize test
    test = FiberBundleTest(
        r_min=0.1,
        r_max=10.0,
        n_r=100,
        alpha=0.05,
        window_size=5
    )
    
    # Run test
    print("Running hypothesis test...")
    results = test.run_test(embeddings)
    
    # Print results
    print(f"\nResults:")
    print(f"Total tokens: {results['total_tokens']}")
    print(f"Rejections: {results['total_rejections']}")
    print(f"Rejection rate: {results['rejection_rate']:.2%}")


def bert_example():
    """Example using BERT embeddings."""
    print("\nBERT Embeddings Example")
    print("=" * 25)
    
    # Sample sentences
    sentences = [
        "The bank near the river was flooded.",
        "She deposited money in the bank account.",
        "The river flowed through the valley.",
        "He wrote code for the software.",
        "The secret code unlocked the door."
    ]
    
    target_tokens = ["bank", "bank", "river", "code", "code"]
    
    print(f"Processing {len(sentences)} sentences...")
    
    # Extract BERT embeddings
    extractor = BERTEmbeddingExtractor()
    embeddings = extractor.get_embeddings(sentences, target_tokens)
    
    print(f"Extracted embeddings shape: {embeddings.shape}")
    
    # Run test
    test = FiberBundleTest()
    results = test.run_test(embeddings, verbose=True)
    
    # Print results with context
    token_labels = DataUtils.extract_token_labels(sentences, target_tokens)
    
    print(f"\nDetailed Results:")
    for (idx, decision), label, dims in zip(
        results['results'], token_labels, results['dimensions']
    ):
        print(f"  {label}: {decision}")
        if dims[0] is not None:
            print(f"    Base dim: {dims[0]:.2f}, Fiber dim: {dims[1]:.2f}")


if __name__ == '__main__':
    basic_example()
    bert_example()
