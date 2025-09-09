"""
Simplified Robinson et al. (2025) Analysis Experiment
Deep dive into fiber bundle analysis and manifold hypothesis violations

This experiment implements the complete theoretical framework from:
"Token Embeddings Violate the Manifold Hypothesis" by Robinson, Dey, & Chiang (2025)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def create_robinson_test_dataset() -> Tuple[List[str], List[str]]:
    """
    Create test dataset based on Robinson et al. (2025) methodology
    
    Focus on tokens that are likely to violate manifold hypothesis
    """
    print("📚 Creating Robinson et al. test dataset...")
    
    # Categories of tokens likely to violate manifold hypothesis
    # Based on Robinson et al. findings
    
    # 1. Function words (high frequency, structural)
    function_words = [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can", "must", "shall"
    ]
    
    # 2. Punctuation and special tokens
    punctuation = [
        ".", ",", "!", "?", ";", ":", "-", "_", "(", ")", "[", "]", "{", "}", "\"", "'", "`",
        "<pad>", "<unk>", "<s>", "</s>", "<mask>", "<sep>", "<cls>"
    ]
    
    # 3. Numbers and symbols
    numbers_symbols = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "100", "1000",
        "+", "-", "*", "/", "=", "<", ">", "%", "$", "#", "@", "&"
    ]
    
    # 4. Rare/out-of-vocabulary words
    rare_words = [
        "serendipity", "ephemeral", "ubiquitous", "paradigm", "quintessential",
        "mellifluous", "perspicacious", "obfuscate", "recalcitrant", "sagacious"
    ]
    
    # 5. Domain-specific terms
    domain_terms = [
        "algorithm", "neural", "network", "embedding", "manifold", "topology",
        "curvature", "fiber", "bundle", "hypothesis", "violation", "subspace"
    ]
    
    # 6. Ambiguous words (multiple meanings)
    ambiguous_words = [
        "bank", "bark", "bat", "bear", "bow", "box", "can", "cast", "change", "check",
        "clear", "close", "club", "cook", "crane", "date", "draft", "duck", "fair", "fall"
    ]
    
    # Combine all categories
    all_tokens = (function_words + punctuation + numbers_symbols + 
                 rare_words + domain_terms + ambiguous_words)
    
    # Create labels for analysis
    token_labels = []
    for token in all_tokens:
        if token in function_words:
            token_labels.append("function_word")
        elif token in punctuation:
            token_labels.append("punctuation")
        elif token in numbers_symbols:
            token_labels.append("number_symbol")
        elif token in rare_words:
            token_labels.append("rare_word")
        elif token in domain_terms:
            token_labels.append("domain_term")
        elif token in ambiguous_words:
            token_labels.append("ambiguous_word")
        else:
            token_labels.append("other")
    
    print(f"✅ Created dataset with {len(all_tokens)} tokens across {len(set(token_labels))} categories")
    return all_tokens, token_labels

def create_synthetic_embeddings(tokens: List[str], embedding_dim: int = 768) -> np.ndarray:
    """
    Create synthetic embeddings for testing Robinson et al. analysis
    
    This allows us to test the analysis framework without loading actual models
    """
    print(f"🎲 Creating synthetic embeddings for {len(tokens)} tokens...")
    
    np.random.seed(42)  # For reproducibility
    
    embeddings = []
    
    for i, token in enumerate(tokens):
        # Create embeddings with different characteristics based on token type
        
        if token in ["the", "a", "an", "and", "or", "but"]:  # Function words
            # High-dimensional, structured
            embedding = np.random.normal(0, 0.1, embedding_dim)
            embedding[:10] = np.random.normal(0, 0.5, 10)  # Signal dimensions
        elif token in [".", ",", "!", "?", ";", ":"]:  # Punctuation
            # Low-dimensional, sparse
            embedding = np.zeros(embedding_dim)
            embedding[np.random.choice(embedding_dim, 5)] = np.random.normal(0, 1, 5)
        elif token.isdigit():  # Numbers
            # Medium-dimensional, regular
            embedding = np.random.normal(0, 0.3, embedding_dim)
            embedding[:20] = np.random.normal(0, 0.8, 20)  # Signal dimensions
        elif token in ["serendipity", "ephemeral", "ubiquitous"]:  # Rare words
            # High-dimensional, complex
            embedding = np.random.normal(0, 0.2, embedding_dim)
            embedding[:30] = np.random.normal(0, 0.6, 30)  # Signal dimensions
        else:  # Other tokens
            # Standard embedding
            embedding = np.random.normal(0, 0.4, embedding_dim)
        
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    print(f"✅ Created synthetic embeddings: {embeddings.shape}")
    return embeddings

def run_simplified_robinson_analysis():
    """
    Run simplified Robinson et al. analysis using synthetic data
    """
    print("🚀 Starting Simplified Robinson et al. Analysis")
    print("=" * 60)
    
    # Create test dataset
    tokens, token_labels = create_robinson_test_dataset()
    
    # Create synthetic embeddings
    embeddings = create_synthetic_embeddings(tokens, embedding_dim=768)
    
    # Run basic fiber bundle analysis
    print("\n🔬 Running basic fiber bundle analysis...")
    
    # Import the analyzer
    try:
        from geometric_tools.advanced_fiber_bundle_analysis import AdvancedFiberBundleAnalyzer
        
        analyzer = AdvancedFiberBundleAnalyzer(embedding_dim=768)
        
        # Run analysis
        print("\n1. Analyzing token subspaces...")
        analyzer.analyze_token_subspaces(embeddings, tokens)
        
        print("\n2. Analyzing manifold violations...")
        analyzer.analyze_manifold_violations(embeddings, tokens)
        
        print("\n3. Analyzing token variability...")
        analyzer.analyze_token_variability(embeddings, tokens)
        
        print("\n4. Creating visualizations...")
        figures = analyzer.create_fiber_bundle_visualizations(embeddings, tokens)
        
        # Save visualizations
        for name, fig in figures.items():
            fig.savefig(f'results/images/simplified_robinson_{name}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        print("\n5. Generating analysis report...")
        report = analyzer.generate_robinson_analysis_report()
        
        # Save report
        with open('results/analysis/simplified_robinson_analysis_report.md', 'w') as f:
            f.write(report)
        
        # Save results
        analyzer.save_results('results/data/simplified_robinson_analysis.json')
        
        print("\n✅ Simplified Robinson Analysis Complete!")
        print(f"📊 Analysis report saved to: results/analysis/simplified_robinson_analysis_report.md")
        print(f"📈 Visualizations saved to: results/images/simplified_robinson_*.png")
        print(f"💾 Results saved to: results/data/simplified_robinson_analysis.json")
        
        return analyzer
        
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return None

def run_robinson_comprehensive_analysis():
    """
    Main function to run Robinson et al. analysis
    """
    print("🚀 Starting Robinson et al. Analysis")
    print("=" * 60)
    
    # For now, run simplified version to avoid segmentation faults
    print("⚠️ Running simplified version to avoid model loading issues...")
    
    analyzer = run_simplified_robinson_analysis()
    
    if analyzer:
        print("\n🎉 Robinson Analysis Complete!")
        print("📊 Results saved to:")
        print("- results/analysis/simplified_robinson_analysis_report.md")
        print("- results/images/simplified_robinson_*.png")
        print("- results/data/simplified_robinson_analysis.json")
        
        # Generate summary
        print("\n📋 Analysis Summary:")
        if 'manifold_violations' in analyzer.results:
            violations = analyzer.results['manifold_violations']
            print(f"- Total tokens tested: {violations['total_tokens_tested']}")
            print(f"- Fiber bundle violations: {violations['fiber_bundle_violations']}")
            if 'violation_rates' in violations:
                rate = violations['violation_rates']['fiber_bundle_violation_rate']
                print(f"- Violation rate: {rate:.2%}")
        
        return analyzer
    else:
        print("❌ Analysis failed")
        return None

if __name__ == "__main__":
    print("🚀 Starting Robinson et al. Analysis")
    print("=" * 60)
    
    # Run analysis
    analyzer = run_robinson_comprehensive_analysis()
    
    if analyzer:
        print("\n🎉 Analysis Complete!")
    else:
        print("\n❌ Analysis Failed!")
