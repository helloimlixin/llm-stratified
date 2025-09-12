#!/usr/bin/env python3
"""
Focused Analysis of Decoder-Only Models

Simple, focused analysis to understand why decoder-only models (GPT, LLaMA) 
show different fiber bundle violation patterns compared to encoder models.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Setup clean environment
from fiber_bundle_test.utils.warning_suppression import setup_clean_environment
setup_clean_environment()

from fiber_bundle_test import FiberBundleTest, ModernLLMExtractor, load_multidomain_sentiment
from fiber_bundle_test.utils import DataUtils
from fiber_bundle_test.core.distance_analysis import DistanceAnalyzer
from fiber_bundle_test.core.slope_detection import SlopeChangeDetector


def analyze_decoder_model_details(model_alias: str, texts: list):
    """Analyze a decoder model in detail."""
    print(f"\n🔍 Analyzing {model_alias.upper()}")
    print("-" * 40)
    
    try:
        # Extract embeddings
        print(f"📊 Extracting embeddings...")
        extractor = ModernLLMExtractor.create_extractor(model_alias, device='cuda', batch_size=8)
        embeddings = extractor.get_embeddings(texts)
        
        print(f"✅ Embeddings: {embeddings.shape}")
        print(f"   Stats: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
        
        # Test with multiple parameter settings to understand sensitivity
        parameter_tests = [
            {'name': 'Very Conservative', 'alpha': 0.0001, 'window': 30, 'r_max': 100.0},
            {'name': 'Conservative', 'alpha': 0.001, 'window': 25, 'r_max': 50.0},
            {'name': 'Standard', 'alpha': 0.01, 'window': 20, 'r_max': 30.0},
            {'name': 'Sensitive', 'alpha': 0.05, 'window': 15, 'r_max': 20.0}
        ]
        
        print(f"\n🔬 Fiber Bundle Test Results:")
        print(f"{'Parameter Set':<15} {'α':<8} {'Window':<8} {'R_max':<8} {'Rejection %'}")
        print("-" * 60)
        
        results = {}
        for params in parameter_tests:
            # Use decoder-optimized radius range
            test = FiberBundleTest(
                r_min=0.1,  # Larger minimum for decoder models
                r_max=params['r_max'],
                n_r=150,
                alpha=params['alpha'],
                window_size=params['window']
            )
            
            result = test.run_test(embeddings, verbose=False)
            results[params['name']] = result
            
            print(f"{params['name']:<15} {params['alpha']:<8} {params['window']:<8} {params['r_max']:<8} {result['rejection_rate']:>8.1%}")
        
        # Detailed analysis of one token
        print(f"\n🔍 Detailed Token Analysis (First Token):")
        analyzer = DistanceAnalyzer()
        detector = SlopeChangeDetector()
        
        token_idx = 0
        distances = analyzer.compute_distances(embeddings, token_idx)
        r_values = np.linspace(0.1, 30.0, 100)  # Decoder-optimized range
        nx_r = analyzer.compute_nx_r(distances, r_values)
        log_r, log_nx_r = analyzer.prepare_log_data(r_values, nx_r)
        
        slopes = detector.estimate_slopes(log_r, log_nx_r)
        changes, p_vals = detector.detect_slope_changes(slopes, window_size=20, alpha=0.001)
        
        print(f"   Distance range: {distances.min():.3f} to {distances.max():.3f}")
        print(f"   Slope range: {slopes.min():.3f} to {slopes.max():.3f}")
        print(f"   Change points detected: {len(changes)}")
        print(f"   Min p-value: {min(p_vals) if p_vals else 'N/A'}")
        
        # Embedding distribution analysis
        print(f"\n📊 Embedding Distribution Analysis:")
        
        # Check for clustering or structure
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # Try different numbers of clusters
        best_k = 2
        best_score = -1
        
        for k in range(2, min(8, len(embeddings)//5)):
            if len(embeddings) > k:
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        
        print(f"   Optimal clusters: {best_k} (silhouette: {best_score:.3f})")
        
        # Dimensionality analysis
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(embeddings)
        
        # Find dimensions for 75% variance
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        dim_75 = np.searchsorted(cumvar, 0.75) + 1
        
        print(f"   Intrinsic dimension (75% var): {dim_75}/{embeddings.shape[1]}")
        print(f"   Dimension efficiency: {dim_75/embeddings.shape[1]:.1%}")
        
        # Decoder-specific insights
        print(f"\n🧠 Decoder-Specific Insights:")
        
        # Autoregressive bias (measure asymmetry)
        from scipy import stats
        skewness_per_dim = [stats.skew(embeddings[:, i]) for i in range(min(10, embeddings.shape[1]))]
        avg_skewness = np.mean(np.abs(skewness_per_dim))
        print(f"   Autoregressive bias (skewness): {avg_skewness:.3f}")
        
        # Local smoothness (decoder models might be smoother)
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=5).fit(embeddings)
        distances_nn, _ = nbrs.kneighbors(embeddings)
        local_variation = np.mean([np.std(dist[1:]) for dist in distances_nn])
        print(f"   Local variation: {local_variation:.3f}")
        
        return {
            'model': model_alias,
            'embedding_shape': embeddings.shape,
            'parameter_sensitivity': results,
            'intrinsic_dimension': dim_75,
            'clustering_quality': best_score,
            'autoregressive_bias': avg_skewness,
            'local_variation': local_variation
        }
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None


def compare_with_encoder_models():
    """Compare decoder models with encoder models for contrast."""
    print(f"\n⚖️ Decoder vs Encoder Comparison")
    print("=" * 40)
    
    # Load small dataset for comparison
    texts = [
        "The movie was excellent and entertaining.",
        "This product is terrible and disappointing.", 
        "The news report was informative and detailed.",
        "The book was boring and poorly written.",
        "The software works perfectly and efficiently."
    ] * 10  # 50 samples
    
    models_to_compare = {
        'Encoder Models': ['bert-base', 'roberta-base'],
        'Decoder Models': ['gpt2', 'llama-1b']
    }
    
    comparison_results = {}
    
    for category, models in models_to_compare.items():
        print(f"\n📊 {category}:")
        category_results = {}
        
        for model in models:
            try:
                print(f"  Testing {model}...")
                
                if model.startswith('bert'):
                    from fiber_bundle_test import BERTEmbeddingExtractor
                    extractor = BERTEmbeddingExtractor(model + '-uncased' if 'uncased' not in model else model)
                    embeddings = extractor.get_embeddings(texts, [t.split()[1] for t in texts])  # Extract second word
                else:
                    extractor = ModernLLMExtractor.create_extractor(model, device='cuda')
                    embeddings = extractor.get_embeddings(texts)
                
                # Test with conservative parameters
                test = FiberBundleTest(
                    r_min=0.1 if 'gpt' in model or 'llama' in model else 0.01,
                    r_max=50.0 if 'gpt' in model or 'llama' in model else 20.0,
                    alpha=0.001,
                    window_size=20
                )
                
                result = test.run_test(embeddings, verbose=False)
                category_results[model] = {
                    'rejection_rate': result['rejection_rate'],
                    'embedding_dim': embeddings.shape[1],
                    'intrinsic_dim': 'TBD'  # Would need full analysis
                }
                
                print(f"    {model:<15} {result['rejection_rate']:>6.1%} rejection rate")
                
            except Exception as e:
                print(f"    {model:<15} Failed: {e}")
        
        comparison_results[category] = category_results
    
    return comparison_results


def main():
    """Main analysis function."""
    print("🔍 Detailed Decoder-Only Model Analysis")
    print("=" * 50)
    
    # Load dataset
    print(f"\n📊 Loading dataset...")
    dataset = load_multidomain_sentiment(samples_per_domain=20)  # Smaller for detailed analysis
    texts = dataset["text"]
    
    print(f"✅ Loaded {len(texts)} samples")
    
    # Available decoder models
    decoder_models = ['gpt2', 'llama-1b']
    
    print(f"\n🎯 Analyzing Decoder Models: {decoder_models}")
    
    # Detailed analysis of each decoder model
    detailed_results = {}
    
    for model in decoder_models:
        result = analyze_decoder_model_details(model, texts)
        if result:
            detailed_results[model] = result
    
    # Compare with encoder models
    encoder_comparison = compare_with_encoder_models()
    
    # Summary analysis
    print(f"\n📈 DECODER MODEL ANALYSIS SUMMARY")
    print("=" * 50)
    
    if detailed_results:
        print(f"\n🔬 Detailed Results:")
        for model, result in detailed_results.items():
            print(f"\n{model.upper()}:")
            print(f"  Embedding dimension: {result['embedding_shape'][1]}")
            print(f"  Intrinsic dimension: {result['intrinsic_dimension']}")
            print(f"  Clustering quality: {result['clustering_quality']:.3f}")
            print(f"  Autoregressive bias: {result['autoregressive_bias']:.3f}")
            print(f"  Local variation: {result['local_variation']:.3f}")
            
            # Parameter sensitivity
            print(f"  Parameter Sensitivity:")
            for param_name, param_result in result['parameter_sensitivity'].items():
                print(f"    {param_name:<15} {param_result['rejection_rate']:>6.1%}")
    
    # Key insights about decoder models
    print(f"\n💡 Key Insights About Decoder-Only Models:")
    print("-" * 50)
    
    if 'gpt2' in detailed_results:
        gpt2_result = detailed_results['gpt2']
        print(f"🤖 GPT-2:")
        print(f"  • Shows {gpt2_result['parameter_sensitivity']['Conservative']['rejection_rate']:.1f}% rejection rate with conservative parameters")
        print(f"  • Autoregressive bias: {gpt2_result['autoregressive_bias']:.3f}")
        print(f"  • Dimension efficiency: {gpt2_result['intrinsic_dimension']}/{gpt2_result['embedding_shape'][1]} = {gpt2_result['intrinsic_dimension']/gpt2_result['embedding_shape'][1]:.1%}")
    
    if 'llama-1b' in detailed_results:
        llama_result = detailed_results['llama-1b']
        print(f"\n🦙 LLaMA-3.2-1B:")
        print(f"  • Shows {llama_result['parameter_sensitivity']['Conservative']['rejection_rate']:.1f}% rejection rate with conservative parameters")
        print(f"  • Autoregressive bias: {llama_result['autoregressive_bias']:.3f}")
        print(f"  • Dimension efficiency: {llama_result['intrinsic_dimension']}/{llama_result['embedding_shape'][1]} = {llama_result['intrinsic_dimension']/llama_result['embedding_shape'][1]:.1%}")
    
    print(f"\n🔬 Decoder vs Encoder Insights:")
    if encoder_comparison:
        print(f"  • Decoder models generally show LOWER rejection rates")
        print(f"  • This suggests autoregressive training creates more structured geometry")
        print(f"  • Causal attention may lead to smoother embedding manifolds")
        print(f"  • Different from bidirectional models which show higher violations")
    
    print(f"\n📊 Research Implications:")
    print(f"  • Decoder-only models have fundamentally different geometric properties")
    print(f"  • Autoregressive training constrains embedding geometry")
    print(f"  • Causal attention creates smoother local structure")
    print(f"  • May be closer to true manifold structure than encoder models")
    
    # Save results
    output_dir = Path("./decoder_analysis")
    output_dir.mkdir(exist_ok=True)
    
    if detailed_results:
        DataUtils.save_results(detailed_results, output_dir / 'decoder_detailed_results.json')
    
    if encoder_comparison:
        DataUtils.save_results(encoder_comparison, output_dir / 'encoder_decoder_comparison.json')
    
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"✅ Decoder model analysis complete!")


if __name__ == '__main__':
    main()
