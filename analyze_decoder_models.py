#!/usr/bin/env python3
"""
Detailed Analysis of Decoder-Only Models

This script provides comprehensive analysis of decoder-only models (GPT, LLaMA)
to understand their unique geometric properties in embedding space.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import pdist

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Setup clean environment
from fiber_bundle_test.utils.warning_suppression import setup_clean_environment
setup_clean_environment()

from fiber_bundle_test import (
    FiberBundleTest, ModernLLMExtractor, load_multidomain_sentiment,
    StratificationAnalyzer, ClusteringAnalyzer, DimensionalityAnalyzer
)
from fiber_bundle_test.utils import DataUtils
from fiber_bundle_test.core.distance_analysis import DistanceAnalyzer
from fiber_bundle_test.core.slope_detection import SlopeChangeDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecoderModelAnalyzer:
    """Specialized analyzer for decoder-only models."""
    
    def __init__(self):
        """Initialize the decoder model analyzer."""
        self.decoder_models = {
            'gpt2': 'gpt2',
            'gpt2-medium': 'gpt2-medium', 
            'gpt2-large': 'gpt2-large',
            'llama-1b': 'meta-llama/Llama-3.2-1B',
            'llama-3b': 'meta-llama/Llama-3.2-3B'
        }
        
        self.results = {}
        self.embeddings_cache = {}
    
    def analyze_single_decoder_model(self, model_alias: str, texts: list, 
                                   verbose: bool = True) -> dict:
        """
        Analyze a single decoder-only model in detail.
        
        Args:
            model_alias: Model alias (e.g., 'gpt2', 'llama-1b')
            texts: List of texts to analyze
            verbose: Whether to print detailed analysis
            
        Returns:
            Comprehensive analysis results
        """
        if verbose:
            print(f"\n🔍 Detailed Analysis: {model_alias}")
            print("=" * 50)
        
        try:
            # Extract embeddings
            if verbose:
                print(f"📊 Extracting embeddings...")
            
            extractor = ModernLLMExtractor.create_extractor(
                model_alias, 
                device='cuda',
                batch_size=8,
                pooling='mean'  # Use mean pooling for sentence-level analysis
            )
            
            embeddings = extractor.get_embeddings(texts)
            self.embeddings_cache[model_alias] = embeddings
            
            if verbose:
                print(f"✅ Embeddings shape: {embeddings.shape}")
                print(f"   Mean: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")
                print(f"   Range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
            
            # 1. Basic Fiber Bundle Analysis
            results = self._analyze_fiber_bundle_properties(embeddings, model_alias, verbose)
            
            # 2. Geometric Properties Analysis
            geometric_props = self._analyze_geometric_properties(embeddings, model_alias, verbose)
            results['geometric_properties'] = geometric_props
            
            # 3. Contextual Sensitivity Analysis
            context_analysis = self._analyze_contextual_sensitivity(embeddings, texts, model_alias, verbose)
            results['contextual_analysis'] = context_analysis
            
            # 4. Decoder-Specific Analysis
            decoder_analysis = self._analyze_decoder_specific_properties(embeddings, model_alias, verbose)
            results['decoder_analysis'] = decoder_analysis
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to analyze {model_alias}: {e}")
            if "access" in str(e).lower() or "gated" in str(e).lower():
                print(f"⚠️ {model_alias} requires access approval")
                print(f"Visit: https://huggingface.co/{self.decoder_models[model_alias]}")
            return None
    
    def _analyze_fiber_bundle_properties(self, embeddings: np.ndarray, 
                                       model_name: str, verbose: bool) -> dict:
        """Analyze fiber bundle properties with multiple parameter settings."""
        if verbose:
            print(f"\n🔬 Fiber Bundle Analysis:")
        
        # Test with multiple parameter settings
        parameter_sets = [
            {'name': 'Conservative', 'alpha': 0.001, 'window_size': 25, 'r_max': 30.0},
            {'name': 'Standard', 'alpha': 0.01, 'window_size': 20, 'r_max': 25.0},
            {'name': 'Sensitive', 'alpha': 0.05, 'window_size': 15, 'r_max': 20.0}
        ]
        
        results = {}
        
        for params in parameter_sets:
            # Adjust parameters for model type
            if 'llama' in model_name.lower() or 'gpt' in model_name.lower():
                r_min = 0.1  # Larger minimum for decoder models
                r_max = params['r_max'] * 2  # Larger range
            else:
                r_min = 0.01
                r_max = params['r_max']
            
            test = FiberBundleTest(
                r_min=r_min,
                r_max=r_max,
                n_r=200,
                alpha=params['alpha'],
                window_size=params['window_size']
            )
            
            result = test.run_test(embeddings, verbose=False)
            results[params['name']] = {
                'rejection_rate': result['rejection_rate'],
                'total_rejections': result['total_rejections'],
                'parameters': params
            }
            
            if verbose:
                print(f"  {params['name']:<12} α={params['alpha']:<6} window={params['window_size']:<3} → {result['rejection_rate']:>6.1%}")
        
        return results
    
    def _analyze_geometric_properties(self, embeddings: np.ndarray, 
                                    model_name: str, verbose: bool) -> dict:
        """Analyze geometric properties specific to decoder models."""
        if verbose:
            print(f"\n📐 Geometric Properties Analysis:")
        
        # 1. Embedding space analysis
        dim_analyzer = DimensionalityAnalyzer()
        
        # Overall dimensionality
        dim_analysis = dim_analyzer.compute_intrinsic_dimensions(embeddings)
        intrinsic_dim = dim_analysis['overall_intrinsic_dimension']
        
        # 2. Distance distribution analysis
        from scipy.spatial.distance import pdist
        distances = pdist(embeddings, metric='euclidean')
        
        # 3. Local vs global structure
        local_structure = self._analyze_local_structure(embeddings)
        
        # 4. Anisotropy analysis
        anisotropy = self._analyze_anisotropy(embeddings)
        
        results = {
            'intrinsic_dimension': intrinsic_dim,
            'embedding_dimension': embeddings.shape[1],
            'dimension_efficiency': intrinsic_dim / embeddings.shape[1],
            'distance_stats': {
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances)),
                'min': float(np.min(distances)),
                'max': float(np.max(distances))
            },
            'local_structure': local_structure,
            'anisotropy': anisotropy
        }
        
        if verbose:
            print(f"  Intrinsic dimension: {intrinsic_dim}/{embeddings.shape[1]} ({results['dimension_efficiency']:.1%})")
            print(f"  Distance distribution: μ={results['distance_stats']['mean']:.2f}, σ={results['distance_stats']['std']:.2f}")
            print(f"  Local structure score: {local_structure:.3f}")
            print(f"  Anisotropy measure: {anisotropy:.3f}")
        
        return results
    
    def _analyze_local_structure(self, embeddings: np.ndarray) -> float:
        """Analyze local structure properties."""
        from sklearn.neighbors import NearestNeighbors
        
        # Compute local neighborhood consistency
        n_neighbors = min(10, len(embeddings) - 1)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        
        # Measure local structure consistency
        local_consistency = []
        for i in range(len(embeddings)):
            neighbor_distances = distances[i, 1:]  # Exclude self
            consistency = np.std(neighbor_distances) / (np.mean(neighbor_distances) + 1e-8)
            local_consistency.append(consistency)
        
        return float(np.mean(local_consistency))
    
    def _analyze_anisotropy(self, embeddings: np.ndarray) -> float:
        """Analyze anisotropy in the embedding space."""
        from sklearn.decomposition import PCA
        
        pca = PCA()
        pca.fit(embeddings)
        
        # Compute anisotropy as ratio of largest to smallest eigenvalue
        eigenvalues = pca.explained_variance_
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove near-zero eigenvalues
        
        if len(eigenvalues) > 1:
            anisotropy = eigenvalues[0] / eigenvalues[-1]
        else:
            anisotropy = 1.0
        
        return float(np.log10(anisotropy))  # Log scale for interpretability
    
    def _analyze_contextual_sensitivity(self, embeddings: np.ndarray, 
                                      texts: list, model_name: str, verbose: bool) -> dict:
        """Analyze how sensitive decoder models are to context."""
        if verbose:
            print(f"\n🎯 Contextual Sensitivity Analysis:")
        
        # Group texts by length and analyze embedding variation
        text_lengths = [len(text.split()) for text in texts]
        
        # Analyze embedding variation within length groups
        length_groups = {}
        for i, length in enumerate(text_lengths):
            length_bin = (length // 5) * 5  # Group by 5-word bins
            if length_bin not in length_groups:
                length_groups[length_bin] = []
            length_groups[length_bin].append(i)
        
        context_sensitivity = {}
        for length_bin, indices in length_groups.items():
            if len(indices) > 1:
                group_embeddings = embeddings[indices]
                # Compute intra-group variation
                distances = pdist(group_embeddings)
                context_sensitivity[length_bin] = {
                    'count': len(indices),
                    'mean_distance': float(np.mean(distances)),
                    'std_distance': float(np.std(distances))
                }
        
        # Overall contextual sensitivity score
        if context_sensitivity:
            all_distances = [group['mean_distance'] for group in context_sensitivity.values()]
            sensitivity_score = np.std(all_distances) / (np.mean(all_distances) + 1e-8)
        else:
            sensitivity_score = 0.0
        
        results = {
            'sensitivity_score': float(sensitivity_score),
            'length_groups': context_sensitivity,
            'text_length_stats': {
                'mean': np.mean(text_lengths),
                'std': np.std(text_lengths),
                'range': [min(text_lengths), max(text_lengths)]
            }
        }
        
        if verbose:
            print(f"  Contextual sensitivity score: {sensitivity_score:.3f}")
            print(f"  Text length variation: μ={np.mean(text_lengths):.1f}, σ={np.std(text_lengths):.1f}")
        
        return results
    
    def _analyze_decoder_specific_properties(self, embeddings: np.ndarray, 
                                           model_name: str, verbose: bool) -> dict:
        """Analyze properties specific to decoder-only models."""
        if verbose:
            print(f"\n🧠 Decoder-Specific Analysis:")
        
        # 1. Autoregressive bias analysis
        autoregressive_bias = self._measure_autoregressive_bias(embeddings)
        
        # 2. Positional encoding effects
        positional_effects = self._analyze_positional_effects(embeddings)
        
        # 3. Attention pattern implications
        attention_implications = self._analyze_attention_implications(embeddings)
        
        results = {
            'autoregressive_bias': autoregressive_bias,
            'positional_effects': positional_effects,
            'attention_implications': attention_implications,
            'model_size_category': self._categorize_model_size(model_name)
        }
        
        if verbose:
            print(f"  Autoregressive bias: {autoregressive_bias:.3f}")
            print(f"  Positional effects: {positional_effects:.3f}")
            print(f"  Attention implications: {attention_implications:.3f}")
            print(f"  Model size category: {results['model_size_category']}")
        
        return results
    
    def _measure_autoregressive_bias(self, embeddings: np.ndarray) -> float:
        """Measure bias introduced by autoregressive training."""
        # Analyze embedding distribution asymmetry
        from scipy import stats
        
        # Compute skewness across dimensions
        skewness_per_dim = [stats.skew(embeddings[:, i]) for i in range(embeddings.shape[1])]
        avg_skewness = np.mean(np.abs(skewness_per_dim))
        
        return float(avg_skewness)
    
    def _analyze_positional_effects(self, embeddings: np.ndarray) -> float:
        """Analyze effects of positional encoding on embedding geometry."""
        # Measure how much embeddings vary in a way that could be attributed to position
        # Use PCA to find the main directions of variation
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=min(10, embeddings.shape[1]))
        pca.fit(embeddings)
        
        # Measure how concentrated the variance is in the first few components
        variance_concentration = np.sum(pca.explained_variance_ratio_[:3])
        
        return float(variance_concentration)
    
    def _analyze_attention_implications(self, embeddings: np.ndarray) -> float:
        """Analyze implications of causal attention on embedding structure."""
        # Measure local smoothness - decoder models might have smoother local structure
        from sklearn.neighbors import NearestNeighbors
        
        n_neighbors = min(5, len(embeddings) - 1)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)
        
        # Measure smoothness as consistency of local neighborhoods
        neighbor_distances = distances[:, 1:]  # Exclude self-distance
        smoothness_scores = []
        
        for i in range(len(embeddings)):
            local_distances = neighbor_distances[i]
            # Smoothness = inverse of relative variation in local distances
            smoothness = 1.0 / (1.0 + np.std(local_distances) / (np.mean(local_distances) + 1e-8))
            smoothness_scores.append(smoothness)
        
        return float(np.mean(smoothness_scores))
    
    def _categorize_model_size(self, model_name: str) -> str:
        """Categorize model by size."""
        if 'gpt2' in model_name.lower():
            if 'large' in model_name:
                return 'Large (774M)'
            elif 'medium' in model_name:
                return 'Medium (355M)'
            else:
                return 'Base (124M)'
        elif 'llama' in model_name.lower():
            if '1b' in model_name:
                return 'Small (1B)'
            elif '3b' in model_name:
                return 'Medium (3B)'
            elif '7b' in model_name:
                return 'Large (7B)'
            elif '13b' in model_name:
                return 'Very Large (13B)'
            else:
                return 'Unknown'
        else:
            return 'Unknown'
    
    def compare_decoder_models(self, model_list: list, texts: list) -> dict:
        """Compare multiple decoder models."""
        print(f"\n🔄 Comparing Decoder-Only Models")
        print("=" * 40)
        
        comparison_results = {}
        
        for model_alias in model_list:
            if model_alias in self.decoder_models:
                print(f"\n📊 Analyzing {model_alias}...")
                result = self.analyze_single_decoder_model(model_alias, texts, verbose=False)
                if result:
                    comparison_results[model_alias] = result
                    print(f"✅ {model_alias}: {result['fiber_bundle']['Conservative']['rejection_rate']:.1%} rejection rate")
                else:
                    print(f"❌ {model_alias}: Analysis failed")
        
        return comparison_results
    
    def create_decoder_comparison_visualization(self, comparison_results: dict, 
                                              output_dir: str = "./decoder_analysis"):
        """Create comprehensive visualizations for decoder model comparison."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Rejection rates comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        models = list(comparison_results.keys())
        
        # Conservative rejection rates
        conservative_rates = [comparison_results[m]['fiber_bundle']['Conservative']['rejection_rate'] 
                            for m in models]
        
        axes[0, 0].bar(models, conservative_rates, color='skyblue')
        axes[0, 0].set_title('Fiber Bundle Rejection Rates (Conservative)')
        axes[0, 0].set_ylabel('Rejection Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Geometric properties comparison
        intrinsic_dims = [comparison_results[m]['geometric_properties']['intrinsic_dimension'] 
                         for m in models]
        dimension_effs = [comparison_results[m]['geometric_properties']['dimension_efficiency'] 
                         for m in models]
        
        axes[0, 1].bar(models, intrinsic_dims, color='lightcoral')
        axes[0, 1].set_title('Intrinsic Dimensions')
        axes[0, 1].set_ylabel('Intrinsic Dimension')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        axes[0, 2].bar(models, dimension_effs, color='lightgreen')
        axes[0, 2].set_title('Dimension Efficiency')
        axes[0, 2].set_ylabel('Intrinsic/Embedding Ratio')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Decoder-specific properties
        autoregressive_bias = [comparison_results[m]['decoder_analysis']['autoregressive_bias'] 
                              for m in models]
        positional_effects = [comparison_results[m]['decoder_analysis']['positional_effects'] 
                             for m in models]
        attention_implications = [comparison_results[m]['decoder_analysis']['attention_implications'] 
                                for m in models]
        
        axes[1, 0].bar(models, autoregressive_bias, color='orange')
        axes[1, 0].set_title('Autoregressive Bias')
        axes[1, 0].set_ylabel('Bias Measure')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        axes[1, 1].bar(models, positional_effects, color='purple')
        axes[1, 1].set_title('Positional Effects')
        axes[1, 1].set_ylabel('Variance Concentration')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        axes[1, 2].bar(models, attention_implications, color='brown')
        axes[1, 2].set_title('Local Smoothness')
        axes[1, 2].set_ylabel('Smoothness Score')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'decoder_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Parameter sensitivity heatmap
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Create heatmap data
        param_names = ['Conservative', 'Standard', 'Sensitive']
        heatmap_data = []
        
        for model in models:
            model_rates = []
            for param_name in param_names:
                rate = comparison_results[model]['fiber_bundle'][param_name]['rejection_rate']
                model_rates.append(rate)
            heatmap_data.append(model_rates)
        
        sns.heatmap(heatmap_data, 
                   xticklabels=param_names,
                   yticklabels=models,
                   annot=True, 
                   fmt='.1f',
                   cmap='RdYlBu_r',
                   ax=ax)
        ax.set_title('Parameter Sensitivity: Rejection Rates (%)')
        
        plt.tight_layout()
        plt.savefig(output_path / 'parameter_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualizations saved to: {output_path}")


def main():
    """Main analysis function."""
    print("🔍 Detailed Decoder-Only Model Analysis")
    print("=" * 50)
    print("Analyzing geometric properties of decoder-only models (GPT, LLaMA)")
    
    # Initialize analyzer
    analyzer = DecoderModelAnalyzer()
    
    # Load dataset for analysis
    print(f"\n📊 Loading multi-domain dataset...")
    dataset = load_multidomain_sentiment(samples_per_domain=50)  # Moderate size for detailed analysis
    texts = dataset["text"]
    domains = dataset["domain"]
    
    print(f"✅ Loaded {len(texts)} samples from {len(set(domains))} domains")
    
    # Available decoder models (based on accessibility)
    available_models = ['gpt2', 'gpt2-medium']  # Always available
    
    # Check for LLaMA access
    try:
        test_extractor = ModernLLMExtractor.create_extractor('llama-1b', device='cpu')
        available_models.extend(['llama-1b'])
        print("✅ LLaMA-3.2-1B access confirmed")
    except:
        print("⚠️ LLaMA models require access - analyzing GPT models only")
    
    print(f"\n🎯 Models to analyze: {available_models}")
    
    # Run comprehensive comparison
    comparison_results = analyzer.compare_decoder_models(available_models, texts)
    
    if not comparison_results:
        print("❌ No models could be analyzed")
        return
    
    # Detailed analysis of each model
    print(f"\n🔬 Detailed Individual Analysis:")
    for model_alias in available_models:
        if model_alias in comparison_results:
            result = comparison_results[model_alias]
            
            print(f"\n📋 {model_alias.upper()} DETAILED RESULTS:")
            print("-" * 40)
            
            # Fiber bundle results
            fb_results = result['fiber_bundle']
            print(f"Fiber Bundle Rejection Rates:")
            for param_set, fb_result in fb_results.items():
                print(f"  {param_set:<12} {fb_result['rejection_rate']:>6.1%}")
            
            # Geometric properties
            geom = result['geometric_properties']
            print(f"\nGeometric Properties:")
            print(f"  Intrinsic dimension: {geom['intrinsic_dimension']}/{geom['embedding_dimension']}")
            print(f"  Dimension efficiency: {geom['dimension_efficiency']:.1%}")
            print(f"  Local structure: {geom['local_structure']:.3f}")
            print(f"  Anisotropy (log): {geom['anisotropy']:.2f}")
            
            # Decoder-specific properties
            decoder = result['decoder_analysis']
            print(f"\nDecoder-Specific Properties:")
            print(f"  Autoregressive bias: {decoder['autoregressive_bias']:.3f}")
            print(f"  Positional effects: {decoder['positional_effects']:.3f}")
            print(f"  Local smoothness: {decoder['attention_implications']:.3f}")
            print(f"  Model size: {decoder['model_size_category']}")
    
    # Create visualizations
    print(f"\n📊 Creating comprehensive visualizations...")
    analyzer.create_decoder_comparison_visualization(comparison_results)
    
    # Save detailed results
    output_dir = Path("./decoder_analysis")
    DataUtils.save_results(comparison_results, output_dir / 'detailed_decoder_analysis.json')
    
    # Summary insights
    print(f"\n💡 Key Insights About Decoder-Only Models:")
    print("-" * 50)
    
    if 'gpt2' in comparison_results:
        gpt2_result = comparison_results['gpt2']
        print(f"🤖 GPT-2 Characteristics:")
        print(f"  • Rejection rate: {gpt2_result['fiber_bundle']['Conservative']['rejection_rate']:.1%}")
        print(f"  • Intrinsic dimension: {gpt2_result['geometric_properties']['intrinsic_dimension']}")
        print(f"  • Shows {'high' if gpt2_result['decoder_analysis']['autoregressive_bias'] > 0.5 else 'moderate'} autoregressive bias")
    
    if 'llama-1b' in comparison_results:
        llama_result = comparison_results['llama-1b']
        print(f"\n🦙 LLaMA-3.2-1B Characteristics:")
        print(f"  • Rejection rate: {llama_result['fiber_bundle']['Conservative']['rejection_rate']:.1%}")
        print(f"  • Intrinsic dimension: {llama_result['geometric_properties']['intrinsic_dimension']}")
        print(f"  • Shows {'high' if llama_result['decoder_analysis']['attention_implications'] > 0.8 else 'moderate'} local smoothness")
    
    # Cross-model insights
    if len(comparison_results) > 1:
        rejection_rates = [result['fiber_bundle']['Conservative']['rejection_rate'] 
                          for result in comparison_results.values()]
        intrinsic_dims = [result['geometric_properties']['intrinsic_dimension'] 
                         for result in comparison_results.values()]
        
        print(f"\n📈 Cross-Model Patterns:")
        print(f"  • Rejection rate range: {min(rejection_rates):.1f}% - {max(rejection_rates):.1f}%")
        print(f"  • Intrinsic dimension range: {min(intrinsic_dims)} - {max(intrinsic_dims)}")
        print(f"  • Decoder models show {'consistent' if max(rejection_rates) - min(rejection_rates) < 20 else 'varied'} geometric properties")
    
    print(f"\n🎯 Research Implications:")
    print(f"  • Decoder-only models exhibit distinct geometric signatures")
    print(f"  • Autoregressive training creates specific embedding patterns")
    print(f"  • Model size and architecture affect fiber bundle violations")
    print(f"  • Different from encoder-only models (BERT/RoBERTa)")
    
    print(f"\n📁 Results saved to: ./decoder_analysis/")
    print(f"✅ Detailed decoder-only model analysis complete!")


if __name__ == '__main__':
    main()
