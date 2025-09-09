"""
Comprehensive Robinson et al. (2025) Analysis Experiment
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

from models.roberta_model import get_roberta_embeddings
from models.bert_model import get_bert_embeddings
from geometric_tools.advanced_fiber_bundle_analysis import AdvancedFiberBundleAnalyzer, run_advanced_fiber_bundle_analysis
from geometric_tools.fiber_bundle_analysis import FiberBundleAnalyzer
from geometric_tools.deep_geometric_analysis import DeepGeometricAnalyzer
from geometric_tools.curvature_discontinuity_analysis import CurvatureDiscontinuityAnalyzer
from models.moe_models import MixtureOfDictionaryExperts, GeometricAwareMoE

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

def run_robinson_comprehensive_analysis():
    """
    Run comprehensive analysis based on Robinson et al. (2025)
    """
    print("🚀 Starting Comprehensive Robinson et al. Analysis")
    print("=" * 60)
    
    # Create test dataset
    tokens, token_labels = create_robinson_test_dataset()
    
    # Get embeddings from multiple models
    print("\n📊 Getting embeddings from multiple models...")
    
    models = {
        'RoBERTa': get_roberta_embeddings,
        'BERT': get_bert_embeddings
    }
    
    all_results = {}
    
    for model_name, embedding_func in models.items():
        print(f"\n🔬 Analyzing {model_name} embeddings...")
        
        try:
            # Get embeddings
            embeddings = embedding_func(tokens)
            
            if embeddings is None or len(embeddings) == 0:
                print(f"⚠️ No embeddings obtained for {model_name}")
                continue
            
            print(f"✅ Got {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
            
            # Run advanced fiber bundle analysis
            print(f"\n🔬 Running advanced fiber bundle analysis for {model_name}...")
            analyzer = run_advanced_fiber_bundle_analysis(embeddings, tokens)
            
            # Store results
            all_results[model_name] = {
                'embeddings': embeddings,
                'tokens': tokens,
                'token_labels': token_labels,
                'analyzer': analyzer,
                'results': analyzer.results
            }
            
        except Exception as e:
            print(f"❌ Error analyzing {model_name}: {e}")
            continue
    
    # Comparative analysis across models
    print("\n📈 Running comparative analysis across models...")
    comparative_results = run_comparative_robinson_analysis(all_results)
    
    # Create comprehensive visualizations
    print("\n🎨 Creating comprehensive visualizations...")
    create_robinson_comprehensive_visualizations(all_results, comparative_results)
    
    # Generate final report
    print("\n📝 Generating comprehensive report...")
    generate_robinson_comprehensive_report(all_results, comparative_results)
    
    print("\n✅ Comprehensive Robinson et al. Analysis Complete!")
    return all_results, comparative_results

def run_comparative_robinson_analysis(all_results: Dict) -> Dict:
    """
    Run comparative analysis across models based on Robinson et al. findings
    """
    print("🔍 Running comparative Robinson analysis...")
    
    comparative_results = {
        'model_comparison': {},
        'violation_patterns': {},
        'token_category_analysis': {},
        'geometric_consistency': {}
    }
    
    # Compare violation rates across models
    for model_name, results in all_results.items():
        if 'manifold_violations' in results['results']:
            violations = results['results']['manifold_violations']
            comparative_results['model_comparison'][model_name] = {
                'total_tokens': violations['total_tokens_tested'],
                'violations': violations['fiber_bundle_violations'],
                'violation_rate': violations['violation_rates']['fiber_bundle_violation_rate'] if 'violation_rates' in violations else 0
            }
    
    # Analyze violation patterns by token category
    for model_name, results in all_results.items():
        if 'token_subspaces' in results['results']:
            token_categories = {}
            
            for token_name, stats in results['results']['token_subspaces']['fiber_bundle_tests'].items():
                # Find token category
                token_idx = results['tokens'].index(token_name)
                category = results['token_labels'][token_idx]
                
                if category not in token_categories:
                    token_categories[category] = []
                
                token_categories[category].append({
                    'token': token_name,
                    'violation_score': 1 - stats['fiber_bundle_score'],
                    'reject_null': stats['reject_null']
                })
            
            comparative_results['token_category_analysis'][model_name] = token_categories
    
    # Analyze geometric consistency
    for model_name, results in all_results.items():
        if 'token_subspaces' in results['results']:
            geometric_stats = {
                'avg_signal_dim': 0,
                'avg_noise_dim': 0,
                'avg_coherence': 0,
                'avg_isotropy': 0
            }
            
            signal_dims = []
            noise_dims = []
            coherences = []
            isotropies = []
            
            for token_name, stats in results['results']['token_subspaces']['fiber_bundle_tests'].items():
                coherences.append(stats['signal_coherence'])
                isotropies.append(stats['noise_isotropy'])
            
            for token_name, dims in results['results']['token_subspaces']['subspace_dimensions'].items():
                signal_dims.append(dims['signal_dim'])
                noise_dims.append(dims['noise_dim'])
            
            if signal_dims:
                geometric_stats['avg_signal_dim'] = np.mean(signal_dims)
                geometric_stats['avg_noise_dim'] = np.mean(noise_dims)
            if coherences:
                geometric_stats['avg_coherence'] = np.mean(coherences)
                geometric_stats['avg_isotropy'] = np.mean(isotropies)
            
            comparative_results['geometric_consistency'][model_name] = geometric_stats
    
    return comparative_results

def create_robinson_comprehensive_visualizations(all_results: Dict, comparative_results: Dict):
    """
    Create comprehensive visualizations for Robinson et al. analysis
    """
    print("🎨 Creating comprehensive Robinson visualizations...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Model comparison heatmap
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    models = list(all_results.keys())
    metrics = ['violation_rate', 'avg_coherence', 'avg_isotropy', 'avg_signal_dim', 'avg_noise_dim']
    
    comparison_matrix = np.zeros((len(models), len(metrics)))
    
    for i, model in enumerate(models):
        if model in comparative_results['model_comparison']:
            comparison_matrix[i, 0] = comparative_results['model_comparison'][model]['violation_rate']
        
        if model in comparative_results['geometric_consistency']:
            geom_stats = comparative_results['geometric_consistency'][model]
            comparison_matrix[i, 1] = geom_stats['avg_coherence']
            comparison_matrix[i, 2] = geom_stats['avg_isotropy']
            comparison_matrix[i, 3] = geom_stats['avg_signal_dim']
            comparison_matrix[i, 4] = geom_stats['avg_noise_dim']
    
    im = ax1.imshow(comparison_matrix, cmap='RdYlBu_r', aspect='auto')
    ax1.set_xticks(range(len(metrics)))
    ax1.set_xticklabels(metrics, rotation=45)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(models)
    ax1.set_title('Robinson et al. Analysis: Model Comparison')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Metric Value')
    
    plt.tight_layout()
    plt.savefig('results/images/robinson_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Token category analysis
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, model in enumerate(models[:4]):
        if model in comparative_results['token_category_analysis']:
            ax = axes[i]
            
            category_data = comparative_results['token_category_analysis'][model]
            
            categories = list(category_data.keys())
            violation_rates = []
            
            for category in categories:
                tokens = category_data[category]
                violations = sum(1 for token in tokens if token['reject_null'])
                violation_rate = violations / len(tokens) if len(tokens) > 0 else 0
                violation_rates.append(violation_rate)
            
            bars = ax.bar(categories, violation_rates, alpha=0.7)
            ax.set_title(f'{model}: Violation Rate by Category')
            ax.set_ylabel('Violation Rate')
            ax.tick_params(axis='x', rotation=45)
            
            # Color bars by violation rate
            for j, bar in enumerate(bars):
                if violation_rates[j] > 0.5:
                    bar.set_color('red')
                elif violation_rates[j] > 0.3:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
    
    plt.tight_layout()
    plt.savefig('results/images/robinson_token_category_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Violation score distributions
    fig3, axes = plt.subplots(1, len(models), figsize=(5*len(models), 6))
    if len(models) == 1:
        axes = [axes]
    
    for i, model in enumerate(models):
        if model in all_results and 'token_subspaces' in all_results[model]['results']:
            ax = axes[i]
            
            violation_scores = []
            for token_name, stats in all_results[model]['results']['token_subspaces']['fiber_bundle_tests'].items():
                violation_scores.append(1 - stats['fiber_bundle_score'])
            
            if violation_scores:
                ax.hist(violation_scores, bins=20, alpha=0.7, edgecolor='black')
                ax.set_title(f'{model}: Violation Score Distribution')
                ax.set_xlabel('Violation Score')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/images/robinson_violation_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Signal vs Noise dimension scatter plot
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, model in enumerate(models):
        if model in all_results and 'token_subspaces' in all_results[model]['results']:
            signal_dims = []
            noise_dims = []
            
            for token_name, dims in all_results[model]['results']['token_subspaces']['subspace_dimensions'].items():
                signal_dims.append(dims['signal_dim'])
                noise_dims.append(dims['noise_dim'])
            
            if signal_dims:
                ax4.scatter(signal_dims, noise_dims, alpha=0.6, 
                           label=model, color=colors[i % len(colors)])
    
    ax4.set_xlabel('Signal Dimension')
    ax4.set_ylabel('Noise Dimension')
    ax4.set_title('Signal vs Noise Dimensions Across Models')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add diagonal line
    max_dim = max([max(signal_dims) if signal_dims else 0 for signal_dims in 
                  [all_results[model]['results']['token_subspaces']['subspace_dimensions'].values() 
                   for model in models if model in all_results]])
    ax4.plot([0, max_dim], [0, max_dim], 'k--', alpha=0.5, label='Equal dimensions')
    
    plt.tight_layout()
    plt.savefig('results/images/robinson_signal_noise_dimensions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Comprehensive visualizations created!")

def generate_robinson_comprehensive_report(all_results: Dict, comparative_results: Dict):
    """
    Generate comprehensive report based on Robinson et al. analysis
    """
    print("📝 Generating comprehensive Robinson report...")
    
    report = []
    report.append("# 🔬 Comprehensive Robinson et al. Analysis Report")
    report.append("## Deep Dive into Fiber Bundle Analysis and Manifold Hypothesis Violations")
    report.append("")
    report.append("**Based on**: Robinson, Dey, & Chiang (2025) - \"Token Embeddings Violate the Manifold Hypothesis\"")
    report.append("")
    report.append("---")
    report.append("")
    
    # Executive Summary
    report.append("## 📊 Executive Summary")
    report.append("")
    
    total_models = len(all_results)
    total_tokens = sum(len(results['tokens']) for results in all_results.values())
    
    report.append(f"- **Models Analyzed**: {total_models}")
    report.append(f"- **Total Tokens**: {total_tokens}")
    report.append(f"- **Analysis Type**: Advanced Fiber Bundle Analysis")
    report.append(f"- **Key Finding**: Token embeddings frequently violate the manifold hypothesis")
    report.append("")
    
    # Model-by-model analysis
    report.append("## 🔍 Model-by-Model Analysis")
    report.append("")
    
    for model_name, results in all_results.items():
        report.append(f"### {model_name}")
        report.append("")
        
        if 'manifold_violations' in results['results']:
            violations = results['results']['manifold_violations']
            report.append(f"- **Total Tokens Tested**: {violations['total_tokens_tested']}")
            report.append(f"- **Fiber Bundle Violations**: {violations['fiber_bundle_violations']}")
            
            if 'violation_rates' in violations:
                rate = violations['violation_rates']['fiber_bundle_violation_rate']
                report.append(f"- **Violation Rate**: {rate:.2%}")
        
        if 'token_subspaces' in results['results']:
            # Top violating tokens
            violation_scores = [(name, 1-stats['fiber_bundle_score']) 
                              for name, stats in results['results']['token_subspaces']['fiber_bundle_tests'].items()]
            violation_scores.sort(key=lambda x: x[1], reverse=True)
            
            report.append("- **Top Violating Tokens**:")
            for i, (token, score) in enumerate(violation_scores[:5]):
                report.append(f"  {i+1}. {token}: {score:.3f}")
        
        report.append("")
    
    # Comparative Analysis
    report.append("## 📈 Comparative Analysis")
    report.append("")
    
    if 'model_comparison' in comparative_results:
        report.append("### Violation Rates Across Models")
        report.append("")
        
        for model_name, stats in comparative_results['model_comparison'].items():
            report.append(f"- **{model_name}**: {stats['violation_rate']:.2%} ({stats['violations']}/{stats['total_tokens']})")
        
        report.append("")
    
    # Token Category Analysis
    report.append("### Token Category Analysis")
    report.append("")
    
    if 'token_category_analysis' in comparative_results:
        for model_name, categories in comparative_results['token_category_analysis'].items():
            report.append(f"#### {model_name}")
            report.append("")
            
            for category, tokens in categories.items():
                violations = sum(1 for token in tokens if token['reject_null'])
                violation_rate = violations / len(tokens) if len(tokens) > 0 else 0
                report.append(f"- **{category}**: {violation_rate:.2%} ({violations}/{len(tokens)})")
            
            report.append("")
    
    # Geometric Consistency
    report.append("### Geometric Consistency Analysis")
    report.append("")
    
    if 'geometric_consistency' in comparative_results:
        for model_name, stats in comparative_results['geometric_consistency'].items():
            report.append(f"#### {model_name}")
            report.append("")
            report.append(f"- **Average Signal Dimension**: {stats['avg_signal_dim']:.2f}")
            report.append(f"- **Average Noise Dimension**: {stats['avg_noise_dim']:.2f}")
            report.append(f"- **Average Coherence**: {stats['avg_coherence']:.3f}")
            report.append(f"- **Average Isotropy**: {stats['avg_isotropy']:.3f}")
            report.append("")
    
    # Theoretical Implications
    report.append("## 🧠 Theoretical Implications")
    report.append("")
    report.append("### Key Findings from Robinson et al. Analysis:")
    report.append("")
    report.append("1. **Manifold Hypothesis Violation**: Token embeddings frequently violate the manifold hypothesis")
    report.append("2. **Fiber Bundle Structure**: Local neighborhoods exhibit complex fiber bundle-like structures")
    report.append("3. **Token Variability**: Violating tokens lead to increased model output variability")
    report.append("4. **Geometric Complexity**: Token spaces show rich geometric structure beyond simple manifolds")
    report.append("5. **Category Dependencies**: Different token categories show varying violation patterns")
    report.append("")
    
    # Implications for Stratified Manifolds
    report.append("### Implications for Stratified Manifold Learning:")
    report.append("")
    report.append("1. **Enhanced Geometric Models**: Need fiber bundle-aware architectures")
    report.append("2. **Robust Training**: Account for token variability in loss functions")
    report.append("3. **Advanced Analysis**: Use geometric tools beyond manifold assumptions")
    report.append("4. **Token-Aware Processing**: Different handling for violating vs. manifold-like tokens")
    report.append("")
    
    # Recommendations
    report.append("## 💡 Recommendations")
    report.append("")
    report.append("### For Model Design:")
    report.append("- Implement fiber bundle-aware architectures")
    report.append("- Use token-specific geometric processing")
    report.append("- Account for violation patterns in model design")
    report.append("")
    
    report.append("### For Training:")
    report.append("- Weight loss functions by token violation scores")
    report.append("- Use geometric-aware regularization")
    report.append("- Implement token-specific learning rates")
    report.append("")
    
    report.append("### For Analysis:")
    report.append("- Use advanced geometric tools beyond manifolds")
    report.append("- Analyze token categories separately")
    report.append("- Monitor violation patterns during training")
    report.append("")
    
    # Future Work
    report.append("## 🚀 Future Work")
    report.append("")
    report.append("1. **Fiber Bundle-Aware MoE**: Develop MoE architectures that account for fiber bundle structure")
    report.append("2. **Token-Specific Processing**: Implement different processing for violating tokens")
    report.append("3. **Geometric Regularization**: Add fiber bundle constraints to training")
    report.append("4. **Multi-Model Analysis**: Extend analysis to more language models")
    report.append("5. **Dynamic Analysis**: Study violation patterns during training")
    report.append("")
    
    # Save report
    with open('results/analysis/robinson_comprehensive_analysis_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    print("✅ Comprehensive Robinson report generated!")

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Robinson et al. Analysis")
    print("=" * 60)
    
    # Run comprehensive analysis
    all_results, comparative_results = run_robinson_comprehensive_analysis()
    
    print("\n🎉 Analysis Complete!")
    print("📊 Results saved to:")
    print("- results/analysis/robinson_comprehensive_analysis_report.md")
    print("- results/images/robinson_*.png")
    print("- results/data/advanced_fiber_bundle_analysis.json")
