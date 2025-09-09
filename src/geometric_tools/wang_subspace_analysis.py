"""
Low-Dimensional Residual Subspace Analysis
Based on Wang et al. (2025) - "Attention Layers Add Into Low-Dimensional Residual Subspaces"

This module implements the theoretical framework from the paper to analyze
attention outputs as low-dimensional residual subspaces and address dead features.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class LowDimensionalSubspaceAnalyzer:
    """
    Low-dimensional residual subspace analysis based on Wang et al. (2025)
    
    Implements the theoretical framework for analyzing attention outputs
    as low-dimensional subspaces and addressing dead features.
    """
    
    def __init__(self, embedding_dim: int = 768, variance_threshold: float = 0.99):
        self.embedding_dim = embedding_dim
        self.variance_threshold = variance_threshold
        self.results = {}
        
    def analyze_attention_subspaces(self, attention_outputs: np.ndarray, 
                                  layer_names: List[str] = None) -> Dict:
        """
        Analyze attention outputs as low-dimensional residual subspaces
        
        Based on Wang et al. (2025) finding that ~60% of directions account for 99% variance
        """
        print("🔬 Analyzing attention outputs as low-dimensional subspaces...")
        
        n_samples, dim = attention_outputs.shape
        results = {
            'subspace_dimensions': {},
            'variance_explained': {},
            'residual_subspaces': {},
            'dead_feature_analysis': {},
            'subspace_constraints': {}
        }
        
        # Perform PCA to find principal components
        pca = PCA()
        pca.fit(attention_outputs)
        
        # Calculate cumulative variance explained
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Find dimension that explains threshold variance
        active_dim = np.where(cumulative_variance >= self.variance_threshold)[0][0] + 1
        
        # Calculate percentage of directions needed
        directions_percentage = (active_dim / dim) * 100
        
        print(f"📊 Found {active_dim} directions explain {self.variance_threshold:.1%} variance")
        print(f"📊 This represents {directions_percentage:.1f}% of total directions")
        
        # Store results
        results['subspace_dimensions'] = {
            'total_dimensions': dim,
            'active_dimensions': active_dim,
            'directions_percentage': directions_percentage,
            'variance_threshold': self.variance_threshold
        }
        
        results['variance_explained'] = {
            'explained_variance_ratio': pca.explained_variance_ratio_[:active_dim].tolist(),
            'cumulative_variance': cumulative_variance[:active_dim].tolist(),
            'total_variance_explained': cumulative_variance[active_dim-1]
        }
        
        # Analyze residual subspaces
        results['residual_subspaces'] = self._analyze_residual_subspaces(
            attention_outputs, pca, active_dim
        )
        
        # Analyze dead features
        results['dead_feature_analysis'] = self._analyze_dead_features(
            attention_outputs, pca, active_dim
        )
        
        # Generate subspace constraints
        results['subspace_constraints'] = self._generate_subspace_constraints(
            pca, active_dim
        )
        
        self.results['attention_subspaces'] = results
        return results
    
    def _analyze_residual_subspaces(self, attention_outputs: np.ndarray, 
                                  pca: PCA, active_dim: int) -> Dict:
        """
        Analyze residual subspaces in attention outputs
        
        Based on Wang et al. finding that attention outputs are confined to low-dimensional subspaces
        """
        # Project data onto active subspace
        active_subspace = pca.components_[:active_dim]
        active_projections = attention_outputs @ active_subspace.T
        
        # Project data onto residual subspace
        residual_subspace = pca.components_[active_dim:]
        residual_projections = attention_outputs @ residual_subspace.T
        
        # Analyze subspace properties
        active_variance = np.var(active_projections, axis=0)
        residual_variance = np.var(residual_projections, axis=0)
        
        # Calculate subspace coherence
        active_coherence = np.mean(active_variance) / np.std(active_variance) if np.std(active_variance) > 0 else 0
        residual_coherence = np.mean(residual_variance) / np.std(residual_variance) if np.std(residual_variance) > 0 else 0
        
        return {
            'active_subspace_dim': active_dim,
            'residual_subspace_dim': len(residual_subspace),
            'active_variance': active_variance.tolist(),
            'residual_variance': residual_variance.tolist(),
            'active_coherence': active_coherence,
            'residual_coherence': residual_coherence,
            'subspace_ratio': active_dim / (len(residual_subspace) + active_dim)
        }
    
    def _analyze_dead_features(self, attention_outputs: np.ndarray, 
                             pca: PCA, active_dim: int) -> Dict:
        """
        Analyze dead features in sparse dictionary learning
        
        Based on Wang et al. finding that low-rank structure causes dead features
        """
        # Simulate random initialization (as in sparse autoencoders)
        n_features = min(1000, attention_outputs.shape[1])  # Simulate 1M features
        random_features = np.random.normal(0, 0.1, (n_features, attention_outputs.shape[1]))
        
        # Project random features onto attention space
        projected_features = random_features @ attention_outputs.T
        
        # Identify dead features (low activation)
        feature_activations = np.max(np.abs(projected_features), axis=1)
        dead_threshold = np.percentile(feature_activations, 10)  # Bottom 10% are "dead"
        dead_features = feature_activations < dead_threshold
        
        dead_percentage = np.mean(dead_features) * 100
        
        print(f"💀 Dead features analysis: {dead_percentage:.1f}% dead features")
        
        # Analyze feature alignment with active subspace
        active_subspace = pca.components_[:active_dim]
        feature_alignment = np.abs(random_features @ active_subspace.T)
        avg_alignment = np.mean(feature_alignment, axis=1)
        
        # Features aligned with active subspace should be more active
        aligned_features = avg_alignment > np.percentile(avg_alignment, 50)
        aligned_dead_rate = np.mean(dead_features[aligned_features]) * 100
        misaligned_dead_rate = np.mean(dead_features[~aligned_features]) * 100
        
        return {
            'total_features': n_features,
            'dead_features': np.sum(dead_features),
            'dead_percentage': dead_percentage,
            'aligned_dead_rate': aligned_dead_rate,
            'misaligned_dead_rate': misaligned_dead_rate,
            'feature_activations': feature_activations.tolist(),
            'dead_threshold': dead_threshold
        }
    
    def _generate_subspace_constraints(self, pca: PCA, active_dim: int) -> Dict:
        """
        Generate subspace constraints for training
        
        Based on Wang et al. subspace-constrained training method
        """
        # Active subspace basis
        active_basis = pca.components_[:active_dim]
        
        # Residual subspace basis
        residual_basis = pca.components_[active_dim:]
        
        # Generate initialization constraints
        constraints = {
            'active_subspace_basis': active_basis.tolist(),
            'residual_subspace_basis': residual_basis.tolist(),
            'active_dimension': active_dim,
            'residual_dimension': len(residual_basis),
            'initialization_strategy': 'subspace_constrained',
            'constraint_strength': 0.1  # Hyperparameter for constraint strength
        }
        
        return constraints
    
    def analyze_subspace_evolution(self, attention_outputs_by_layer: Dict[str, np.ndarray]) -> Dict:
        """
        Analyze how attention subspaces evolve across layers
        
        Based on Wang et al. findings across diverse model families
        """
        print("🔄 Analyzing subspace evolution across layers...")
        
        evolution_results = {
            'layer_subspaces': {},
            'subspace_stability': {},
            'dimensionality_trends': {},
            'cross_layer_analysis': {}
        }
        
        layer_names = list(attention_outputs_by_layer.keys())
        layer_dimensions = []
        layer_variances = []
        
        for layer_name, attention_outputs in attention_outputs_by_layer.items():
            # Analyze each layer
            layer_results = self.analyze_attention_subspaces(attention_outputs, [layer_name])
            
            evolution_results['layer_subspaces'][layer_name] = layer_results
            
            # Extract key metrics
            active_dim = layer_results['subspace_dimensions']['active_dimensions']
            directions_pct = layer_results['subspace_dimensions']['directions_percentage']
            
            layer_dimensions.append(active_dim)
            layer_variances.append(directions_pct)
        
        # Analyze trends across layers
        if len(layer_dimensions) > 1:
            evolution_results['dimensionality_trends'] = {
                'dimension_trend': np.polyfit(range(len(layer_dimensions)), layer_dimensions, 1)[0],
                'variance_trend': np.polyfit(range(len(layer_variances)), layer_variances, 1)[0],
                'dimension_std': np.std(layer_dimensions),
                'variance_std': np.std(layer_variances)
            }
        
        # Cross-layer subspace analysis
        if len(layer_names) > 1:
            evolution_results['cross_layer_analysis'] = self._analyze_cross_layer_subspaces(
                attention_outputs_by_layer
            )
        
        self.results['subspace_evolution'] = evolution_results
        return evolution_results
    
    def _analyze_cross_layer_subspaces(self, attention_outputs_by_layer: Dict[str, np.ndarray]) -> Dict:
        """
        Analyze relationships between subspaces across layers
        """
        layer_names = list(attention_outputs_by_layer.keys())
        subspace_overlaps = {}
        
        for i, layer1 in enumerate(layer_names):
            for j, layer2 in enumerate(layer_names[i+1:], i+1):
                # Get attention outputs for both layers
                outputs1 = attention_outputs_by_layer[layer1]
                outputs2 = attention_outputs_by_layer[layer2]
                
                # Find active subspaces for both layers
                pca1 = PCA().fit(outputs1)
                pca2 = PCA().fit(outputs2)
                
                cumvar1 = np.cumsum(pca1.explained_variance_ratio_)
                cumvar2 = np.cumsum(pca2.explained_variance_ratio_)
                
                active_dim1 = np.where(cumvar1 >= self.variance_threshold)[0][0] + 1
                active_dim2 = np.where(cumvar2 >= self.variance_threshold)[0][0] + 1
                
                # Calculate subspace overlap
                subspace1 = pca1.components_[:active_dim1]
                subspace2 = pca2.components_[:active_dim2]
                
                # Compute principal angles between subspaces
                overlap = self._compute_subspace_overlap(subspace1, subspace2)
                
                subspace_overlaps[f"{layer1}_vs_{layer2}"] = {
                    'overlap_score': overlap,
                    'active_dim1': active_dim1,
                    'active_dim2': active_dim2,
                    'overlap_percentage': overlap * 100
                }
        
        return subspace_overlaps
    
    def _compute_subspace_overlap(self, subspace1: np.ndarray, subspace2: np.ndarray) -> float:
        """
        Compute overlap between two subspaces using principal angles
        """
        # Ensure both subspaces have the same dimension
        min_dim = min(subspace1.shape[0], subspace2.shape[1])
        subspace1_trunc = subspace1[:min_dim]
        subspace2_trunc = subspace2[:min_dim]
        
        # Compute principal angles
        U1, s, U2 = svd(subspace1_trunc @ subspace2_trunc.T)
        principal_angles = np.arccos(np.clip(s, 0, 1))
        
        # Overlap score (higher = more similar)
        overlap_score = np.mean(np.cos(principal_angles))
        
        return overlap_score
    
    def create_subspace_visualizations(self, attention_outputs: np.ndarray, 
                                     layer_names: List[str] = None) -> Dict[str, plt.Figure]:
        """
        Create comprehensive visualizations of subspace analysis
        
        Based on Wang et al. (2025) methodology
        """
        print("🎨 Creating subspace visualizations...")
        
        if 'attention_subspaces' not in self.results:
            self.analyze_attention_subspaces(attention_outputs, layer_names)
        
        figures = {}
        
        # 1. Variance explained plot
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        
        pca = PCA().fit(attention_outputs)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        
        ax1.plot(range(1, len(cumvar) + 1), cumvar, 'b-', linewidth=2, label='Cumulative Variance')
        ax1.axhline(y=self.variance_threshold, color='r', linestyle='--', 
                   label=f'{self.variance_threshold:.1%} Threshold')
        
        # Find active dimension
        active_dim = np.where(cumvar >= self.variance_threshold)[0][0] + 1
        ax1.axvline(x=active_dim, color='g', linestyle='--', 
                   label=f'Active Dimension ({active_dim})')
        
        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('Cumulative Variance Explained')
        ax1.set_title('Attention Output Subspace Analysis (Wang et al. 2025)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        figures['variance_explained'] = fig1
        
        # 2. Dead features analysis
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Simulate dead features analysis
        n_features = 1000
        random_features = np.random.normal(0, 0.1, (n_features, attention_outputs.shape[1]))
        projected_features = random_features @ attention_outputs.T
        feature_activations = np.max(np.abs(projected_features), axis=1)
        
        # Plot activation distribution
        ax2a.hist(feature_activations, bins=50, alpha=0.7, edgecolor='black')
        ax2a.axvline(x=np.percentile(feature_activations, 10), color='r', linestyle='--',
                    label='Dead Feature Threshold')
        ax2a.set_xlabel('Feature Activation')
        ax2a.set_ylabel('Frequency')
        ax2a.set_title('Feature Activation Distribution')
        ax2a.legend()
        ax2a.grid(True, alpha=0.3)
        
        # Plot dead features percentage
        dead_thresholds = np.arange(0.01, 0.5, 0.01)
        dead_percentages = []
        
        for threshold in dead_thresholds:
            dead_features = feature_activations < np.percentile(feature_activations, threshold * 100)
            dead_percentages.append(np.mean(dead_features) * 100)
        
        ax2b.plot(dead_thresholds * 100, dead_percentages, 'r-', linewidth=2)
        ax2b.set_xlabel('Dead Feature Threshold (%)')
        ax2b.set_ylabel('Dead Features (%)')
        ax2b.set_title('Dead Features vs Threshold')
        ax2b.grid(True, alpha=0.3)
        
        figures['dead_features'] = fig2
        
        # 3. Subspace dimension analysis
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        
        # Plot active vs residual dimensions
        active_dim = self.results['attention_subspaces']['subspace_dimensions']['active_dimensions']
        residual_dim = self.results['attention_subspaces']['residual_subspaces']['residual_subspace_dim']
        
        categories = ['Active Subspace', 'Residual Subspace']
        dimensions = [active_dim, residual_dim]
        colors = ['green', 'red']
        
        bars = ax3.bar(categories, dimensions, color=colors, alpha=0.7)
        ax3.set_ylabel('Number of Dimensions')
        ax3.set_title('Active vs Residual Subspace Dimensions')
        ax3.grid(True, alpha=0.3)
        
        # Add percentage labels
        total_dim = active_dim + residual_dim
        for i, bar in enumerate(bars):
            height = bar.get_height()
            percentage = (height / total_dim) * 100
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height}\n({percentage:.1f}%)', ha='center', va='bottom')
        
        figures['subspace_dimensions'] = fig3
        
        return figures
    
    def generate_wang_analysis_report(self) -> str:
        """
        Generate comprehensive analysis report based on Wang et al. (2025)
        """
        if not self.results:
            return "No analysis results available. Run analysis first."
        
        report = []
        report.append("# 🔬 Low-Dimensional Residual Subspace Analysis Report")
        report.append("## Based on Wang et al. (2025)")
        report.append("")
        
        # Attention subspace analysis
        if 'attention_subspaces' in self.results:
            report.append("### 📊 Attention Subspace Analysis")
            report.append("")
            
            subspace_dims = self.results['attention_subspaces']['subspace_dimensions']
            report.append(f"- **Total Dimensions**: {subspace_dims['total_dimensions']}")
            report.append(f"- **Active Dimensions**: {subspace_dims['active_dimensions']}")
            report.append(f"- **Directions Percentage**: {subspace_dims['directions_percentage']:.1f}%")
            report.append(f"- **Variance Threshold**: {subspace_dims['variance_threshold']:.1%}")
            report.append("")
            
            # Dead features analysis
            dead_features = self.results['attention_subspaces']['dead_feature_analysis']
            report.append("### 💀 Dead Features Analysis")
            report.append("")
            report.append(f"- **Total Features**: {dead_features['total_features']}")
            report.append(f"- **Dead Features**: {dead_features['dead_features']}")
            report.append(f"- **Dead Percentage**: {dead_features['dead_percentage']:.1f}%")
            report.append(f"- **Aligned Dead Rate**: {dead_features['aligned_dead_rate']:.1f}%")
            report.append(f"- **Misaligned Dead Rate**: {dead_features['misaligned_dead_rate']:.1f}%")
            report.append("")
        
        # Subspace evolution analysis
        if 'subspace_evolution' in self.results:
            report.append("### 🔄 Subspace Evolution Analysis")
            report.append("")
            
            evolution = self.results['subspace_evolution']
            if 'dimensionality_trends' in evolution:
                trends = evolution['dimensionality_trends']
                report.append(f"- **Dimension Trend**: {trends['dimension_trend']:.3f}")
                report.append(f"- **Variance Trend**: {trends['variance_trend']:.3f}")
                report.append(f"- **Dimension Std**: {trends['dimension_std']:.3f}")
                report.append(f"- **Variance Std**: {trends['variance_std']:.3f}")
            report.append("")
        
        # Theoretical implications
        report.append("### 🧠 Theoretical Implications")
        report.append("")
        report.append("Based on Wang et al. (2025) findings:")
        report.append("")
        report.append("1. **Low-Dimensional Subspaces**: Attention outputs are confined to surprisingly low-dimensional subspaces")
        report.append("2. **60% Rule**: About 60% of directions account for 99% of variance")
        report.append("3. **Dead Features**: Low-rank structure causes dead features in sparse dictionary learning")
        report.append("4. **Subspace Constraints**: Subspace-constrained training reduces dead features significantly")
        report.append("")
        
        # Implications for stratified manifolds
        report.append("### Implications for Stratified Manifold Learning:")
        report.append("")
        report.append("1. **Subspace-Aware MoE**: Design MoE architectures that respect attention subspaces")
        report.append("2. **Dead Feature Prevention**: Use subspace constraints in sparse dictionary learning")
        report.append("3. **Layer-Specific Analysis**: Account for subspace evolution across layers")
        report.append("4. **Geometric Regularization**: Add subspace constraints to training")
        report.append("")
        
        # Recommendations
        report.append("### 💡 Recommendations")
        report.append("")
        report.append("1. **Subspace-Constrained Training**: Initialize features in active subspaces")
        report.append("2. **Dead Feature Monitoring**: Track dead features during training")
        report.append("3. **Layer-Aware Processing**: Account for subspace differences across layers")
        report.append("4. **Geometric Integration**: Combine with stratified manifold analysis")
        report.append("")
        
        return "\n".join(report)
    
    def save_results(self, filepath: str):
        """Save analysis results to file"""
        import json
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        converted_results = convert_numpy(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"💾 Results saved to {filepath}")

def run_wang_subspace_analysis(attention_outputs: np.ndarray, 
                              layer_names: List[str] = None) -> LowDimensionalSubspaceAnalyzer:
    """
    Run comprehensive subspace analysis based on Wang et al. (2025)
    """
    print("🚀 Starting Wang et al. Subspace Analysis")
    print("Based on 'Attention Layers Add Into Low-Dimensional Residual Subspaces'")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = LowDimensionalSubspaceAnalyzer(embedding_dim=attention_outputs.shape[1])
    
    # Run comprehensive analysis
    print("\n1. Analyzing attention subspaces...")
    analyzer.analyze_attention_subspaces(attention_outputs, layer_names)
    
    print("\n2. Creating visualizations...")
    figures = analyzer.create_subspace_visualizations(attention_outputs, layer_names)
    
    # Save visualizations
    for name, fig in figures.items():
        fig.savefig(f'results/images/wang_subspace_{name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print("\n3. Generating analysis report...")
    report = analyzer.generate_wang_analysis_report()
    
    # Save report
    with open('results/analysis/wang_subspace_analysis_report.md', 'w') as f:
        f.write(report)
    
    # Save results
    analyzer.save_results('results/data/wang_subspace_analysis.json')
    
    print("\n✅ Wang et al. Subspace Analysis Complete!")
    print(f"📊 Analysis report saved to: results/analysis/wang_subspace_analysis_report.md")
    print(f"📈 Visualizations saved to: results/images/wang_subspace_*.png")
    print(f"💾 Results saved to: results/data/wang_subspace_analysis.json")
    
    return analyzer
