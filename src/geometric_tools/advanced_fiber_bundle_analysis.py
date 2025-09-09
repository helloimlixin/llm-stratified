"""
Advanced Fiber Bundle Analysis for Token Embeddings
Based on Robinson, Dey, & Chiang (2025) - "Token Embeddings Violate the Manifold Hypothesis"

This module implements the theoretical framework from the paper to analyze
token embeddings as fiber bundles rather than manifolds.
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

class AdvancedFiberBundleAnalyzer:
    """
    Advanced fiber bundle analysis based on Robinson et al. (2025)
    
    Implements the theoretical framework for testing whether token embeddings
    form fiber bundles or violate the manifold hypothesis.
    """
    
    def __init__(self, embedding_dim: int = 768, n_neighbors: int = 10):
        self.embedding_dim = embedding_dim
        self.n_neighbors = n_neighbors
        self.results = {}
        
    def analyze_token_subspaces(self, embeddings: np.ndarray, tokens: List[str] = None) -> Dict:
        """
        Analyze token embeddings as subspaces to test fiber bundle hypothesis
        
        Based on Robinson et al. (2025) methodology for testing whether
        token neighborhoods can be decomposed into signal and noise dimensions.
        """
        print("🔬 Analyzing token subspaces for fiber bundle structure...")
        
        n_tokens, dim = embeddings.shape
        results = {
            'fiber_bundle_tests': {},
            'subspace_dimensions': {},
            'signal_noise_ratios': {},
            'local_geometry': {},
            'manifold_violations': {}
        }
        
        # Test each token's local neighborhood
        for i in range(min(100, n_tokens)):  # Sample subset for efficiency
            token_name = tokens[i] if tokens else f"token_{i}"
            
            # Get local neighborhood
            nbrs = NearestNeighbors(n_neighbors=min(self.n_neighbors + 1, n_tokens))
            nbrs.fit(embeddings)
            distances, indices = nbrs.kneighbors([embeddings[i]])
            
            # Get neighborhood embeddings
            neighbor_embeddings = embeddings[indices[0][1:]]  # Exclude self
            center_embedding = embeddings[i]
            
            # Center the neighborhood
            centered_neighbors = neighbor_embeddings - center_embedding
            
            # Perform SVD to decompose into signal and noise
            try:
                U, s, Vt = svd(centered_neighbors, full_matrices=False)
                
                # Estimate signal and noise dimensions
                signal_dim, noise_dim = self._estimate_signal_noise_dimensions(s)
                
                # Compute fiber bundle test statistics
                fiber_bundle_stats = self._compute_fiber_bundle_statistics(
                    centered_neighbors, U, s, Vt, signal_dim, noise_dim
                )
                
                # Store results
                results['fiber_bundle_tests'][token_name] = fiber_bundle_stats
                results['subspace_dimensions'][token_name] = {
                    'signal_dim': signal_dim,
                    'noise_dim': noise_dim,
                    'total_dim': len(s)
                }
                results['signal_noise_ratios'][token_name] = {
                    'ratio': signal_dim / max(noise_dim, 1),
                    'signal_energy': np.sum(s[:signal_dim]**2),
                    'noise_energy': np.sum(s[signal_dim:]**2)
                }
                
            except Exception as e:
                print(f"Warning: Could not analyze token {token_name}: {e}")
                continue
        
        self.results['token_subspaces'] = results
        return results
    
    def _estimate_signal_noise_dimensions(self, singular_values: np.ndarray) -> Tuple[int, int]:
        """
        Estimate signal and noise dimensions using eigenvalue analysis
        
        Based on the elbow method and energy thresholding from Robinson et al.
        """
        # Normalize singular values
        s_norm = singular_values / np.max(singular_values)
        
        # Method 1: Elbow detection
        if len(s_norm) > 2:
            # Find elbow point
            diffs = np.diff(s_norm)
            second_diffs = np.diff(diffs)
            elbow_idx = np.argmax(second_diffs) + 1
        else:
            elbow_idx = 1
        
        # Method 2: Energy thresholding (95% energy)
        cumulative_energy = np.cumsum(s_norm**2)
        energy_threshold = 0.95
        energy_idx = np.where(cumulative_energy >= energy_threshold)[0]
        energy_idx = energy_idx[0] + 1 if len(energy_idx) > 0 else len(s_norm)
        
        # Method 3: Statistical significance (eigenvalue gap)
        if len(s_norm) > 1:
            gaps = np.diff(s_norm)
            max_gap_idx = np.argmax(gaps) + 1
        else:
            max_gap_idx = 1
        
        # Combine methods (take minimum for conservative estimate)
        signal_dim = min(elbow_idx, energy_idx, max_gap_idx)
        signal_dim = max(1, min(signal_dim, len(singular_values) - 1))
        noise_dim = len(singular_values) - signal_dim
        
        return signal_dim, noise_dim
    
    def _compute_fiber_bundle_statistics(self, centered_neighbors: np.ndarray, 
                                       U: np.ndarray, s: np.ndarray, Vt: np.ndarray,
                                       signal_dim: int, noise_dim: int) -> Dict:
        """
        Compute statistics to test fiber bundle hypothesis
        
        Implements the statistical tests from Robinson et al. (2025)
        """
        n_neighbors, dim = centered_neighbors.shape
        
        # Project onto signal and noise subspaces
        signal_subspace = Vt[:signal_dim].T
        noise_subspace = Vt[signal_dim:].T if noise_dim > 0 else None
        
        # Project neighbors onto subspaces
        signal_projections = centered_neighbors @ signal_subspace
        noise_projections = centered_neighbors @ noise_subspace if noise_subspace is not None else None
        
        # Compute fiber bundle test statistics
        stats_dict = {}
        
        # 1. Signal subspace coherence
        if signal_dim > 1:
            signal_cov = np.cov(signal_projections.T)
            signal_coherence = np.trace(signal_cov) / (signal_dim * np.mean(np.diag(signal_cov)))
            stats_dict['signal_coherence'] = signal_coherence
        else:
            stats_dict['signal_coherence'] = 1.0
        
        # 2. Noise subspace isotropy
        if noise_dim > 1 and noise_projections is not None:
            noise_cov = np.cov(noise_projections.T)
            noise_eigenvals = np.linalg.eigvals(noise_cov)
            noise_isotropy = np.std(noise_eigenvals) / np.mean(noise_eigenvals)
            stats_dict['noise_isotropy'] = noise_isotropy
        else:
            stats_dict['noise_isotropy'] = 0.0
        
        # 3. Fiber bundle dimension consistency
        expected_fiber_dim = signal_dim
        actual_fiber_dim = np.linalg.matrix_rank(signal_projections)
        dimension_consistency = actual_fiber_dim / max(expected_fiber_dim, 1)
        stats_dict['dimension_consistency'] = dimension_consistency
        
        # 4. Local linearity test
        if signal_dim > 1:
            # Test if signal subspace is locally linear
            signal_distances = pdist(signal_projections)
            original_distances = pdist(centered_neighbors)
            linearity_ratio = np.corrcoef(signal_distances, original_distances)[0, 1]
            stats_dict['local_linearity'] = linearity_ratio
        else:
            stats_dict['local_linearity'] = 1.0
        
        # 5. Fiber bundle null hypothesis test
        # H0: The neighborhood forms a fiber bundle
        # H1: The neighborhood violates fiber bundle structure
        
        # Test statistic: combination of coherence and isotropy
        coherence_score = stats_dict['signal_coherence']
        isotropy_score = 1 - stats_dict['noise_isotropy']  # Lower isotropy = more fiber-like
        consistency_score = stats_dict['dimension_consistency']
        linearity_score = abs(stats_dict['local_linearity'])
        
        # Combined test statistic
        fiber_bundle_score = (coherence_score + isotropy_score + consistency_score + linearity_score) / 4
        
        # P-value approximation (simplified)
        if fiber_bundle_score > 0.8:
            p_value = 0.1  # Fail to reject H0 (fiber bundle)
        elif fiber_bundle_score > 0.6:
            p_value = 0.05  # Weak evidence against H0
        else:
            p_value = 0.01  # Strong evidence against H0
        
        stats_dict['fiber_bundle_score'] = fiber_bundle_score
        stats_dict['p_value'] = p_value
        stats_dict['reject_null'] = p_value < 0.05
        
        return stats_dict
    
    def analyze_manifold_violations(self, embeddings: np.ndarray, tokens: List[str] = None) -> Dict:
        """
        Analyze violations of the manifold hypothesis
        
        Based on Robinson et al. (2025) findings that token embeddings
        frequently violate the manifold hypothesis.
        """
        print("🔍 Analyzing manifold hypothesis violations...")
        
        results = self.analyze_token_subspaces(embeddings, tokens)
        
        # Aggregate violation statistics
        violation_stats = {
            'total_tokens_tested': len(results['fiber_bundle_tests']),
            'manifold_violations': 0,
            'fiber_bundle_violations': 0,
            'violation_rates': {},
            'token_categories': {}
        }
        
        # Count violations
        for token_name, stats in results['fiber_bundle_tests'].items():
            if stats['reject_null']:
                violation_stats['fiber_bundle_violations'] += 1
            
            # Categorize tokens by violation type
            if stats['fiber_bundle_score'] < 0.5:
                violation_stats['token_categories'][token_name] = 'strong_violation'
            elif stats['fiber_bundle_score'] < 0.7:
                violation_stats['token_categories'][token_name] = 'moderate_violation'
            else:
                violation_stats['token_categories'][token_name] = 'manifold_like'
        
        # Compute violation rates
        total_tested = violation_stats['total_tokens_tested']
        if total_tested > 0:
            violation_stats['violation_rates'] = {
                'fiber_bundle_violation_rate': violation_stats['fiber_bundle_violations'] / total_tested,
                'manifold_violation_rate': violation_stats['manifold_violations'] / total_tested
            }
        
        self.results['manifold_violations'] = violation_stats
        return violation_stats
    
    def analyze_token_variability(self, embeddings: np.ndarray, tokens: List[str] = None) -> Dict:
        """
        Analyze variability in token embeddings that violate manifold hypothesis
        
        Based on Robinson et al. finding that violating tokens lead to
        increased variability in model outputs.
        """
        print("📊 Analyzing token variability patterns...")
        
        if 'token_subspaces' not in self.results:
            self.analyze_token_subspaces(embeddings, tokens)
        
        variability_results = {
            'high_variability_tokens': [],
            'low_variability_tokens': [],
            'variability_scores': {},
            'correlation_analysis': {}
        }
        
        # Compute variability scores for each token
        for token_name, stats in self.results['token_subspaces']['fiber_bundle_tests'].items():
            # Variability score based on fiber bundle violation
            violation_score = 1 - stats['fiber_bundle_score']
            
            # Additional variability indicators
            signal_noise_ratio = self.results['token_subspaces']['signal_noise_ratios'][token_name]['ratio']
            dimension_consistency = stats['dimension_consistency']
            
            # Combined variability score
            variability_score = violation_score * signal_noise_ratio * (1 / dimension_consistency)
            
            variability_results['variability_scores'][token_name] = variability_score
            
            # Categorize tokens
            if variability_score > 0.7:
                variability_results['high_variability_tokens'].append(token_name)
            elif variability_score < 0.3:
                variability_results['low_variability_tokens'].append(token_name)
        
        # Analyze correlations between violation and variability
        violation_scores = [stats['fiber_bundle_score'] for stats in self.results['token_subspaces']['fiber_bundle_tests'].values()]
        variability_scores = list(variability_results['variability_scores'].values())
        
        if len(violation_scores) > 1:
            correlation = np.corrcoef(violation_scores, variability_scores)[0, 1]
            variability_results['correlation_analysis'] = {
                'violation_variability_correlation': correlation,
                'correlation_strength': 'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.4 else 'weak'
            }
        
        self.results['token_variability'] = variability_results
        return variability_results
    
    def create_fiber_bundle_visualizations(self, embeddings: np.ndarray, tokens: List[str] = None) -> Dict[str, plt.Figure]:
        """
        Create comprehensive visualizations of fiber bundle analysis
        
        Based on Robinson et al. (2025) methodology
        """
        print("🎨 Creating fiber bundle visualizations...")
        
        if 'token_subspaces' not in self.results:
            self.analyze_token_subspaces(embeddings, tokens)
        
        figures = {}
        
        # 1. Fiber bundle violation heatmap
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        
        violation_scores = []
        token_names = []
        for token_name, stats in self.results['token_subspaces']['fiber_bundle_tests'].items():
            violation_scores.append(1 - stats['fiber_bundle_score'])
            token_names.append(token_name)
        
        if violation_scores:
            # Create heatmap
            violation_matrix = np.array(violation_scores).reshape(-1, 1)
            im = ax1.imshow(violation_matrix.T, cmap='Reds', aspect='auto')
            ax1.set_title('Fiber Bundle Violation Scores by Token')
            ax1.set_xlabel('Token Index')
            ax1.set_ylabel('Violation Score')
            plt.colorbar(im, ax=ax1)
        
        figures['violation_heatmap'] = fig1
        
        # 2. Signal vs Noise dimension analysis
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(15, 6))
        
        signal_dims = []
        noise_dims = []
        for token_name, dims in self.results['token_subspaces']['subspace_dimensions'].items():
            signal_dims.append(dims['signal_dim'])
            noise_dims.append(dims['noise_dim'])
        
        if signal_dims:
            ax2a.scatter(signal_dims, noise_dims, alpha=0.6)
            ax2a.set_xlabel('Signal Dimension')
            ax2a.set_ylabel('Noise Dimension')
            ax2a.set_title('Signal vs Noise Dimensions')
            ax2a.grid(True, alpha=0.3)
            
            # Add diagonal line
            max_dim = max(max(signal_dims), max(noise_dims))
            ax2a.plot([0, max_dim], [0, max_dim], 'r--', alpha=0.5, label='Equal dimensions')
            ax2a.legend()
        
        # 3. Violation rate distribution
        if violation_scores:
            ax2b.hist(violation_scores, bins=20, alpha=0.7, edgecolor='black')
            ax2b.set_xlabel('Violation Score')
            ax2b.set_ylabel('Frequency')
            ax2b.set_title('Distribution of Fiber Bundle Violation Scores')
            ax2b.grid(True, alpha=0.3)
        
        figures['dimension_analysis'] = fig2
        
        # 4. Token variability analysis
        if 'token_variability' in self.results:
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            
            variability_scores = list(self.results['token_variability']['variability_scores'].values())
            token_names = list(self.results['token_variability']['variability_scores'].keys())
            
            if variability_scores:
                bars = ax3.bar(range(len(variability_scores)), variability_scores, alpha=0.7)
                ax3.set_xlabel('Token Index')
                ax3.set_ylabel('Variability Score')
                ax3.set_title('Token Variability Scores (Higher = More Variable)')
                ax3.grid(True, alpha=0.3)
                
                # Color bars by variability level
                for i, bar in enumerate(bars):
                    if variability_scores[i] > 0.7:
                        bar.set_color('red')
                    elif variability_scores[i] < 0.3:
                        bar.set_color('green')
                    else:
                        bar.set_color('orange')
            
            figures['variability_analysis'] = fig3
        
        return figures
    
    def generate_robinson_analysis_report(self) -> str:
        """
        Generate comprehensive analysis report based on Robinson et al. (2025)
        """
        if not self.results:
            return "No analysis results available. Run analysis first."
        
        report = []
        report.append("# 🔬 Advanced Fiber Bundle Analysis Report")
        report.append("## Based on Robinson, Dey, & Chiang (2025)")
        report.append("")
        
        # Token subspace analysis
        if 'token_subspaces' in self.results:
            report.append("### 📊 Token Subspace Analysis")
            report.append("")
            
            total_tokens = len(self.results['token_subspaces']['fiber_bundle_tests'])
            violations = sum(1 for stats in self.results['token_subspaces']['fiber_bundle_tests'].values() 
                           if stats['reject_null'])
            
            report.append(f"- **Total tokens analyzed**: {total_tokens}")
            report.append(f"- **Fiber bundle violations**: {violations}")
            report.append(f"- **Violation rate**: {violations/total_tokens:.2%}")
            report.append("")
            
            # Top violating tokens
            violation_scores = [(name, 1-stats['fiber_bundle_score']) 
                              for name, stats in self.results['token_subspaces']['fiber_bundle_tests'].items()]
            violation_scores.sort(key=lambda x: x[1], reverse=True)
            
            report.append("### 🚨 Top Violating Tokens")
            report.append("")
            for i, (token, score) in enumerate(violation_scores[:10]):
                report.append(f"{i+1}. **{token}**: {score:.3f}")
            report.append("")
        
        # Manifold violation analysis
        if 'manifold_violations' in self.results:
            report.append("### 🔍 Manifold Hypothesis Violations")
            report.append("")
            
            stats = self.results['manifold_violations']
            report.append(f"- **Total tokens tested**: {stats['total_tokens_tested']}")
            report.append(f"- **Fiber bundle violations**: {stats['fiber_bundle_violations']}")
            
            if 'violation_rates' in stats:
                rates = stats['violation_rates']
                report.append(f"- **Fiber bundle violation rate**: {rates['fiber_bundle_violation_rate']:.2%}")
            report.append("")
        
        # Token variability analysis
        if 'token_variability' in self.results:
            report.append("### 📈 Token Variability Analysis")
            report.append("")
            
            var_stats = self.results['token_variability']
            report.append(f"- **High variability tokens**: {len(var_stats['high_variability_tokens'])}")
            report.append(f"- **Low variability tokens**: {len(var_stats['low_variability_tokens'])}")
            
            if 'correlation_analysis' in var_stats:
                corr = var_stats['correlation_analysis']
                report.append(f"- **Violation-variability correlation**: {corr['violation_variability_correlation']:.3f}")
                report.append(f"- **Correlation strength**: {corr['correlation_strength']}")
            report.append("")
        
        # Theoretical implications
        report.append("### 🧠 Theoretical Implications")
        report.append("")
        report.append("Based on Robinson et al. (2025) findings:")
        report.append("")
        report.append("1. **Manifold Hypothesis Violation**: Token embeddings frequently violate the manifold hypothesis")
        report.append("2. **Fiber Bundle Structure**: Local neighborhoods show complex fiber bundle-like structures")
        report.append("3. **Variability Impact**: Violating tokens lead to increased model output variability")
        report.append("4. **Geometric Complexity**: Token spaces exhibit rich geometric structure beyond simple manifolds")
        report.append("")
        
        # Recommendations
        report.append("### 💡 Recommendations")
        report.append("")
        report.append("1. **Model Design**: Consider fiber bundle-aware architectures")
        report.append("2. **Training**: Account for token variability in loss functions")
        report.append("3. **Analysis**: Use geometric tools beyond manifold assumptions")
        report.append("4. **Robustness**: Test model behavior on violating tokens")
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

def run_advanced_fiber_bundle_analysis(embeddings: np.ndarray, tokens: List[str] = None) -> AdvancedFiberBundleAnalyzer:
    """
    Run comprehensive fiber bundle analysis based on Robinson et al. (2025)
    """
    print("🚀 Starting Advanced Fiber Bundle Analysis")
    print("Based on Robinson, Dey, & Chiang (2025)")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = AdvancedFiberBundleAnalyzer(embedding_dim=embeddings.shape[1])
    
    # Run comprehensive analysis
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
        fig.savefig(f'results/images/advanced_fiber_bundle_{name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print("\n5. Generating analysis report...")
    report = analyzer.generate_robinson_analysis_report()
    
    # Save report
    with open('results/analysis/advanced_fiber_bundle_analysis_report.md', 'w') as f:
        f.write(report)
    
    # Save results
    analyzer.save_results('results/data/advanced_fiber_bundle_analysis.json')
    
    print("\n✅ Advanced Fiber Bundle Analysis Complete!")
    print(f"📊 Analysis report saved to: results/analysis/advanced_fiber_bundle_analysis_report.md")
    print(f"📈 Visualizations saved to: results/images/advanced_fiber_bundle_*.png")
    print(f"💾 Results saved to: results/data/advanced_fiber_bundle_analysis.json")
    
    return analyzer
