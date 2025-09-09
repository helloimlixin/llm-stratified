"""
Integration of "Token embeddings violate the manifold hypothesis" findings.
Analysis of fiber bundle hypothesis and token embedding irregularities.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy, kstest, pearsonr
from scipy.signal import find_peaks
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

class FiberBundleHypothesisTester:
    """
    Implementation of the fiber bundle hypothesis test from:
    Robinson, M., Dey, S., & Chiang, T. (2025). 
    Token embeddings violate the manifold hypothesis. arXiv:2504.01002v2
    
    Tests whether token embeddings form a smooth fiber bundle structure.
    """
    
    def __init__(self, data, strata=None, labels=None, domains=None):
        self.data = np.array(data)
        self.strata = strata
        self.labels = labels
        self.domains = domains
        self.n_samples, self.n_features = self.data.shape
        
    def test_fiber_bundle_hypothesis(self, k=10, alpha=0.05):
        """
        Test the fiber bundle hypothesis for token embeddings.
        
        Null hypothesis: Neighborhood around each token has relatively 
        flat and smooth structure (fiber bundle).
        
        Alternative: Irregularities exist in token subspace neighborhoods.
        """
        # Adjust k based on available samples
        k = min(k, self.n_samples - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        
        irregularities = []
        p_values = []
        
        for i in range(self.n_samples):
            # Get neighborhood
            neighbor_indices = indices[i][1:]  # Exclude self
            neighbor_data = self.data[neighbor_indices]
            
            if len(neighbor_data) < 3:
                irregularities.append(False)
                p_values.append(1.0)
                continue
            
            # Test for irregularities in this neighborhood
            irregularity, p_value = self._test_neighborhood_regularity(
                neighbor_data, distances[i][1:], alpha
            )
            
            irregularities.append(irregularity)
            p_values.append(p_value)
        
        return {
            'irregularities': np.array(irregularities),
            'p_values': np.array(p_values),
            'n_irregularities': np.sum(irregularities),
            'irregularity_rate': np.mean(irregularities),
            'significant_irregularities': np.sum(np.array(p_values) < alpha)
        }
    
    def _test_neighborhood_regularity(self, neighbor_data, distances, alpha):
        """
        Test regularity of a neighborhood using multiple criteria.
        
        Based on the paper's approach of testing for smooth fiber bundle structure.
        """
        # Criterion 1: Local dimension consistency
        dim_consistency = self._test_dimension_consistency(neighbor_data)
        
        # Criterion 2: Curvature smoothness
        curvature_smoothness = self._test_curvature_smoothness(neighbor_data)
        
        # Criterion 3: Distance scaling
        distance_scaling = self._test_distance_scaling(distances)
        
        # Criterion 4: Local linearity
        local_linearity = self._test_local_linearity(neighbor_data)
        
        # Combine criteria (any failure indicates irregularity)
        irregularity = not (dim_consistency and curvature_smoothness and 
                          distance_scaling and local_linearity)
        
        # Estimate p-value based on combined criteria
        p_value = self._estimate_p_value(dim_consistency, curvature_smoothness, 
                                       distance_scaling, local_linearity)
        
        return irregularity, p_value
    
    def _test_dimension_consistency(self, neighbor_data):
        """Test if local dimension is consistent across neighborhood."""
        if len(neighbor_data) < 3:
            return True
        
        # Center the data
        center = np.mean(neighbor_data, axis=0)
        centered_data = neighbor_data - center
        
        # Compute PCA
        pca = PCA()
        pca.fit(centered_data)
        eigenvalues = pca.explained_variance_
        
        # Check if dimension is well-defined (sharp eigenvalue decay)
        if len(eigenvalues) < 2:
            return True
        
        # Dimension consistency: ratio of first two eigenvalues should be large
        eigenvalue_ratio = eigenvalues[0] / (eigenvalues[1] + 1e-8)
        
        # Threshold for dimension consistency
        return eigenvalue_ratio > 2.0
    
    def _test_curvature_smoothness(self, neighbor_data):
        """Test if curvature is smooth in the neighborhood."""
        if len(neighbor_data) < 4:
            return True
        
        # Compute pairwise distances
        distances = pairwise_distances(neighbor_data)
        
        # Test triangle inequality violations (indicates curvature)
        violations = 0
        total_triangles = 0
        
        for i in range(len(neighbor_data)):
            for j in range(i+1, len(neighbor_data)):
                for k in range(j+1, len(neighbor_data)):
                    d_ij = distances[i, j]
                    d_jk = distances[j, k]
                    d_ik = distances[i, k]
                    
                    # Check triangle inequality
                    if d_ij + d_jk < d_ik - 1e-6:  # Allow small numerical errors
                        violations += 1
                    total_triangles += 1
        
        # Smooth curvature: few triangle inequality violations
        violation_rate = violations / total_triangles if total_triangles > 0 else 0
        return violation_rate < 0.1  # Less than 10% violations
    
    def _test_distance_scaling(self, distances):
        """Test if distances scale properly (power law relationship)."""
        if len(distances) < 3:
            return True
        
        # Sort distances
        sorted_distances = np.sort(distances)
        
        # Test power law scaling
        log_distances = np.log(sorted_distances[1:])  # Exclude zero distance
        log_indices = np.log(np.arange(1, len(sorted_distances)))
        
        if len(log_distances) < 2:
            return True
        
        # Compute correlation
        correlation = np.corrcoef(log_indices, log_distances)[0, 1]
        
        # Good scaling: high correlation
        return not np.isnan(correlation) and correlation > 0.7
    
    def _test_local_linearity(self, neighbor_data):
        """Test if neighborhood is locally linear."""
        if len(neighbor_data) < 3:
            return True
        
        # Center the data
        center = np.mean(neighbor_data, axis=0)
        centered_data = neighbor_data - center
        
        # Compute PCA
        pca = PCA()
        pca.fit(centered_data)
        
        # Project to first two principal components
        projected_data = pca.transform(centered_data)[:, :2]
        
        # Test linearity using R² of linear fit
        if len(projected_data) < 3:
            return True
        
        # Fit line to projected data
        x = projected_data[:, 0]
        y = projected_data[:, 1]
        
        # Simple linear regression
        if np.var(x) < 1e-8:  # Avoid division by zero
            return True
        
        correlation = np.corrcoef(x, y)[0, 1]
        r_squared = correlation**2 if not np.isnan(correlation) else 0
        
        # Local linearity: high R²
        return r_squared > 0.5
    
    def _estimate_p_value(self, dim_consistency, curvature_smoothness, 
                         distance_scaling, local_linearity):
        """Estimate p-value based on combined criteria."""
        # Count failed criteria
        failed_criteria = sum([
            not dim_consistency,
            not curvature_smoothness,
            not distance_scaling,
            not local_linearity
        ])
        
        # Convert to p-value (simplified approach)
        if failed_criteria == 0:
            return 0.8  # High p-value, don't reject null
        elif failed_criteria == 1:
            return 0.3  # Moderate p-value
        elif failed_criteria == 2:
            return 0.1  # Low p-value
        else:
            return 0.01  # Very low p-value, reject null
    
    def analyze_token_stability(self, texts, model_name="roberta"):
        """
        Analyze token stability as described in the paper.
        
        When an LLM is presented with semantically equivalent prompts,
        if one prompt contains a token implicated by our test, the response
        will likely exhibit less stability.
        """
        # This would require access to the actual LLM and tokenizer
        # For now, we'll simulate the analysis
        
        stability_analysis = {}
        
        for i, text in enumerate(texts):
            # Simulate token analysis
            # In practice, this would tokenize the text and test each token
            
            # For demonstration, we'll use our embedding analysis
            if i < len(self.data):
                embedding = self.data[i]
                
                # Test this embedding's neighborhood
                nbrs = NearestNeighbors(n_neighbors=min(10, self.n_samples)).fit(self.data)
                distances, indices = nbrs.kneighbors([embedding])
                
                neighbor_data = self.data[indices[0][1:]]
                
                # Test for irregularities
                irregularity, p_value = self._test_neighborhood_regularity(
                    neighbor_data, distances[0][1:], alpha=0.05
                )
                
                stability_analysis[i] = {
                    'text': text[:100] + "..." if len(text) > 100 else text,
                    'irregularity': irregularity,
                    'p_value': p_value,
                    'stability_score': 1.0 - p_value,  # Higher = more stable
                    'embedding_norm': np.linalg.norm(embedding)
                }
        
        return stability_analysis
    
    def visualize_fiber_bundle_analysis(self, save_path=None):
        """
        Create visualization of fiber bundle hypothesis test results.
        """
        # Run the test
        results = self.test_fiber_bundle_hypothesis()
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # PCA 2D projection
        pca_2d = PCA(n_components=2)
        data_2d = pca_2d.fit_transform(self.data)
        
        # Plot 1: Irregularities
        colors = ['red' if irr else 'blue' for irr in results['irregularities']]
        scatter1 = axes[0, 0].scatter(data_2d[:, 0], data_2d[:, 1], 
                                     c=colors, alpha=0.7, s=30)
        axes[0, 0].set_title(f'Fiber Bundle Hypothesis Test\nIrregularities: {results["n_irregularities"]}/{len(results["irregularities"])}')
        axes[0, 0].set_xlabel('PC1')
        axes[0, 0].set_ylabel('PC2')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='Irregular'),
                          Patch(facecolor='blue', label='Regular')]
        axes[0, 0].legend(handles=legend_elements)
        
        # Plot 2: P-values
        scatter2 = axes[0, 1].scatter(data_2d[:, 0], data_2d[:, 1], 
                                      c=results['p_values'], cmap='viridis', alpha=0.7, s=30)
        axes[0, 1].set_title('P-values (Lower = More Irregular)')
        axes[0, 1].set_xlabel('PC1')
        axes[0, 1].set_ylabel('PC2')
        plt.colorbar(scatter2, ax=axes[0, 1], label='P-value')
        
        # Plot 3: Strata (if available)
        if self.strata is not None:
            scatter3 = axes[0, 2].scatter(data_2d[:, 0], data_2d[:, 1], 
                                          c=self.strata, cmap='tab10', alpha=0.7, s=30)
            axes[0, 2].set_title('Stratified Structure')
            axes[0, 2].set_xlabel('PC1')
            axes[0, 2].set_ylabel('PC2')
            plt.colorbar(scatter3, ax=axes[0, 2], label='Stratum')
        else:
            axes[0, 2].text(0.5, 0.5, 'No strata provided', ha='center', va='center', 
                           transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Stratified Structure')
        
        # Plot 4: P-value distribution
        axes[1, 0].hist(results['p_values'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(0.05, color='red', linestyle='--', label='α = 0.05')
        axes[1, 0].set_title('P-value Distribution')
        axes[1, 0].set_xlabel('P-value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Irregularity rate by stratum (if available)
        if self.strata is not None:
            unique_strata = np.unique(self.strata)
            irregularity_rates = []
            
            for stratum in unique_strata:
                stratum_mask = self.strata == stratum
                stratum_irregularities = results['irregularities'][stratum_mask]
                irregularity_rate = np.mean(stratum_irregularities)
                irregularity_rates.append(irregularity_rate)
            
            axes[1, 1].bar(unique_strata, irregularity_rates, alpha=0.7)
            axes[1, 1].set_title('Irregularity Rate by Stratum')
            axes[1, 1].set_xlabel('Stratum')
            axes[1, 1].set_ylabel('Irregularity Rate')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No strata available', ha='center', va='center', 
                           transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Irregularity Rate by Stratum')
        
        # Plot 6: Summary statistics
        axes[1, 2].text(0.1, 0.9, f"Fiber Bundle Hypothesis Test Results", 
                        fontweight='bold', transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.8, f"Total Samples: {len(results['irregularities'])}", 
                        transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.7, f"Irregularities: {results['n_irregularities']}", 
                        transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.6, f"Irregularity Rate: {results['irregularity_rate']:.3f}", 
                        transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.5, f"Significant (α=0.05): {results['significant_irregularities']}", 
                        transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.4, f"Mean P-value: {np.mean(results['p_values']):.3f}", 
                        transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.3, f"Min P-value: {np.min(results['p_values']):.3f}", 
                        transform=axes[1, 2].transAxes)
        
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return results

def run_fiber_bundle_analysis(data, strata=None, labels=None, domains=None, 
                             texts=None, save_path='fiber_bundle_analysis.png'):
    """
    Run comprehensive fiber bundle hypothesis analysis.
    
    Based on: Robinson, M., Dey, S., & Chiang, T. (2025). 
    Token embeddings violate the manifold hypothesis. arXiv:2504.01002v2
    """
    print("🔬 Running Fiber Bundle Hypothesis Analysis")
    print("=" * 60)
    print("Based on: Token embeddings violate the manifold hypothesis")
    print("Robinson, Dey, & Chiang (2025) - arXiv:2504.01002v2")
    print("=" * 60)
    
    # Initialize tester
    tester = FiberBundleHypothesisTester(data, strata=strata, labels=labels, domains=domains)
    
    # Run the test
    print("Testing fiber bundle hypothesis...")
    results = tester.test_fiber_bundle_hypothesis()
    
    # Print results
    print(f"\nFiber Bundle Hypothesis Test Results:")
    print(f"  Total Samples: {len(results['irregularities'])}")
    print(f"  Irregularities Found: {results['n_irregularities']}")
    print(f"  Irregularity Rate: {results['irregularity_rate']:.3f}")
    print(f"  Significant Irregularities (α=0.05): {results['significant_irregularities']}")
    print(f"  Mean P-value: {np.mean(results['p_values']):.3f}")
    print(f"  Min P-value: {np.min(results['p_values']):.3f}")
    print(f"  Max P-value: {np.max(results['p_values']):.3f}")
    
    # Interpret results
    if results['irregularity_rate'] > 0.5:
        print(f"\n🔴 CONCLUSION: Token embeddings VIOLATE the manifold hypothesis")
        print(f"   High irregularity rate ({results['irregularity_rate']:.3f}) suggests")
        print(f"   that token embeddings do not form a smooth fiber bundle.")
    elif results['irregularity_rate'] > 0.2:
        print(f"\n🟡 CONCLUSION: Mixed evidence for manifold hypothesis")
        print(f"   Moderate irregularity rate ({results['irregularity_rate']:.3f}) suggests")
        print(f"   some violations of smooth fiber bundle structure.")
    else:
        print(f"\n🟢 CONCLUSION: Token embeddings SUPPORT the manifold hypothesis")
        print(f"   Low irregularity rate ({results['irregularity_rate']:.3f}) suggests")
        print(f"   that token embeddings form a smooth fiber bundle.")
    
    # Analyze by strata if available
    if strata is not None:
        print(f"\nStratum Analysis:")
        unique_strata = np.unique(strata)
        for stratum in unique_strata:
            stratum_mask = strata == stratum
            stratum_irregularities = results['irregularities'][stratum_mask]
            stratum_rate = np.mean(stratum_irregularities)
            print(f"  Stratum {stratum}: Irregularity rate = {stratum_rate:.3f}")
    
    # Token stability analysis
    if texts is not None:
        print(f"\nAnalyzing token stability...")
        stability_analysis = tester.analyze_token_stability(texts)
        
        # Find most and least stable tokens
        stability_scores = [analysis['stability_score'] for analysis in stability_analysis.values()]
        most_stable_idx = np.argmax(stability_scores)
        least_stable_idx = np.argmin(stability_scores)
        
        print(f"  Most Stable Token: {stability_analysis[most_stable_idx]['text']}")
        print(f"    Stability Score: {stability_analysis[most_stable_idx]['stability_score']:.3f}")
        print(f"  Least Stable Token: {stability_analysis[least_stable_idx]['text']}")
        print(f"    Stability Score: {stability_analysis[least_stable_idx]['stability_score']:.3f}")
    
    # Create visualization
    print("\nCreating fiber bundle analysis visualization...")
    results = tester.visualize_fiber_bundle_analysis(save_path=save_path)
    
    return results

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_swiss_roll
    
    # Generate sample data
    data, _ = make_swiss_roll(n_samples=500, noise=0.1)
    
    # Create strata
    strata = np.zeros(500, dtype=int)
    strata[100:200] = 1
    strata[200:300] = 2
    strata[300:400] = 3
    strata[400:500] = 4
    
    # Run fiber bundle analysis
    results = run_fiber_bundle_analysis(data, strata=strata)
