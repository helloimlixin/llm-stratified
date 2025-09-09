"""
Curvature discontinuity analysis for stratified manifold hypothesis.
Tests the claim that there should be abrupt changes in curvature between different strata.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy, kstest
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

class CurvatureDiscontinuityAnalyzer:
    """
    Analyzes curvature discontinuities between stratified manifolds.
    
    Tests the stratified manifold hypothesis that there should be
    abrupt changes in curvature at stratum boundaries.
    """
    
    def __init__(self, data, strata, labels=None, domains=None):
        self.data = np.array(data)
        self.strata = np.array(strata)
        self.labels = labels
        self.domains = domains
        self.n_samples, self.n_features = self.data.shape
        self.unique_strata = np.unique(strata)
        
    def compute_stratum_boundary_curvature(self, k=10):
        """
        Compute curvature at stratum boundaries to detect discontinuities.
        
        Returns curvature values for points near stratum boundaries.
        """
        # Adjust k based on available samples
        k = min(k, self.n_samples - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        
        boundary_curvatures = []
        boundary_points = []
        
        for i in range(self.n_samples):
            neighbor_indices = indices[i][1:]  # Exclude self
            neighbor_strata = self.strata[neighbor_indices]
            current_stratum = self.strata[i]
            
            # Check if this point is near a stratum boundary
            cross_stratum_neighbors = np.sum(neighbor_strata != current_stratum)
            cross_stratum_ratio = cross_stratum_neighbors / len(neighbor_strata)
            
            if cross_stratum_ratio > 0.3:  # Point is near boundary
                # Compute local curvature
                neighbor_data = self.data[neighbor_indices]
                curvature = self._compute_local_riemannian_curvature(neighbor_data)
                
                boundary_curvatures.append(curvature)
                boundary_points.append(i)
        
        return np.array(boundary_curvatures), np.array(boundary_points)
    
    def _compute_local_riemannian_curvature(self, local_data):
        """Compute local Riemannian curvature from neighborhood data."""
        if len(local_data) < 3:
            return 0.0
        
        # Center the data
        center = np.mean(local_data, axis=0)
        centered_data = local_data - center
        
        # Compute PCA
        pca = PCA()
        pca.fit(centered_data)
        eigenvalues = pca.explained_variance_
        
        if len(eigenvalues) < 2 or eigenvalues[0] == 0:
            return 0.0
        
        # Curvature estimate based on eigenvalue ratios
        curvature = eigenvalues[1] / eigenvalues[0] if eigenvalues[0] > 0 else 0
        
        return curvature
    
    def compute_stratum_interior_curvature(self, k=10):
        """
        Compute curvature for points in stratum interiors (away from boundaries).
        """
        # Adjust k based on available samples
        k = min(k, self.n_samples - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        
        interior_curvatures = []
        interior_points = []
        
        for i in range(self.n_samples):
            neighbor_indices = indices[i][1:]  # Exclude self
            neighbor_strata = self.strata[neighbor_indices]
            current_stratum = self.strata[i]
            
            # Check if this point is in stratum interior
            same_stratum_neighbors = np.sum(neighbor_strata == current_stratum)
            same_stratum_ratio = same_stratum_neighbors / len(neighbor_strata)
            
            if same_stratum_ratio > 0.7:  # Point is in interior
                # Compute local curvature
                neighbor_data = self.data[neighbor_indices]
                curvature = self._compute_local_riemannian_curvature(neighbor_data)
                
                interior_curvatures.append(curvature)
                interior_points.append(i)
        
        return np.array(interior_curvatures), np.array(interior_points)
    
    def detect_curvature_discontinuities(self, k=10, threshold_percentile=90):
        """
        Detect abrupt changes in curvature between strata.
        
        Returns:
        - discontinuity_scores: Measure of curvature discontinuity per stratum pair
        - boundary_curvature_jumps: Actual curvature differences at boundaries
        """
        # Get boundary and interior curvatures
        boundary_curvatures, boundary_points = self.compute_stratum_boundary_curvature(k)
        interior_curvatures, interior_points = self.compute_stratum_interior_curvature(k)
        
        if len(boundary_curvatures) == 0 or len(interior_curvatures) == 0:
            return {}, {}
        
        # Compute discontinuity scores between stratum pairs
        discontinuity_scores = {}
        boundary_curvature_jumps = {}
        
        for i, stratum1 in enumerate(self.unique_strata):
            for j, stratum2 in enumerate(self.unique_strata):
                if i >= j:  # Avoid duplicates and self-comparison
                    continue
                
                # Get boundary points between these strata
                stratum1_boundary_mask = self.strata[boundary_points] == stratum1
                stratum2_boundary_mask = self.strata[boundary_points] == stratum2
                
                # Find points that are boundaries between stratum1 and stratum2
                cross_boundary_mask = np.zeros(len(boundary_points), dtype=bool)
                
                # Re-fit neighbors for boundary points
                nbrs_boundary = NearestNeighbors(n_neighbors=k+1).fit(self.data)
                
                for idx, point_idx in enumerate(boundary_points):
                    neighbor_indices = nbrs_boundary.kneighbors([self.data[point_idx]])[1][0][1:]
                    neighbor_strata = self.strata[neighbor_indices]
                    
                    has_stratum1_neighbors = np.any(neighbor_strata == stratum1)
                    has_stratum2_neighbors = np.any(neighbor_strata == stratum2)
                    
                    if has_stratum1_neighbors and has_stratum2_neighbors:
                        cross_boundary_mask[idx] = True
                
                if np.sum(cross_boundary_mask) < 2:
                    continue
                
                # Compute curvature differences at boundaries
                cross_boundary_curvatures = boundary_curvatures[cross_boundary_mask]
                
                # Get interior curvatures for both strata
                stratum1_interior_mask = self.strata[interior_points] == stratum1
                stratum2_interior_mask = self.strata[interior_points] == stratum2
                
                stratum1_interior_curv = interior_curvatures[stratum1_interior_mask]
                stratum2_interior_curv = interior_curvatures[stratum2_interior_mask]
                
                if len(stratum1_interior_curv) == 0 or len(stratum2_interior_curv) == 0:
                    continue
                
                # Compute discontinuity score
                interior_diff = abs(np.mean(stratum1_interior_curv) - np.mean(stratum2_interior_curv))
                boundary_variance = np.var(cross_boundary_curvatures)
                
                # Discontinuity score: large interior difference + high boundary variance
                discontinuity_score = interior_diff + boundary_variance
                
                discontinuity_scores[f"{stratum1}-{stratum2}"] = discontinuity_score
                boundary_curvature_jumps[f"{stratum1}-{stratum2}"] = {
                    'interior_diff': interior_diff,
                    'boundary_variance': boundary_variance,
                    'boundary_curvatures': cross_boundary_curvatures.tolist(),
                    'stratum1_interior_mean': np.mean(stratum1_interior_curv),
                    'stratum2_interior_mean': np.mean(stratum2_interior_curv)
                }
        
        return discontinuity_scores, boundary_curvature_jumps
    
    def analyze_curvature_gradients(self, k=10):
        """
        Analyze curvature gradients across stratum boundaries.
        
        Tests if curvature changes abruptly (high gradient) at boundaries.
        """
        # Adjust k based on available samples
        k = min(k, self.n_samples - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.data)
        
        gradients = []
        gradient_locations = []
        
        for i in range(self.n_samples):
            # Get neighbors
            distances, indices = nbrs.kneighbors([self.data[i]])
            neighbor_indices = indices[0][1:]  # Exclude self
            
            # Compute curvature at this point
            neighbor_data = self.data[neighbor_indices]
            current_curvature = self._compute_local_riemannian_curvature(neighbor_data)
            
            # Compute curvature at neighbors
            neighbor_curvatures = []
            for neighbor_idx in neighbor_indices:
                neighbor_neighbors = nbrs.kneighbors([self.data[neighbor_idx]])[1][0][1:]
                neighbor_neighbor_data = self.data[neighbor_neighbors]
                neighbor_curv = self._compute_local_riemannian_curvature(neighbor_neighbor_data)
                neighbor_curvatures.append(neighbor_curv)
            
            # Compute gradient (curvature change rate)
            if len(neighbor_curvatures) > 0:
                curvature_gradient = np.std(neighbor_curvatures)
                gradients.append(curvature_gradient)
                gradient_locations.append(i)
        
        return np.array(gradients), np.array(gradient_locations)
    
    def test_stratified_manifold_hypothesis(self, k=10, significance_level=0.05):
        """
        Test the stratified manifold hypothesis using statistical tests.
        
        Hypothesis: There should be abrupt changes in curvature between strata.
        """
        # Get boundary and interior curvatures
        boundary_curvatures, boundary_points = self.compute_stratum_boundary_curvature(k)
        interior_curvatures, interior_points = self.compute_stratum_interior_curvature(k)
        
        if len(boundary_curvatures) == 0 or len(interior_curvatures) == 0:
            return {
                'hypothesis_supported': False,
                'reason': 'Insufficient boundary or interior points',
                'statistics': {}
            }
        
        # Statistical test: Are boundary curvatures significantly different from interior?
        from scipy.stats import ttest_ind, mannwhitneyu
        
        # T-test
        t_stat, t_pvalue = ttest_ind(boundary_curvatures, interior_curvatures)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pvalue = mannwhitneyu(boundary_curvatures, interior_curvatures, 
                                       alternative='two-sided')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(boundary_curvatures) - 1) * np.var(boundary_curvatures) + 
                             (len(interior_curvatures) - 1) * np.var(interior_curvatures)) / 
                            (len(boundary_curvatures) + len(interior_curvatures) - 2))
        cohens_d = (np.mean(boundary_curvatures) - np.mean(interior_curvatures)) / pooled_std
        
        # Discontinuity detection
        discontinuity_scores, boundary_jumps = self.detect_curvature_discontinuities(k)
        
        # Gradient analysis
        gradients, gradient_locations = self.analyze_curvature_gradients(k)
        
        # Determine if hypothesis is supported
        hypothesis_supported = (t_pvalue < significance_level or u_pvalue < significance_level) and abs(cohens_d) > 0.5
        
        results = {
            'hypothesis_supported': hypothesis_supported,
            'significance_level': significance_level,
            'statistics': {
                'boundary_curvature_mean': float(np.mean(boundary_curvatures)),
                'interior_curvature_mean': float(np.mean(interior_curvatures)),
                'boundary_curvature_std': float(np.std(boundary_curvatures)),
                'interior_curvature_std': float(np.std(interior_curvatures)),
                't_statistic': float(t_stat),
                't_pvalue': float(t_pvalue),
                'u_statistic': float(u_stat),
                'u_pvalue': float(u_pvalue),
                'cohens_d': float(cohens_d),
                'n_boundary_points': len(boundary_curvatures),
                'n_interior_points': len(interior_curvatures)
            },
            'discontinuity_scores': discontinuity_scores,
            'boundary_jumps': boundary_jumps,
            'gradient_statistics': {
                'gradient_mean': float(np.mean(gradients)),
                'gradient_std': float(np.std(gradients)),
                'high_gradient_points': int(np.sum(gradients > np.percentile(gradients, 90)))
            }
        }
        
        return results
    
    def visualize_curvature_discontinuities(self, save_path=None):
        """
        Create comprehensive visualization of curvature discontinuities.
        """
        # Get boundary and interior curvatures
        boundary_curvatures, boundary_points = self.compute_stratum_boundary_curvature()
        interior_curvatures, interior_points = self.compute_stratum_interior_curvature()
        
        # Get gradients
        gradients, gradient_locations = self.analyze_curvature_gradients()
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Curvature by location type
        if len(boundary_curvatures) > 0 and len(interior_curvatures) > 0:
            data_to_plot = [interior_curvatures, boundary_curvatures]
            labels_to_plot = ['Interior', 'Boundary']
            
            axes[0, 0].boxplot(data_to_plot, labels=labels_to_plot)
            axes[0, 0].set_title('Curvature: Interior vs Boundary')
            axes[0, 0].set_ylabel('Riemannian Curvature')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Curvature distribution by stratum
        stratum_curvatures = {}
        for stratum in self.unique_strata:
            stratum_mask = self.strata == stratum
            stratum_data = self.data[stratum_mask]
            if len(stratum_data) > 3:
                analyzer = CurvatureDiscontinuityAnalyzer(stratum_data, 
                                                         self.strata[stratum_mask])
                curvatures, _ = analyzer.compute_stratum_interior_curvature()
                if len(curvatures) > 0:
                    stratum_curvatures[stratum] = curvatures
        
        if stratum_curvatures:
            axes[0, 1].boxplot(list(stratum_curvatures.values()), 
                              labels=list(stratum_curvatures.keys()))
            axes[0, 1].set_title('Curvature Distribution by Stratum')
            axes[0, 1].set_ylabel('Riemannian Curvature')
            axes[0, 1].set_xlabel('Stratum')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Curvature gradients
        if len(gradients) > 0:
            axes[0, 2].hist(gradients, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 2].axvline(np.mean(gradients), color='red', linestyle='--', 
                             label=f'Mean: {np.mean(gradients):.3f}')
            axes[0, 2].set_title('Curvature Gradient Distribution')
            axes[0, 2].set_xlabel('Curvature Gradient')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: 2D projection colored by curvature
        pca_2d = PCA(n_components=2)
        data_2d = pca_2d.fit_transform(self.data)
        
        # Compute curvature for all points
        all_curvatures = []
        for i in range(self.n_samples):
            nbrs = NearestNeighbors(n_neighbors=min(10, self.n_samples)).fit(self.data)
            distances, indices = nbrs.kneighbors([self.data[i]])
            neighbor_data = self.data[indices[0][1:]]
            curvature = self._compute_local_riemannian_curvature(neighbor_data)
            all_curvatures.append(curvature)
        
        scatter = axes[1, 0].scatter(data_2d[:, 0], data_2d[:, 1], 
                                    c=all_curvatures, cmap='plasma', alpha=0.7, s=30)
        axes[1, 0].set_title('2D Projection: Colored by Curvature')
        axes[1, 0].set_xlabel('PC1')
        axes[1, 0].set_ylabel('PC2')
        plt.colorbar(scatter, ax=axes[1, 0], label='Curvature')
        
        # Plot 5: 2D projection colored by strata
        scatter2 = axes[1, 1].scatter(data_2d[:, 0], data_2d[:, 1], 
                                     c=self.strata, cmap='tab10', alpha=0.7, s=30)
        axes[1, 1].set_title('2D Projection: Colored by Strata')
        axes[1, 1].set_xlabel('PC1')
        axes[1, 1].set_ylabel('PC2')
        plt.colorbar(scatter2, ax=axes[1, 1], label='Stratum')
        
        # Plot 6: Boundary points highlighted
        if len(boundary_points) > 0:
            # Color all points
            colors = ['lightgray'] * self.n_samples
            # Highlight boundary points
            for bp in boundary_points:
                colors[bp] = 'red'
            
            scatter3 = axes[1, 2].scatter(data_2d[:, 0], data_2d[:, 1], 
                                         c=colors, alpha=0.7, s=30)
            axes[1, 2].set_title('2D Projection: Boundary Points (Red)')
            axes[1, 2].set_xlabel('PC1')
            axes[1, 2].set_ylabel('PC2')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def run_curvature_discontinuity_analysis(data, strata, labels=None, domains=None, 
                                       save_path='curvature_discontinuity_analysis.png'):
    """
    Run comprehensive curvature discontinuity analysis.
    
    Tests the stratified manifold hypothesis that there should be
    abrupt changes in curvature between different strata.
    """
    print("🔬 Testing Stratified Manifold Hypothesis")
    print("=" * 60)
    print("Hypothesis: There should be abrupt changes in curvature between strata")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = CurvatureDiscontinuityAnalyzer(data, strata, labels=labels, domains=domains)
    
    # Test the hypothesis
    print("Testing stratified manifold hypothesis...")
    results = analyzer.test_stratified_manifold_hypothesis()
    
    # Print results
    print(f"\nHypothesis Test Results:")
    print(f"  Hypothesis Supported: {results['hypothesis_supported']}")
    print(f"  Significance Level: {results['significance_level']}")
    
    stats = results['statistics']
    print(f"\nStatistical Results:")
    print(f"  Boundary Curvature: {stats['boundary_curvature_mean']:.4f} ± {stats['boundary_curvature_std']:.4f}")
    print(f"  Interior Curvature: {stats['interior_curvature_mean']:.4f} ± {stats['interior_curvature_std']:.4f}")
    print(f"  T-test p-value: {stats['t_pvalue']:.6f}")
    print(f"  Mann-Whitney U p-value: {stats['u_pvalue']:.6f}")
    print(f"  Cohen's d (effect size): {stats['cohens_d']:.4f}")
    print(f"  Boundary Points: {stats['n_boundary_points']}")
    print(f"  Interior Points: {stats['n_interior_points']}")
    
    # Discontinuity scores
    if results['discontinuity_scores']:
        print(f"\nDiscontinuity Scores (Higher = More Abrupt):")
        for pair, score in results['discontinuity_scores'].items():
            print(f"  {pair}: {score:.4f}")
    
    # Gradient analysis
    grad_stats = results['gradient_statistics']
    print(f"\nCurvature Gradient Analysis:")
    print(f"  Mean Gradient: {grad_stats['gradient_mean']:.4f} ± {grad_stats['gradient_std']:.4f}")
    print(f"  High Gradient Points: {grad_stats['high_gradient_points']}")
    
    # Create visualization
    print("\nCreating discontinuity visualization...")
    analyzer.visualize_curvature_discontinuities(save_path=save_path)
    
    return results

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_swiss_roll
    
    # Generate sample data with clear strata
    data, _ = make_swiss_roll(n_samples=500, noise=0.1)
    
    # Create strata with clear boundaries
    strata = np.zeros(500, dtype=int)
    strata[100:200] = 1
    strata[200:300] = 2
    strata[300:400] = 3
    strata[400:500] = 4
    
    # Run discontinuity analysis
    results = run_curvature_discontinuity_analysis(data, strata)
