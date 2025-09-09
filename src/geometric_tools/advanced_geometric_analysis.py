"""
Advanced geometric analysis tools for stratified manifold learning.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, UMAP
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
from scipy.stats import entropy
import networkx as nx
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

class AdvancedGeometricAnalyzer:
    """Advanced geometric analysis for stratified manifolds."""
    
    def __init__(self, data, labels=None, domains=None):
        self.data = np.array(data)
        self.labels = labels
        self.domains = domains
        self.n_samples, self.n_features = self.data.shape
        
    def compute_local_dimension(self, k=10, method='pca'):
        """Compute local intrinsic dimension using various methods."""
        if method == 'pca':
            return self._local_dimension_pca(k)
        elif method == 'neighbors':
            return self._local_dimension_neighbors(k)
        elif method == 'correlation':
            return self._local_dimension_correlation(k)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _local_dimension_pca(self, k):
        """Compute local dimension using PCA on k-nearest neighbors."""
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        
        local_dims = []
        for i in range(self.n_samples):
            neighbor_data = self.data[indices[i][1:]]  # Exclude self
            if len(neighbor_data) < 2:
                local_dims.append(1)
                continue
                
            pca = PCA()
            pca.fit(neighbor_data)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            local_dim = np.searchsorted(cumulative_variance, 0.95) + 1
            local_dims.append(local_dim)
        
        return np.array(local_dims)
    
    def _local_dimension_neighbors(self, k):
        """Compute local dimension using neighbor distance scaling."""
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        
        local_dims = []
        for i in range(self.n_samples):
            neighbor_distances = distances[i][1:]  # Exclude self
            if len(neighbor_distances) < 2:
                local_dims.append(1)
                continue
            
            # Estimate dimension from distance scaling
            log_distances = np.log(neighbor_distances + 1e-8)
            log_ranks = np.log(np.arange(1, len(neighbor_distances) + 1))
            
            # Linear regression slope gives dimension estimate
            slope = np.polyfit(log_ranks, log_distances, 1)[0]
            local_dim = max(1, int(-slope))
            local_dims.append(local_dim)
        
        return np.array(local_dims)
    
    def _local_dimension_correlation(self, k):
        """Compute local dimension using correlation dimension."""
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        
        local_dims = []
        for i in range(self.n_samples):
            neighbor_distances = distances[i][1:]  # Exclude self
            if len(neighbor_distances) < 2:
                local_dims.append(1)
                continue
            
            # Correlation dimension estimation
            r_max = np.max(neighbor_distances)
            r_min = np.min(neighbor_distances)
            
            if r_max <= r_min:
                local_dims.append(1)
                continue
            
            # Estimate dimension from correlation integral
            log_r = np.log(neighbor_distances + 1e-8)
            log_C = np.log(np.arange(1, len(neighbor_distances) + 1) / len(neighbor_distances))
            
            slope = np.polyfit(log_r, log_C, 1)[0]
            local_dim = max(1, int(slope))
            local_dims.append(local_dim)
        
        return np.array(local_dims)
    
    def compute_curvature(self, k=10):
        """Compute local curvature using PCA on neighborhoods."""
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        
        curvatures = []
        for i in range(self.n_samples):
            neighbor_data = self.data[indices[i][1:]]  # Exclude self
            if len(neighbor_data) < 3:
                curvatures.append(0.0)
                continue
            
            # Center the neighborhood
            center = np.mean(neighbor_data, axis=0)
            centered_data = neighbor_data - center
            
            # Compute PCA
            pca = PCA()
            pca.fit(centered_data)
            
            # Curvature is related to the ratio of eigenvalues
            eigenvalues = pca.explained_variance_
            if len(eigenvalues) < 2 or eigenvalues[0] == 0:
                curvatures.append(0.0)
                continue
            
            # Curvature estimate
            curvature = eigenvalues[1] / eigenvalues[0] if eigenvalues[0] > 0 else 0
            curvatures.append(curvature)
        
        return np.array(curvatures)
    
    def compute_manifold_density(self, k=10):
        """Compute local manifold density."""
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        
        densities = []
        for i in range(self.n_samples):
            neighbor_distances = distances[i][1:]  # Exclude self
            avg_distance = np.mean(neighbor_distances)
            density = 1.0 / (avg_distance + 1e-8)
            densities.append(density)
        
        return np.array(densities)
    
    def detect_manifold_boundaries(self, k=10, threshold=0.5):
        """Detect manifold boundaries using local geometry."""
        local_dims = self.compute_local_dimension(k)
        curvatures = self.compute_curvature(k)
        densities = self.compute_manifold_density(k)
        
        # Boundary detection based on local properties
        boundary_scores = np.zeros(self.n_samples)
        
        # High curvature indicates boundaries
        boundary_scores += curvatures / (np.max(curvatures) + 1e-8)
        
        # Low density indicates boundaries
        boundary_scores += (1 - densities / (np.max(densities) + 1e-8))
        
        # Dimension changes indicate boundaries
        dim_std = np.std(local_dims)
        if dim_std > 0:
            dim_deviation = np.abs(local_dims - np.mean(local_dims)) / dim_std
            boundary_scores += dim_deviation / (np.max(dim_deviation) + 1e-8)
        
        # Normalize scores
        boundary_scores = boundary_scores / 3.0
        
        # Identify boundaries
        boundaries = boundary_scores > threshold
        
        return boundaries, boundary_scores
    
    def compute_stratified_metrics(self, strata):
        """Compute metrics specific to stratified manifolds."""
        unique_strata = np.unique(strata)
        n_strata = len(unique_strata)
        
        metrics = {}
        
        # Stratum separation
        stratum_centers = []
        stratum_radii = []
        
        for stratum in unique_strata:
            stratum_data = self.data[strata == stratum]
            center = np.mean(stratum_data, axis=0)
            radius = np.mean(np.linalg.norm(stratum_data - center, axis=1))
            stratum_centers.append(center)
            stratum_radii.append(radius)
        
        stratum_centers = np.array(stratum_centers)
        stratum_radii = np.array(stratum_radii)
        
        # Inter-stratum distances
        inter_stratum_distances = pdist(stratum_centers)
        avg_inter_distance = np.mean(inter_stratum_distances)
        
        # Stratum compactness
        avg_stratum_radius = np.mean(stratum_radii)
        
        # Separation ratio
        separation_ratio = avg_inter_distance / (avg_stratum_radius + 1e-8)
        
        metrics['separation_ratio'] = separation_ratio
        metrics['avg_inter_distance'] = avg_inter_distance
        metrics['avg_stratum_radius'] = avg_stratum_radius
        metrics['n_strata'] = n_strata
        
        return metrics
    
    def visualize_geometric_properties(self, strata=None, save_path=None):
        """Create comprehensive visualization of geometric properties."""
        # Compute geometric properties
        local_dims = self.compute_local_dimension()
        curvatures = self.compute_curvature()
        densities = self.compute_manifold_density()
        boundaries, boundary_scores = self.detect_manifold_boundaries()
        
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(self.data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Local dimensions
        scatter1 = axes[0, 0].scatter(data_2d[:, 0], data_2d[:, 1], c=local_dims, cmap='viridis', alpha=0.7, s=30)
        axes[0, 0].set_title('Local Intrinsic Dimensions')
        axes[0, 0].set_xlabel('PC1')
        axes[0, 0].set_ylabel('PC2')
        plt.colorbar(scatter1, ax=axes[0, 0], label='Dimension')
        
        # Plot 2: Curvatures
        scatter2 = axes[0, 1].scatter(data_2d[:, 0], data_2d[:, 1], c=curvatures, cmap='plasma', alpha=0.7, s=30)
        axes[0, 1].set_title('Local Curvatures')
        axes[0, 1].set_xlabel('PC1')
        axes[0, 1].set_ylabel('PC2')
        plt.colorbar(scatter2, ax=axes[0, 1], label='Curvature')
        
        # Plot 3: Densities
        scatter3 = axes[0, 2].scatter(data_2d[:, 0], data_2d[:, 1], c=densities, cmap='coolwarm', alpha=0.7, s=30)
        axes[0, 2].set_title('Local Densities')
        axes[0, 2].set_xlabel('PC1')
        axes[0, 2].set_ylabel('PC2')
        plt.colorbar(scatter3, ax=axes[0, 2], label='Density')
        
        # Plot 4: Boundary detection
        scatter4 = axes[1, 0].scatter(data_2d[:, 0], data_2d[:, 1], c=boundary_scores, cmap='Reds', alpha=0.7, s=30)
        axes[1, 0].set_title('Manifold Boundaries')
        axes[1, 0].set_xlabel('PC1')
        axes[1, 0].set_ylabel('PC2')
        plt.colorbar(scatter4, ax=axes[1, 0], label='Boundary Score')
        
        # Plot 5: Strata (if provided)
        if strata is not None:
            scatter5 = axes[1, 1].scatter(data_2d[:, 0], data_2d[:, 1], c=strata, cmap='tab10', alpha=0.7, s=30)
            axes[1, 1].set_title('Stratified Structure')
            axes[1, 1].set_xlabel('PC1')
            axes[1, 1].set_ylabel('PC2')
            plt.colorbar(scatter5, ax=axes[1, 1], label='Stratum')
        else:
            axes[1, 1].text(0.5, 0.5, 'No strata provided', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Stratified Structure')
        
        # Plot 6: Domains (if provided)
        if self.domains is not None:
            domain_colors = {domain: i for i, domain in enumerate(set(self.domains))}
            domain_numeric = [domain_colors[d] for d in self.domains]
            scatter6 = axes[1, 2].scatter(data_2d[:, 0], data_2d[:, 1], c=domain_numeric, cmap='Set3', alpha=0.7, s=30)
            axes[1, 2].set_title('Domain Structure')
            axes[1, 2].set_xlabel('PC1')
            axes[1, 2].set_ylabel('PC2')
            plt.colorbar(scatter6, ax=axes[1, 2], label='Domain')
        else:
            axes[1, 2].text(0.5, 0.5, 'No domains provided', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Domain Structure')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return {
            'local_dims': local_dims,
            'curvatures': curvatures,
            'densities': densities,
            'boundaries': boundaries,
            'boundary_scores': boundary_scores
        }

class TopologicalAnalyzer:
    """Topological analysis for stratified manifolds."""
    
    def __init__(self, data):
        self.data = np.array(data)
        self.n_samples, self.n_features = self.data.shape
    
    def compute_persistence_diagram(self, max_dimension=2):
        """Compute persistence diagram using Vietoris-Rips complex."""
        try:
            import ripser
            from ripser import ripser
            
            # Compute persistence diagram
            dgm = ripser(self.data, maxdim=max_dimension)
            
            return dgm
        except ImportError:
            print("ripser not available, returning None")
            return None
    
    def compute_betti_numbers(self, persistence_diagram):
        """Compute Betti numbers from persistence diagram."""
        if persistence_diagram is None:
            return None
        
        betti_numbers = {}
        for dim in range(len(persistence_diagram['dgms'])):
            dgm = persistence_diagram['dgms'][dim]
            if len(dgm) > 0:
                # Count persistent features
                persistence = dgm[:, 1] - dgm[:, 0]
                betti_numbers[dim] = len(persistence[persistence > 0.1])  # Threshold for persistence
        
        return betti_numbers
    
    def compute_topological_entropy(self, k=10):
        """Compute topological entropy using neighborhood graph."""
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        
        # Create adjacency matrix
        n = self.n_samples
        adj_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in indices[i][1:]:  # Exclude self
                adj_matrix[i, j] = 1
        
        # Compute graph entropy
        degrees = np.sum(adj_matrix, axis=1)
        degree_probs = degrees / np.sum(degrees)
        entropy_val = entropy(degree_probs[degree_probs > 0])
        
        return entropy_val

def run_advanced_geometric_analysis(data, strata=None, domains=None, save_path='geometric_analysis.png'):
    """Run comprehensive geometric analysis."""
    print("🔬 Running Advanced Geometric Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = AdvancedGeometricAnalyzer(data, labels=None, domains=domains)
    
    # Compute geometric properties
    print("Computing geometric properties...")
    geometric_props = analyzer.visualize_geometric_properties(strata=strata, save_path=save_path)
    
    # Compute stratified metrics if strata provided
    if strata is not None:
        print("Computing stratified metrics...")
        stratified_metrics = analyzer.compute_stratified_metrics(strata)
        
        print(f"\nStratified Metrics:")
        print(f"  Separation Ratio: {stratified_metrics['separation_ratio']:.4f}")
        print(f"  Average Inter-Distance: {stratified_metrics['avg_inter_distance']:.4f}")
        print(f"  Average Stratum Radius: {stratified_metrics['avg_stratum_radius']:.4f}")
        print(f"  Number of Strata: {stratified_metrics['n_strata']}")
    
    # Topological analysis
    print("\nRunning topological analysis...")
    topo_analyzer = TopologicalAnalyzer(data)
    
    # Persistence diagram
    persistence_diagram = topo_analyzer.compute_persistence_diagram()
    
    # Betti numbers
    if persistence_diagram is not None:
        betti_numbers = topo_analyzer.compute_betti_numbers(persistence_diagram)
        print(f"Betti Numbers: {betti_numbers}")
    
    # Topological entropy
    topo_entropy = topo_analyzer.compute_topological_entropy()
    print(f"Topological Entropy: {topo_entropy:.4f}")
    
    # Summary statistics
    print(f"\nGeometric Properties Summary:")
    print(f"  Average Local Dimension: {np.mean(geometric_props['local_dims']):.2f}")
    print(f"  Average Curvature: {np.mean(geometric_props['curvatures']):.4f}")
    print(f"  Average Density: {np.mean(geometric_props['densities']):.4f}")
    print(f"  Boundary Points: {np.sum(geometric_props['boundaries'])}")
    
    return {
        'geometric_properties': geometric_props,
        'stratified_metrics': stratified_metrics if strata is not None else None,
        'betti_numbers': betti_numbers if persistence_diagram is not None else None,
        'topological_entropy': topo_entropy
    }

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_swiss_roll
    
    # Generate sample data
    data, _ = make_swiss_roll(n_samples=500, noise=0.1)
    
    # Create some strata
    strata = np.zeros(500, dtype=int)
    strata[100:200] = 1
    strata[200:300] = 2
    strata[300:400] = 3
    strata[400:500] = 4
    
    # Run analysis
    results = run_advanced_geometric_analysis(data, strata=strata)
