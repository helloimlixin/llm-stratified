"""
Deep geometric analysis for stratified manifold learning.
Advanced tools for understanding manifold structure, topology, and curvature.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
# UMAP is not available in sklearn, removing for now
# from sklearn.manifold import UMAP
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
from scipy.stats import entropy, kstest, pearsonr
from scipy.signal import find_peaks
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

class DeepGeometricAnalyzer:
    """
    Deep geometric analysis for stratified manifolds.
    
    Implements advanced geometric concepts:
    - Ricci curvature and scalar curvature
    - Topological data analysis (TDA)
    - Manifold learning and intrinsic dimension
    - Geometric flow analysis
    - Stratum boundary detection
    """
    
    def __init__(self, data, strata, labels=None, domains=None):
        self.data = np.array(data)
        self.strata = np.array(strata)
        self.labels = labels
        self.domains = domains
        self.n_samples, self.n_features = self.data.shape
        self.unique_strata = np.unique(strata)
        
    def compute_ricci_curvature(self, k=10):
        """
        Compute Ricci curvature tensor components.
        
        Ricci curvature is a contraction of the Riemann curvature tensor
        and measures the average sectional curvature in all directions.
        """
        # Adjust k based on available samples
        k = min(k, self.n_samples - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        
        ricci_curvatures = []
        for i in range(self.n_samples):
            neighbor_data = self.data[indices[i][1:]]
            if len(neighbor_data) < 3:
                ricci_curvatures.append(0.0)
                continue
            
            # Compute Ricci curvature using local metric tensor
            ricci_curv = self._compute_local_ricci_curvature(neighbor_data)
            ricci_curvatures.append(ricci_curv)
        
        return np.array(ricci_curvatures)
    
    def _compute_local_ricci_curvature(self, local_data):
        """
        Compute local Ricci curvature from neighborhood data.
        
        Uses the relationship between Ricci curvature and volume growth.
        """
        if len(local_data) < 3:
            return 0.0
        
        # Center the data
        center = np.mean(local_data, axis=0)
        centered_data = local_data - center
        
        # Compute local metric tensor (covariance matrix)
        metric_tensor = np.cov(centered_data.T)
        
        # Compute Ricci curvature as trace of Riemann tensor
        # Approximate using eigenvalues of metric tensor
        eigenvalues = np.linalg.eigvals(metric_tensor)
        eigenvalues = np.real(eigenvalues[eigenvalues > 1e-10])  # Remove near-zero eigenvalues
        
        if len(eigenvalues) < 2:
            return 0.0
        
        # Ricci curvature estimate
        # For a surface, Ricci = 2 * Gaussian curvature
        ricci_curv = np.sum(eigenvalues) / (len(eigenvalues) * np.mean(eigenvalues))
        
        return ricci_curv
    
    def compute_scalar_curvature(self, k=10):
        """
        Compute scalar curvature (trace of Ricci tensor).
        
        Scalar curvature is the simplest curvature invariant
        and measures the average curvature at each point.
        """
        ricci_curvatures = self.compute_ricci_curvature(k)
        
        # Scalar curvature is the trace of Ricci tensor
        # For our purposes, we use the mean Ricci curvature
        scalar_curvature = np.mean(ricci_curvatures)
        
        return scalar_curvature, ricci_curvatures
    
    def analyze_manifold_topology(self, k=10):
        """
        Analyze topological properties of the manifold.
        
        Uses persistent homology concepts to understand
        the topological structure of the data.
        """
        # Compute pairwise distances
        distances = pairwise_distances(self.data)
        
        # Analyze connectivity at different scales
        connectivity_analysis = {}
        
        for threshold in np.linspace(0.1, 2.0, 20):
            # Create adjacency matrix
            adjacency = (distances < threshold).astype(int)
            np.fill_diagonal(adjacency, 0)
            
            # Count connected components
            n_components = self._count_connected_components(adjacency)
            
            # Analyze component sizes
            component_sizes = self._get_component_sizes(adjacency)
            
            connectivity_analysis[threshold] = {
                'n_components': n_components,
                'component_sizes': component_sizes,
                'largest_component_size': max(component_sizes) if component_sizes else 0,
                'connectivity_ratio': max(component_sizes) / self.n_samples if component_sizes else 0
            }
        
        return connectivity_analysis
    
    def _count_connected_components(self, adjacency):
        """Count connected components in adjacency matrix."""
        visited = np.zeros(self.n_samples, dtype=bool)
        n_components = 0
        
        for i in range(self.n_samples):
            if not visited[i]:
                self._dfs_component(i, adjacency, visited)
                n_components += 1
        
        return n_components
    
    def _dfs_component(self, start, adjacency, visited):
        """Depth-first search to find connected component."""
        stack = [start]
        
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            
            visited[node] = True
            
            # Add unvisited neighbors to stack
            neighbors = np.where(adjacency[node] == 1)[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    stack.append(neighbor)
    
    def _get_component_sizes(self, adjacency):
        """Get sizes of connected components."""
        visited = np.zeros(self.n_samples, dtype=bool)
        component_sizes = []
        
        for i in range(self.n_samples):
            if not visited[i]:
                size = self._dfs_component_size(i, adjacency, visited)
                component_sizes.append(size)
        
        return component_sizes
    
    def _dfs_component_size(self, start, adjacency, visited):
        """DFS to compute component size."""
        stack = [start]
        size = 0
        
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            
            visited[node] = True
            size += 1
            
            # Add unvisited neighbors to stack
            neighbors = np.where(adjacency[node] == 1)[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    stack.append(neighbor)
        
        return size
    
    def compute_intrinsic_dimension(self, k=10, method='pca'):
        """
        Compute intrinsic dimension using multiple methods.
        
        Methods:
        - pca: PCA-based dimension estimation
        - neighbors: Nearest neighbor-based estimation
        - correlation: Correlation dimension
        - mle: Maximum likelihood estimation
        """
        if method == 'pca':
            return self._intrinsic_dimension_pca(k)
        elif method == 'neighbors':
            return self._intrinsic_dimension_neighbors(k)
        elif method == 'correlation':
            return self._intrinsic_dimension_correlation(k)
        elif method == 'mle':
            return self._intrinsic_dimension_mle(k)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _intrinsic_dimension_pca(self, k):
        """PCA-based intrinsic dimension estimation."""
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        
        intrinsic_dims = []
        for i in range(self.n_samples):
            neighbor_data = self.data[indices[i][1:]]
            if len(neighbor_data) < 3:
                intrinsic_dims.append(0)
                continue
            
            # Center the data
            center = np.mean(neighbor_data, axis=0)
            centered_data = neighbor_data - center
            
            # Compute PCA
            pca = PCA()
            pca.fit(centered_data)
            eigenvalues = pca.explained_variance_
            
            # Estimate dimension using eigenvalue decay
            cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
            intrinsic_dim = np.argmax(cumulative_variance >= 0.95) + 1
            
            intrinsic_dims.append(intrinsic_dim)
        
        return np.array(intrinsic_dims)
    
    def _intrinsic_dimension_neighbors(self, k):
        """Nearest neighbor-based intrinsic dimension estimation."""
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        
        intrinsic_dims = []
        for i in range(self.n_samples):
            neighbor_distances = distances[i][1:]  # Exclude self
            
            if len(neighbor_distances) < 2:
                intrinsic_dims.append(0)
                continue
            
            # Estimate dimension using distance scaling
            # In d-dimensional space, distances scale as r^(1/d)
            log_distances = np.log(neighbor_distances)
            log_indices = np.log(np.arange(1, len(neighbor_distances) + 1))
            
            # Linear regression to estimate dimension
            if len(log_distances) > 1:
                correlation = np.corrcoef(log_indices, log_distances)[0, 1]
                if not np.isnan(correlation) and correlation > 0:
                    intrinsic_dim = 1 / correlation
                    intrinsic_dims.append(max(1, min(intrinsic_dim, self.n_features)))
                else:
                    intrinsic_dims.append(1)
            else:
                intrinsic_dims.append(1)
        
        return np.array(intrinsic_dims)
    
    def _intrinsic_dimension_correlation(self, k):
        """Correlation dimension estimation."""
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        
        intrinsic_dims = []
        for i in range(self.n_samples):
            neighbor_distances = distances[i][1:]
            
            if len(neighbor_distances) < 2:
                intrinsic_dims.append(0)
                continue
            
            # Estimate correlation dimension
            # C(r) ~ r^d, where d is the correlation dimension
            radii = np.linspace(neighbor_distances[0], neighbor_distances[-1], 10)
            correlation_integrals = []
            
            for r in radii:
                count = np.sum(neighbor_distances <= r)
                correlation_integrals.append(count / len(neighbor_distances))
            
            # Estimate dimension from slope
            log_radii = np.log(radii[correlation_integrals > 0])
            log_correlations = np.log(np.array(correlation_integrals)[correlation_integrals > 0])
            
            if len(log_radii) > 1:
                correlation = np.corrcoef(log_radii, log_correlations)[0, 1]
                if not np.isnan(correlation) and correlation > 0:
                    intrinsic_dim = correlation
                    intrinsic_dims.append(max(1, min(intrinsic_dim, self.n_features)))
                else:
                    intrinsic_dims.append(1)
            else:
                intrinsic_dims.append(1)
        
        return np.array(intrinsic_dims)
    
    def _intrinsic_dimension_mle(self, k):
        """Maximum likelihood estimation of intrinsic dimension."""
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        
        intrinsic_dims = []
        for i in range(self.n_samples):
            neighbor_distances = distances[i][1:]
            
            if len(neighbor_distances) < 2:
                intrinsic_dims.append(0)
                continue
            
            # MLE estimator for intrinsic dimension
            # d = 1 / (1/k * sum(log(r_k/r_1)))
            r_1 = neighbor_distances[0]
            r_k = neighbor_distances[-1]
            
            if r_1 > 0 and r_k > r_1:
                intrinsic_dim = 1 / (np.mean(np.log(neighbor_distances[1:] / r_1)))
                intrinsic_dims.append(max(1, min(intrinsic_dim, self.n_features)))
            else:
                intrinsic_dims.append(1)
        
        return np.array(intrinsic_dims)
    
    def analyze_stratum_boundaries(self, k=10):
        """
        Analyze stratum boundaries using multiple geometric criteria.
        
        Returns detailed boundary analysis including:
        - Boundary strength
        - Geometric properties
        - Topological features
        """
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        
        boundary_analysis = {}
        
        for i in range(self.n_samples):
            neighbor_indices = indices[i][1:]
            neighbor_strata = self.strata[neighbor_indices]
            current_stratum = self.strata[i]
            
            # Compute boundary strength
            cross_stratum_neighbors = np.sum(neighbor_strata != current_stratum)
            cross_stratum_ratio = cross_stratum_neighbors / len(neighbor_strata)
            
            # Compute geometric properties at boundary
            neighbor_data = self.data[neighbor_indices]
            
            # Local curvature
            curvature = self._compute_local_riemannian_curvature(neighbor_data)
            
            # Local density
            density = len(neighbor_indices) / (np.pi * distances[i][-1]**2)
            
            # Local dimension
            intrinsic_dim = self._intrinsic_dimension_pca(k)[i]
            
            boundary_analysis[i] = {
                'is_boundary': cross_stratum_ratio > 0.3,
                'boundary_strength': cross_stratum_ratio,
                'curvature': curvature,
                'density': density,
                'intrinsic_dimension': intrinsic_dim,
                'stratum': current_stratum,
                'neighbor_strata': neighbor_strata.tolist()
            }
        
        return boundary_analysis
    
    def _compute_local_riemannian_curvature(self, local_data):
        """Compute local Riemannian curvature."""
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
        
        # Curvature estimate
        curvature = eigenvalues[1] / eigenvalues[0] if eigenvalues[0] > 0 else 0
        
        return curvature
    
    def compute_geometric_flow(self, k=10):
        """
        Compute geometric flow properties.
        
        Analyzes how geometric properties change across the manifold
        and identifies flow patterns between strata.
        """
        # Compute local geometric properties
        boundary_analysis = self.analyze_stratum_boundaries(k)
        
        # Compute flow vectors
        flow_vectors = []
        flow_strengths = []
        
        for i in range(self.n_samples):
            if boundary_analysis[i]['is_boundary']:
                # Compute flow vector from curvature gradient
                neighbor_indices = nbrs.kneighbors([self.data[i]])[1][0][1:]
                neighbor_curvatures = [boundary_analysis[j]['curvature'] for j in neighbor_indices]
                
                if len(neighbor_curvatures) > 1:
                    # Compute curvature gradient
                    curvature_gradient = np.std(neighbor_curvatures)
                    flow_strengths.append(curvature_gradient)
                    
                    # Compute flow direction (simplified)
                    flow_direction = np.random.randn(self.n_features)  # Placeholder
                    flow_direction = flow_direction / np.linalg.norm(flow_direction)
                    flow_vectors.append(flow_direction)
                else:
                    flow_strengths.append(0.0)
                    flow_vectors.append(np.zeros(self.n_features))
            else:
                flow_strengths.append(0.0)
                flow_vectors.append(np.zeros(self.n_features))
        
        return {
            'flow_vectors': np.array(flow_vectors),
            'flow_strengths': np.array(flow_strengths),
            'boundary_analysis': boundary_analysis
        }
    
    def analyze_stratum_evolution(self, k=10):
        """
        Analyze how strata evolve and interact.
        
        Studies the geometric relationships between different strata
        and how they influence each other.
        """
        stratum_analysis = {}
        
        for stratum in self.unique_strata:
            stratum_mask = self.strata == stratum
            stratum_data = self.data[stratum_mask]
            
            if len(stratum_data) < 3:
                stratum_analysis[stratum] = {
                    'n_points': len(stratum_data),
                    'curvature_mean': 0.0,
                    'curvature_std': 0.0,
                    'intrinsic_dim_mean': 0.0,
                    'intrinsic_dim_std': 0.0,
                    'density_mean': 0.0,
                    'density_std': 0.0
                }
                continue
            
            # Create analyzer for this stratum
            stratum_analyzer = DeepGeometricAnalyzer(stratum_data, 
                                                   self.strata[stratum_mask])
            
            # Compute geometric properties
            ricci_curvatures = stratum_analyzer.compute_ricci_curvature(k)
            intrinsic_dims = stratum_analyzer.compute_intrinsic_dimension(k)
            
            # Compute density
            nbrs = NearestNeighbors(n_neighbors=min(k+1, len(stratum_data))).fit(stratum_data)
            distances, _ = nbrs.kneighbors(stratum_data)
            densities = 1.0 / (distances[:, -1] + 1e-8)
            
            stratum_analysis[stratum] = {
                'n_points': len(stratum_data),
                'curvature_mean': float(np.mean(ricci_curvatures)),
                'curvature_std': float(np.std(ricci_curvatures)),
                'intrinsic_dim_mean': float(np.mean(intrinsic_dims)),
                'intrinsic_dim_std': float(np.std(intrinsic_dims)),
                'density_mean': float(np.mean(densities)),
                'density_std': float(np.std(densities))
            }
        
        return stratum_analysis
    
    def visualize_deep_analysis(self, save_path=None):
        """
        Create comprehensive visualization of deep geometric analysis.
        """
        # Compute various geometric properties
        ricci_curvatures = self.compute_ricci_curvature()
        intrinsic_dims = self.compute_intrinsic_dimension()
        boundary_analysis = self.analyze_stratum_boundaries()
        stratum_analysis = self.analyze_stratum_evolution()
        
        # Create subplots
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        # PCA 2D projection
        pca_2d = PCA(n_components=2)
        data_2d = pca_2d.fit_transform(self.data)
        
        # Plot 1: Ricci curvature
        scatter1 = axes[0, 0].scatter(data_2d[:, 0], data_2d[:, 1], 
                                      c=ricci_curvatures, cmap='plasma', alpha=0.7, s=30)
        axes[0, 0].set_title('Ricci Curvature')
        axes[0, 0].set_xlabel('PC1')
        axes[0, 0].set_ylabel('PC2')
        plt.colorbar(scatter1, ax=axes[0, 0], label='Ricci Curvature')
        
        # Plot 2: Intrinsic dimension
        scatter2 = axes[0, 1].scatter(data_2d[:, 0], data_2d[:, 1], 
                                     c=intrinsic_dims, cmap='viridis', alpha=0.7, s=30)
        axes[0, 1].set_title('Intrinsic Dimension')
        axes[0, 1].set_xlabel('PC1')
        axes[0, 1].set_ylabel('PC2')
        plt.colorbar(scatter2, ax=axes[0, 1], label='Intrinsic Dimension')
        
        # Plot 3: Boundary strength
        boundary_strengths = [boundary_analysis[i]['boundary_strength'] for i in range(self.n_samples)]
        scatter3 = axes[0, 2].scatter(data_2d[:, 0], data_2d[:, 1], 
                                     c=boundary_strengths, cmap='Reds', alpha=0.7, s=30)
        axes[0, 2].set_title('Boundary Strength')
        axes[0, 2].set_xlabel('PC1')
        axes[0, 2].set_ylabel('PC2')
        plt.colorbar(scatter3, ax=axes[0, 2], label='Boundary Strength')
        
        # Plot 4: Strata
        scatter4 = axes[0, 3].scatter(data_2d[:, 0], data_2d[:, 1], 
                                     c=self.strata, cmap='tab10', alpha=0.7, s=30)
        axes[0, 3].set_title('Stratified Structure')
        axes[0, 3].set_xlabel('PC1')
        axes[0, 3].set_ylabel('PC2')
        plt.colorbar(scatter4, ax=axes[0, 3], label='Stratum')
        
        # Plot 5: Curvature distribution by stratum
        stratum_curvatures = {}
        for stratum in self.unique_strata:
            stratum_mask = self.strata == stratum
            stratum_curv = ricci_curvatures[stratum_mask]
            if len(stratum_curv) > 0:
                stratum_curvatures[stratum] = stratum_curv
        
        if stratum_curvatures:
            axes[1, 0].boxplot(list(stratum_curvatures.values()), 
                              labels=list(stratum_curvatures.keys()))
            axes[1, 0].set_title('Ricci Curvature by Stratum')
            axes[1, 0].set_ylabel('Ricci Curvature')
            axes[1, 0].set_xlabel('Stratum')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 6: Intrinsic dimension distribution by stratum
        stratum_dims = {}
        for stratum in self.unique_strata:
            stratum_mask = self.strata == stratum
            stratum_dim = intrinsic_dims[stratum_mask]
            if len(stratum_dim) > 0:
                stratum_dims[stratum] = stratum_dim
        
        if stratum_dims:
            axes[1, 1].boxplot(list(stratum_dims.values()), 
                              labels=list(stratum_dims.keys()))
            axes[1, 1].set_title('Intrinsic Dimension by Stratum')
            axes[1, 1].set_ylabel('Intrinsic Dimension')
            axes[1, 1].set_xlabel('Stratum')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 7: Boundary strength histogram
        axes[1, 2].hist(boundary_strengths, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 2].axvline(np.mean(boundary_strengths), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(boundary_strengths):.3f}')
        axes[1, 2].set_title('Boundary Strength Distribution')
        axes[1, 2].set_xlabel('Boundary Strength')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # Plot 8: Stratum evolution summary
        strata = list(stratum_analysis.keys())
        curvature_means = [stratum_analysis[s]['curvature_mean'] for s in strata]
        intrinsic_dim_means = [stratum_analysis[s]['intrinsic_dim_mean'] for s in strata]
        
        axes[1, 3].scatter(curvature_means, intrinsic_dim_means, 
                          c=strata, cmap='tab10', s=100, alpha=0.7)
        axes[1, 3].set_title('Stratum Evolution')
        axes[1, 3].set_xlabel('Mean Ricci Curvature')
        axes[1, 3].set_ylabel('Mean Intrinsic Dimension')
        axes[1, 3].grid(True, alpha=0.3)
        
        # Add stratum labels
        for i, stratum in enumerate(strata):
            axes[1, 3].annotate(f'S{stratum}', 
                               (curvature_means[i], intrinsic_dim_means[i]),
                               xytext=(5, 5), textcoords='offset points')
        
        # Plot 9: Curvature vs Intrinsic Dimension correlation
        axes[2, 0].scatter(ricci_curvatures, intrinsic_dims, alpha=0.6, s=30)
        axes[2, 0].set_title('Curvature vs Intrinsic Dimension')
        axes[2, 0].set_xlabel('Ricci Curvature')
        axes[2, 0].set_ylabel('Intrinsic Dimension')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(ricci_curvatures, intrinsic_dims)[0, 1]
        axes[2, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[2, 0].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 10: Boundary strength vs Curvature
        axes[2, 1].scatter(boundary_strengths, ricci_curvatures, alpha=0.6, s=30)
        axes[2, 1].set_title('Boundary Strength vs Curvature')
        axes[2, 1].set_xlabel('Boundary Strength')
        axes[2, 1].set_ylabel('Ricci Curvature')
        axes[2, 1].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation2 = np.corrcoef(boundary_strengths, ricci_curvatures)[0, 1]
        axes[2, 1].text(0.05, 0.95, f'Correlation: {correlation2:.3f}', 
                       transform=axes[2, 1].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 11: Stratum size distribution
        stratum_sizes = [stratum_analysis[s]['n_points'] for s in strata]
        axes[2, 2].bar(strata, stratum_sizes, alpha=0.7)
        axes[2, 2].set_title('Stratum Size Distribution')
        axes[2, 2].set_xlabel('Stratum')
        axes[2, 2].set_ylabel('Number of Points')
        axes[2, 2].grid(True, alpha=0.3)
        
        # Plot 12: Summary statistics
        axes[2, 3].text(0.1, 0.9, f"Deep Geometric Analysis Summary", 
                        fontweight='bold', transform=axes[2, 3].transAxes)
        axes[2, 3].text(0.1, 0.8, f"Total Points: {self.n_samples}", 
                        transform=axes[2, 3].transAxes)
        axes[2, 3].text(0.1, 0.7, f"Number of Strata: {len(self.unique_strata)}", 
                        transform=axes[2, 3].transAxes)
        axes[2, 3].text(0.1, 0.6, f"Mean Ricci Curvature: {np.mean(ricci_curvatures):.4f}", 
                        transform=axes[2, 3].transAxes)
        axes[2, 3].text(0.1, 0.5, f"Mean Intrinsic Dimension: {np.mean(intrinsic_dims):.4f}", 
                        transform=axes[2, 3].transAxes)
        axes[2, 3].text(0.1, 0.4, f"Mean Boundary Strength: {np.mean(boundary_strengths):.4f}", 
                        transform=axes[2, 3].transAxes)
        axes[2, 3].text(0.1, 0.3, f"Boundary Points: {np.sum(np.array(boundary_strengths) > 0.3)}", 
                        transform=axes[2, 3].transAxes)
        
        axes[2, 3].set_xlim(0, 1)
        axes[2, 3].set_ylim(0, 1)
        axes[2, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def run_deep_geometric_analysis(data, strata, labels=None, domains=None, 
                               save_path='deep_geometric_analysis.png'):
    """
    Run comprehensive deep geometric analysis.
    
    Analyzes manifold structure, topology, curvature, and stratum evolution.
    """
    print("🔬 Running Deep Geometric Analysis")
    print("=" * 60)
    print("Advanced analysis of stratified manifold structure")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = DeepGeometricAnalyzer(data, strata, labels=labels, domains=domains)
    
    # Compute Ricci curvature
    print("Computing Ricci curvature...")
    ricci_curvatures = analyzer.compute_ricci_curvature()
    
    # Compute scalar curvature
    print("Computing scalar curvature...")
    scalar_curvature, ricci_curvatures = analyzer.compute_scalar_curvature()
    
    # Compute intrinsic dimensions
    print("Computing intrinsic dimensions...")
    intrinsic_dims = analyzer.compute_intrinsic_dimension()
    
    # Analyze stratum boundaries
    print("Analyzing stratum boundaries...")
    boundary_analysis = analyzer.analyze_stratum_boundaries()
    
    # Analyze stratum evolution
    print("Analyzing stratum evolution...")
    stratum_analysis = analyzer.analyze_stratum_evolution()
    
    # Analyze manifold topology
    print("Analyzing manifold topology...")
    topology_analysis = analyzer.analyze_manifold_topology()
    
    # Print results
    print(f"\nDeep Geometric Analysis Results:")
    print(f"  Scalar Curvature: {scalar_curvature:.4f}")
    print(f"  Mean Ricci Curvature: {np.mean(ricci_curvatures):.4f} ± {np.std(ricci_curvatures):.4f}")
    print(f"  Mean Intrinsic Dimension: {np.mean(intrinsic_dims):.4f} ± {np.std(intrinsic_dims):.4f}")
    
    # Boundary analysis
    boundary_points = [i for i in range(len(boundary_analysis)) if boundary_analysis[i]['is_boundary']]
    print(f"  Boundary Points: {len(boundary_points)} / {len(boundary_analysis)} ({len(boundary_points)/len(boundary_analysis)*100:.1f}%)")
    
    # Stratum analysis
    print(f"\nStratum Analysis:")
    for stratum, analysis in stratum_analysis.items():
        print(f"  Stratum {stratum}:")
        print(f"    Points: {analysis['n_points']}")
        print(f"    Ricci Curvature: {analysis['curvature_mean']:.4f} ± {analysis['curvature_std']:.4f}")
        print(f"    Intrinsic Dimension: {analysis['intrinsic_dim_mean']:.4f} ± {analysis['intrinsic_dim_std']:.4f}")
        print(f"    Density: {analysis['density_mean']:.4f} ± {analysis['density_std']:.4f}")
    
    # Topology analysis
    print(f"\nTopology Analysis:")
    connectivity_thresholds = list(topology_analysis.keys())
    connectivity_ratios = [topology_analysis[t]['connectivity_ratio'] for t in connectivity_thresholds]
    
    print(f"  Connectivity Range: {min(connectivity_ratios):.3f} - {max(connectivity_ratios):.3f}")
    print(f"  Peak Connectivity: {max(connectivity_ratios):.3f} at threshold {connectivity_thresholds[np.argmax(connectivity_ratios)]:.3f}")
    
    # Create visualization
    print("\nCreating deep analysis visualization...")
    analyzer.visualize_deep_analysis(save_path=save_path)
    
    return {
        'ricci_curvatures': ricci_curvatures,
        'scalar_curvature': scalar_curvature,
        'intrinsic_dims': intrinsic_dims,
        'boundary_analysis': boundary_analysis,
        'stratum_analysis': stratum_analysis,
        'topology_analysis': topology_analysis
    }

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
    
    # Run deep analysis
    results = run_deep_geometric_analysis(data, strata)
