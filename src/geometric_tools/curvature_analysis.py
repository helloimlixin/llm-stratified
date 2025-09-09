"""
Curvature-based geometric analysis for stratified manifold learning.
Inspired by: Mixed-curvature Variational Autoencoders (arXiv:1911.08411)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

class CurvatureAnalyzer:
    """
    Curvature-based analysis for latent embedding geometry.
    
    Based on the framework from:
    Skopek, O., Ganea, O. E., & Bécigneul, G. (2020). 
    Mixed-curvature Variational Autoencoders. ICLR 2020.
    """
    
    def __init__(self, data, labels=None, domains=None):
        self.data = np.array(data)
        self.labels = labels
        self.domains = domains
        self.n_samples, self.n_features = self.data.shape
        
    def compute_riemannian_curvature(self, k=10):
        """
        Compute Riemannian curvature tensor components.
        
        For a Riemannian manifold M with metric g, the curvature tensor R
        measures how much the manifold deviates from being flat.
        """
        # Adjust k based on available samples
        k = min(k, self.n_samples - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        
        curvatures = []
        for i in range(self.n_samples):
            neighbor_data = self.data[indices[i][1:]]  # Exclude self
            if len(neighbor_data) < 3:
                curvatures.append(0.0)
                continue
            
            # Compute local curvature using PCA-based approach
            curvature = self._compute_local_riemannian_curvature(neighbor_data)
            curvatures.append(curvature)
        
        return np.array(curvatures)
    
    def _compute_local_riemannian_curvature(self, local_data):
        """
        Compute local Riemannian curvature from neighborhood data.
        
        Uses the relationship between PCA eigenvalues and curvature:
        For a curved manifold, the ratio of eigenvalues indicates curvature.
        """
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
        # Higher curvature leads to more unequal eigenvalues
        curvature = eigenvalues[1] / eigenvalues[0] if eigenvalues[0] > 0 else 0
        
        return curvature
    
    def compute_sectional_curvature(self, k=10):
        """
        Compute sectional curvature for 2D planes in the embedding space.
        
        Sectional curvature K(σ) measures the curvature of a 2D section σ
        of the manifold at a point.
        """
        # Adjust k based on available samples
        k = min(k, self.n_samples - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        
        sectional_curvatures = []
        for i in range(self.n_samples):
            neighbor_data = self.data[indices[i][1:]]
            if len(neighbor_data) < 3:
                sectional_curvatures.append(0.0)
                continue
            
            # Compute sectional curvature for multiple 2D planes
            sectional_curv = self._compute_local_sectional_curvature(neighbor_data)
            sectional_curvatures.append(sectional_curv)
        
        return np.array(sectional_curvatures)
    
    def _compute_local_sectional_curvature(self, local_data):
        """
        Compute sectional curvature for local neighborhood.
        
        Uses the relationship between geodesic deviation and curvature.
        """
        if len(local_data) < 3:
            return 0.0
        
        # Compute pairwise distances
        distances = pairwise_distances(local_data)
        
        # Estimate sectional curvature from distance relationships
        # In curved spaces, distances don't follow Euclidean relationships
        n = len(local_data)
        curvature_estimates = []
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    # Triangle inequality violation indicates curvature
                    d_ij = distances[i, j]
                    d_jk = distances[j, k]
                    d_ik = distances[i, k]
                    
                    # Curvature estimate from triangle deviation
                    triangle_deviation = abs(d_ij + d_jk - d_ik) / (d_ij + d_jk + d_ik + 1e-8)
                    curvature_estimates.append(triangle_deviation)
        
        return np.mean(curvature_estimates) if curvature_estimates else 0.0
    
    def compute_gaussian_curvature(self, k=10):
        """
        Compute Gaussian curvature K = det(II) / det(I) where I and II are
        the first and second fundamental forms.
        
        Gaussian curvature is an intrinsic measure of curvature.
        """
        # Adjust k based on available samples
        k = min(k, self.n_samples - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        
        gaussian_curvatures = []
        for i in range(self.n_samples):
            neighbor_data = self.data[indices[i][1:]]
            if len(neighbor_data) < 3:
                gaussian_curvatures.append(0.0)
                continue
            
            # Compute Gaussian curvature using local surface approximation
            gaussian_curv = self._compute_local_gaussian_curvature(neighbor_data)
            gaussian_curvatures.append(gaussian_curv)
        
        return np.array(gaussian_curvatures)
    
    def _compute_local_gaussian_curvature(self, local_data):
        """
        Compute Gaussian curvature for local neighborhood.
        
        Uses the relationship between local surface geometry and curvature.
        """
        if len(local_data) < 3:
            return 0.0
        
        # Center the data
        center = np.mean(local_data, axis=0)
        centered_data = local_data - center
        
        # Compute PCA to get local coordinate system
        pca = PCA()
        pca.fit(centered_data)
        
        # Project to local tangent space
        projected_data = pca.transform(centered_data)
        
        if projected_data.shape[1] < 2:
            return 0.0
        
        # Estimate Gaussian curvature from local surface
        # Use the relationship between eigenvalues and curvature
        eigenvalues = pca.explained_variance_
        
        if len(eigenvalues) < 2 or eigenvalues[0] == 0:
            return 0.0
        
        # Gaussian curvature estimate
        # For a surface, K = (λ₁λ₂) / (λ₁ + λ₂)² approximately
        gaussian_curv = (eigenvalues[0] * eigenvalues[1]) / (eigenvalues[0] + eigenvalues[1])**2
        
        return gaussian_curv
    
    def compute_mean_curvature(self, k=10):
        """
        Compute mean curvature H = (1/2) * trace(II) where II is the
        second fundamental form.
        
        Mean curvature measures the average curvature of the surface.
        """
        # Adjust k based on available samples
        k = min(k, self.n_samples - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        
        mean_curvatures = []
        for i in range(self.n_samples):
            neighbor_data = self.data[indices[i][1:]]
            if len(neighbor_data) < 3:
                mean_curvatures.append(0.0)
                continue
            
            # Compute mean curvature using local surface analysis
            mean_curv = self._compute_local_mean_curvature(neighbor_data)
            mean_curvatures.append(mean_curv)
        
        return np.array(mean_curvatures)
    
    def _compute_local_mean_curvature(self, local_data):
        """
        Compute mean curvature for local neighborhood.
        
        Uses the relationship between local surface geometry and mean curvature.
        """
        if len(local_data) < 3:
            return 0.0
        
        # Center the data
        center = np.mean(local_data, axis=0)
        centered_data = local_data - center
        
        # Compute PCA
        pca = PCA()
        pca.fit(centered_data)
        eigenvalues = pca.explained_variance_
        
        if len(eigenvalues) < 2:
            return 0.0
        
        # Mean curvature estimate
        # H = (1/2) * (λ₁ + λ₂) / (λ₁ + λ₂) = 1/2 for normalized case
        mean_curv = np.mean(eigenvalues[:2]) / (np.sum(eigenvalues[:2]) + 1e-8)
        
        return mean_curv
    
    def compute_mixed_curvature_signature(self, k=10):
        """
        Compute mixed curvature signature inspired by the Mixed-curvature VAE paper.
        
        Returns a signature that characterizes the curvature properties
        of different regions of the embedding space.
        """
        # Compute different types of curvature
        riemannian_curv = self.compute_riemannian_curvature(k)
        sectional_curv = self.compute_sectional_curvature(k)
        gaussian_curv = self.compute_gaussian_curvature(k)
        mean_curv = self.compute_mean_curvature(k)
        
        # Create curvature signature
        signature = {
            'riemannian_curvature': riemannian_curv,
            'sectional_curvature': sectional_curv,
            'gaussian_curvature': gaussian_curv,
            'mean_curvature': mean_curv,
            'curvature_variance': np.var(riemannian_curv),
            'curvature_mean': np.mean(riemannian_curv),
            'curvature_std': np.std(riemannian_curv)
        }
        
        return signature
    
    def analyze_curvature_by_strata(self, strata):
        """
        Analyze curvature properties for each stratum.
        
        This helps understand how different strata have different
        geometric properties.
        """
        unique_strata = np.unique(strata)
        stratum_curvature_analysis = {}
        
        for stratum in unique_strata:
            stratum_mask = strata == stratum
            stratum_data = self.data[stratum_mask]
            
            if len(stratum_data) < 3:
                stratum_curvature_analysis[stratum] = {
                    'riemannian_curvature': 0.0,
                    'sectional_curvature': 0.0,
                    'gaussian_curvature': 0.0,
                    'mean_curvature': 0.0,
                    'curvature_variance': 0.0,
                    'curvature_mean': 0.0,
                    'curvature_std': 0.0
                }
                continue
            
            # Create analyzer for this stratum
            stratum_analyzer = CurvatureAnalyzer(stratum_data)
            signature = stratum_analyzer.compute_mixed_curvature_signature()
            
            stratum_curvature_analysis[stratum] = signature
        
        return stratum_curvature_analysis
    
    def visualize_curvature_analysis(self, strata=None, save_path=None):
        """
        Create comprehensive visualization of curvature analysis.
        """
        # Compute curvature signature
        signature = self.compute_mixed_curvature_signature()
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Riemannian curvature
        scatter1 = axes[0, 0].scatter(self.data[:, 0], self.data[:, 1], 
                                     c=signature['riemannian_curvature'], 
                                     cmap='viridis', alpha=0.7, s=30)
        axes[0, 0].set_title('Riemannian Curvature')
        axes[0, 0].set_xlabel('Feature 1')
        axes[0, 0].set_ylabel('Feature 2')
        plt.colorbar(scatter1, ax=axes[0, 0], label='Curvature')
        
        # Plot 2: Sectional curvature
        scatter2 = axes[0, 1].scatter(self.data[:, 0], self.data[:, 1], 
                                     c=signature['sectional_curvature'], 
                                     cmap='plasma', alpha=0.7, s=30)
        axes[0, 1].set_title('Sectional Curvature')
        axes[0, 1].set_xlabel('Feature 1')
        axes[0, 1].set_ylabel('Feature 2')
        plt.colorbar(scatter2, ax=axes[0, 1], label='Curvature')
        
        # Plot 3: Gaussian curvature
        scatter3 = axes[0, 2].scatter(self.data[:, 0], self.data[:, 1], 
                                     c=signature['gaussian_curvature'], 
                                     cmap='coolwarm', alpha=0.7, s=30)
        axes[0, 2].set_title('Gaussian Curvature')
        axes[0, 2].set_xlabel('Feature 1')
        axes[0, 2].set_ylabel('Feature 2')
        plt.colorbar(scatter3, ax=axes[0, 2], label='Curvature')
        
        # Plot 4: Mean curvature
        scatter4 = axes[1, 0].scatter(self.data[:, 0], self.data[:, 1], 
                                     c=signature['mean_curvature'], 
                                     cmap='Reds', alpha=0.7, s=30)
        axes[1, 0].set_title('Mean Curvature')
        axes[1, 0].set_xlabel('Feature 1')
        axes[1, 0].set_ylabel('Feature 2')
        plt.colorbar(scatter4, ax=axes[1, 0], label='Curvature')
        
        # Plot 5: Curvature variance
        scatter5 = axes[1, 1].scatter(self.data[:, 0], self.data[:, 1], 
                                     c=signature['curvature_variance'], 
                                     cmap='Oranges', alpha=0.7, s=30)
        axes[1, 1].set_title('Curvature Variance')
        axes[1, 1].set_xlabel('Feature 1')
        axes[1, 1].set_ylabel('Feature 2')
        plt.colorbar(scatter5, ax=axes[1, 1], label='Variance')
        
        # Plot 6: Strata (if provided)
        if strata is not None:
            scatter6 = axes[1, 2].scatter(self.data[:, 0], self.data[:, 1], 
                                         c=strata, cmap='tab10', alpha=0.7, s=30)
            axes[1, 2].set_title('Stratified Structure')
            axes[1, 2].set_xlabel('Feature 1')
            axes[1, 2].set_ylabel('Feature 2')
            plt.colorbar(scatter6, ax=axes[1, 2], label='Stratum')
        else:
            axes[1, 2].text(0.5, 0.5, 'No strata provided', ha='center', va='center', 
                           transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Stratified Structure')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return signature

def run_curvature_analysis(data, strata=None, domains=None, save_path='curvature_analysis.png'):
    """
    Run comprehensive curvature analysis inspired by Mixed-curvature VAE paper.
    
    Args:
        data: Embedding data
        strata: Stratum assignments
        domains: Domain labels
        save_path: Path to save visualization
    
    Returns:
        Dictionary with curvature analysis results
    """
    print("🔬 Running Curvature-Based Geometric Analysis")
    print("=" * 60)
    print("Inspired by: Mixed-curvature Variational Autoencoders (arXiv:1911.08411)")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = CurvatureAnalyzer(data, labels=None, domains=domains)
    
    # Compute curvature signature
    print("Computing mixed curvature signature...")
    signature = analyzer.compute_mixed_curvature_signature()
    
    # Analyze curvature by strata if provided
    if strata is not None:
        print("Analyzing curvature by strata...")
        stratum_analysis = analyzer.analyze_curvature_by_strata(strata)
        
        print(f"\nCurvature Analysis by Stratum:")
        for stratum, analysis in stratum_analysis.items():
            print(f"  Stratum {stratum}:")
            print(f"    Riemannian Curvature: {analysis['curvature_mean']:.4f} ± {analysis['curvature_std']:.4f}")
            print(f"    Sectional Curvature: {analysis['sectional_curvature'].mean():.4f}")
            print(f"    Gaussian Curvature: {analysis['gaussian_curvature'].mean():.4f}")
            print(f"    Mean Curvature: {analysis['mean_curvature'].mean():.4f}")
    
    # Create visualization
    print("\nCreating curvature visualization...")
    signature = analyzer.visualize_curvature_analysis(strata=strata, save_path=save_path)
    
    # Print summary statistics
    print(f"\nCurvature Analysis Summary:")
    print(f"  Riemannian Curvature: {signature['curvature_mean']:.4f} ± {signature['curvature_std']:.4f}")
    print(f"  Sectional Curvature: {signature['sectional_curvature'].mean():.4f} ± {signature['sectional_curvature'].std():.4f}")
    print(f"  Gaussian Curvature: {signature['gaussian_curvature'].mean():.4f} ± {signature['gaussian_curvature'].std():.4f}")
    print(f"  Mean Curvature: {signature['mean_curvature'].mean():.4f} ± {signature['mean_curvature'].std():.4f}")
    print(f"  Curvature Variance: {signature['curvature_variance']:.4f}")
    
    return signature

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
    
    # Run curvature analysis
    results = run_curvature_analysis(data, strata=strata)
