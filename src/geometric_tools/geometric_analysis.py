"""
Geometric Analysis Tools for Stratified Manifold Learning

This module contains advanced geometric analysis tools including local dimension estimation,
manifold intersection detection, geometric harmonic analysis, and Riemannian metric learning.
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# Optional imports for advanced geometric analysis
try:
    from ripser import ripser
    from persim import plot_diagrams
    import gudhi as gd
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False
    warnings.warn("TDA libraries (ripser, persim, gudhi) not available. Some functions will be limited.")


class LocalDimensionEstimator:
    """Estimates local intrinsic dimension using multiple methods."""

    def __init__(self, n_neighbors=15, max_dim=100):
        """
        Initialize dimension estimator.
        
        Args:
            n_neighbors: Number of neighbors for local analysis
            max_dim: Maximum dimension to consider
        """
        self.n_neighbors = n_neighbors
        self.max_dim = max_dim

    def _mle_dimension(self, X, local_region):
        """
        Maximum Likelihood Estimation of local intrinsic dimension.
        Uses local_region for neighborhood statistics.
        
        Args:
            X: Data points to estimate dimension for
            local_region: Reference dataset for neighborhood
            
        Returns:
            Tuple of (mean_dimension, std_dimension)
        """
        if len(local_region) < 2:
            return 1.0, 0.0

        k = min(self.n_neighbors, len(local_region) - 1)

        # Get k nearest neighbors using local_region
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(local_region)
        distances, _ = nbrs.kneighbors(X)
        distances = distances[:, 1:]  # Exclude self-distance

        # Compute MLE estimate
        log_dists = np.log(distances / distances[:, -1:][:, np.newaxis])
        inv_mle = np.mean(log_dists, axis=1)
        mle_dims = -1 / inv_mle

        return np.mean(mle_dims), np.std(mle_dims)

    def correlation_dimension(self, X, eps_range=None):
        """
        Estimates correlation dimension using point-wise correlation sum.
        
        Args:
            X: Data points
            eps_range: Range of epsilon values for correlation sum
            
        Returns:
            Estimated correlation dimension
        """
        if eps_range is None:
            dists = pdist(X)
            eps_range = np.logspace(np.log10(np.min(dists)), np.log10(np.max(dists)), 20)

        dists = squareform(pdist(X))
        N = len(X)

        correlation_sums = []
        for eps in eps_range:
            correlation_sum = np.sum(dists < eps) / (N * (N-1))
            correlation_sums.append(correlation_sum)

        # Estimate dimension from slope
        log_eps = np.log(eps_range)
        log_corr = np.log(correlation_sums)
        slope = np.polyfit(log_eps, log_corr, 1)[0]

        return slope

    def tda_dimension(self, X):
        """
        Estimates dimension using persistent homology.
        
        Args:
            X: Data points
            
        Returns:
            Estimated dimension from TDA
        """
        if not TDA_AVAILABLE:
            warnings.warn("TDA libraries not available, returning MLE estimate")
            return self._mle_dimension(X, X)[0]
            
        # Compute persistent homology
        diagrams = ripser(X)['dgms']

        # Analyze persistence lifetimes
        lifetimes = []
        for dim, diagram in enumerate(diagrams):
            if len(diagram) > 0:
                lifetime = diagram[:, 1] - diagram[:, 0]
                lifetimes.append(np.mean(lifetime[np.isfinite(lifetime)]))

        # Estimate dimension from lifetime decay
        if len(lifetimes) > 1:
            decay_rate = -np.polyfit(range(len(lifetimes)), np.log(lifetimes), 1)[0]
            return decay_rate
        return 1

    def estimate_dimension(self, X, local_region=None):
        """
        Combines multiple dimension estimates.

        Args:
            X: Data point(s) to estimate dimension for
            local_region: Optional reference dataset for local neighborhood
            
        Returns:
            Dictionary with various dimension estimates
        """
        if local_region is None:
            if len(X) < self.n_neighbors + 1:
                # If we don't have enough points, return minimum dimension
                return {
                    'dimension': 1.0,
                    'mle': 1.0,
                    'mle_std': 0.0,
                    'correlation': 1.0,
                    'tda': 1.0
                }
            local_region = X

        # Compute MLE dimension using local region
        mle_dim, mle_std = self._mle_dimension(X, local_region)

        # Compute correlation dimension if we have enough points
        if len(local_region) > self.n_neighbors:
            corr_dim = self.correlation_dimension(local_region)
            tda_dim = self.tda_dimension(local_region)
        else:
            corr_dim = mle_dim
            tda_dim = mle_dim

        # Weighted average based on confidence/stability
        weights = [1.0, 0.8, 0.6]  # Weights for MLE, correlation, and TDA
        dims = np.array([mle_dim, corr_dim, tda_dim])
        weighted_dim = np.average(dims, weights=weights)

        return {
            'dimension': weighted_dim,
            'mle': mle_dim,
            'mle_std': mle_std,
            'correlation': corr_dim,
            'tda': tda_dim
        }


class ManifoldIntersectionDetector:
    """Detects and characterizes intersections between manifolds."""

    def __init__(self, n_neighbors=15, min_pts=5):
        """
        Initialize intersection detector.
        
        Args:
            n_neighbors: Number of neighbors for local analysis
            min_pts: Minimum points required for manifold detection
        """
        self.n_neighbors = n_neighbors
        self.min_pts = min_pts

    def compute_local_tangent_space(self, X, point_idx):
        """
        Computes local tangent space using PCA on neighborhood.
        
        Args:
            X: Data points
            point_idx: Index of the point
            
        Returns:
            Tuple of (tangent_space_basis, singular_values)
        """
        # Get local neighborhood
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        distances, indices = nbrs.kneighbors(X[point_idx:point_idx+1])
        neighborhood = X[indices[0]]

        # Center the neighborhood
        centered = neighborhood - np.mean(neighborhood, axis=0)

        # Compute SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)

        return Vt, S

    def detect_intersection(self, X, labels, point_idx):
        """
        Detects if a point lies in manifold intersection.
        
        Args:
            X: Data points
            labels: Manifold labels
            point_idx: Index of the point
            
        Returns:
            Tuple of (is_intersection, intersection_angles)
        """
        # Get local neighborhood
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        distances, indices = nbrs.kneighbors(X[point_idx:point_idx+1])
        neighborhood_labels = labels[indices[0]]

        # Check for multiple manifolds in neighborhood
        unique_labels = np.unique(neighborhood_labels)
        if len(unique_labels) > 1:
            # Compute tangent spaces for each manifold
            tangent_spaces = []
            for label in unique_labels:
                mask = neighborhood_labels == label
                if np.sum(mask) >= self.min_pts:
                    Vt, S = self.compute_local_tangent_space(X[indices[0][mask]], 0)
                    tangent_spaces.append((Vt, S))

            # Analyze intersection angles
            if len(tangent_spaces) > 1:
                angles = self._compute_intersection_angles(tangent_spaces)
                return True, angles

        return False, None

    def _compute_intersection_angles(self, tangent_spaces):
        """
        Computes principal angles between tangent spaces.
        
        Args:
            tangent_spaces: List of tangent space bases
            
        Returns:
            List of principal angles
        """
        angles = []
        for i in range(len(tangent_spaces)):
            for j in range(i+1, len(tangent_spaces)):
                V1, _ = tangent_spaces[i]
                V2, _ = tangent_spaces[j]

                # Compute principal angles
                M = V1 @ V2.T
                S = np.linalg.svd(M, compute_uv=False)
                angles.append(np.arccos(np.clip(S, -1, 1)))

        return angles


class GeometricHarmonicAnalysis:
    """
    Implements geometric harmonic analysis for manifold learning.
    """

    def __init__(self, n_eigenvectors=50, n_neighbors=15):
        """
        Initialize harmonic analyzer.
        
        Args:
            n_eigenvectors: Number of eigenvectors to compute
            n_neighbors: Number of neighbors for graph construction
        """
        self.n_eigenvectors = n_eigenvectors
        self.n_neighbors = n_neighbors

    def compute_laplacian_eigenmaps(self, X):
        """
        Computes Laplacian eigenmaps of the data.
        
        Args:
            X: Data points
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        # Construct adjacency matrix
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        adj_matrix = nbrs.kneighbors_graph(X, mode='distance')
        adj_matrix = np.array(adj_matrix.todense())

        # Gaussian kernel
        sigma = np.mean(adj_matrix[adj_matrix > 0])
        W = np.exp(-adj_matrix**2 / (2 * sigma**2))

        # Compute normalized Laplacian
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        D_norm = np.diag(1.0 / np.sqrt(np.diag(D)))
        L_norm = D_norm @ L @ D_norm

        # Compute eigenvectors
        eigenvalues, eigenvectors = eigsh(L_norm, k=self.n_eigenvectors,
                                        which='SM', sigma=1e-8)

        return eigenvalues, eigenvectors

    def detect_spectral_gaps(self, eigenvalues):
        """
        Detects spectral gaps to determine manifold structure.
        
        Args:
            eigenvalues: Eigenvalues from Laplacian
            
        Returns:
            Indices of significant gaps
        """
        gaps = np.diff(eigenvalues)
        significant_gaps = np.where(gaps > np.mean(gaps) + 2*np.std(gaps))[0]
        return significant_gaps

    def compute_heat_kernel_signature(self, X, t_range=None):
        """
        Computes heat kernel signatures for shape analysis.
        
        Args:
            X: Data points
            t_range: Range of time parameters
            
        Returns:
            Heat kernel signatures
        """
        if t_range is None:
            t_range = np.logspace(-2, 2, 50)

        eigenvalues, eigenvectors = self.compute_laplacian_eigenmaps(X)

        hks = np.zeros((len(X), len(t_range)))
        for i, t in enumerate(t_range):
            hks[:, i] = np.sum(eigenvectors**2 * np.exp(-eigenvalues * t), axis=1)

        return hks


class RiemannianMetricLearner:
    """
    Learns local Riemannian metrics for the manifold.
    """

    def __init__(self, n_neighbors=15, reg_param=1e-8):
        """
        Initialize metric learner.
        
        Args:
            n_neighbors: Number of neighbors for local analysis
            reg_param: Regularization parameter
        """
        self.n_neighbors = n_neighbors
        self.reg_param = reg_param

    def fit_local_metric(self, X, point_idx):
        """
        Fits a local Riemannian metric at a point.
        
        Args:
            X: Data points
            point_idx: Index of the point
            
        Returns:
            Local Riemannian metric matrix
        """
        # Get local neighborhood
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        distances, indices = nbrs.kneighbors(X[point_idx:point_idx+1])
        neighborhood = X[indices[0]]

        # Center and compute local covariance
        centered = neighborhood - np.mean(neighborhood, axis=0)
        cov = centered.T @ centered / len(centered)

        # Regularize and invert
        metric = np.linalg.inv(cov + self.reg_param * np.eye(cov.shape[0]))

        return metric

    def parallel_transport(self, metric1, metric2, tangent_vector):
        """
        Parallel transports a tangent vector between metrics.
        
        Args:
            metric1: Source metric
            metric2: Target metric
            tangent_vector: Vector to transport
            
        Returns:
            Transported vector
        """
        # Compute transformation matrix
        transform = np.linalg.cholesky(metric2 @ np.linalg.inv(metric1))

        # Transport the vector
        transported = transform @ tangent_vector

        return transported


class GeometricDictionaryLearning(nn.Module):
    """
    Geometric dictionary learning with manifold constraints.
    """

    def __init__(self, input_dim, dict_size, manifold_dim, n_neighbors=15):
        """
        Initialize geometric dictionary learning.
        
        Args:
            input_dim: Input dimension
            dict_size: Dictionary size
            manifold_dim: Manifold dimension
            n_neighbors: Number of neighbors for manifold constraints
        """
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.manifold_dim = manifold_dim
        self.n_neighbors = n_neighbors
        
        # Dictionary atoms
        self.dictionary = nn.Parameter(torch.randn(input_dim, dict_size) * 0.01)
        
        # Manifold projection
        self.manifold_proj = nn.Linear(input_dim, manifold_dim)
        self.manifold_recon = nn.Linear(manifold_dim, input_dim)

    def forward(self, x):
        """
        Forward pass of geometric dictionary learning.
        
        Args:
            x: Input data
            
        Returns:
            Tuple of (reconstruction, sparse_codes, manifold_projection)
        """
        # Project to manifold
        manifold_x = self.manifold_proj(x)
        
        # Compute sparse codes on manifold
        codes = self._compute_sparse_codes(manifold_x)
        
        # Reconstruct from manifold
        manifold_recon = self.manifold_recon(manifold_x)
        
        # Reconstruct from dictionary
        dict_recon = torch.matmul(codes, self.dictionary.T)
        
        return dict_recon, codes, manifold_recon

    def _compute_sparse_codes(self, x):
        """
        Compute sparse codes using geometric constraints.
        
        Args:
            x: Manifold-projected data
            
        Returns:
            Sparse codes
        """
        # Simple sparse coding (can be enhanced with geometric constraints)
        codes = torch.matmul(x, self.dictionary)
        codes = F.relu(codes)  # Non-negative constraint
        
        return codes


class StratifiedManifoldLearner:
    """
    Learns stratified manifold structure from data.
    """

    def __init__(self, n_strata=5, n_neighbors=15, reg_param=1e-8):
        """
        Initialize stratified manifold learner.
        
        Args:
            n_strata: Number of strata
            n_neighbors: Number of neighbors for local analysis
            reg_param: Regularization parameter
        """
        self.n_strata = n_strata
        self.n_neighbors = n_neighbors
        self.reg_param = reg_param
        
        # Initialize analyzers
        self.dim_estimator = LocalDimensionEstimator(n_neighbors=n_neighbors)
        self.intersection_detector = ManifoldIntersectionDetector(n_neighbors=n_neighbors)
        self.harmonic_analyzer = GeometricHarmonicAnalysis(n_neighbors=n_neighbors)
        self.metric_learner = RiemannianMetricLearner(n_neighbors=n_neighbors, reg_param=reg_param)

    def learn_stratified_structure(self, X, labels=None):
        """
        Learn stratified manifold structure from data.
        
        Args:
            X: Data points
            labels: Optional stratum labels
            
        Returns:
            Dictionary containing learned structure
        """
        results = {}
        
        # Estimate local dimensions
        print("Estimating local dimensions...")
        local_dims = []
        for i in range(len(X)):
            dim_info = self.dim_estimator.estimate_dimension(X[i:i+1], X)
            local_dims.append(dim_info['dimension'])
        results['local_dimensions'] = np.array(local_dims)
        
        # Detect intersections if labels provided
        if labels is not None:
            print("Detecting manifold intersections...")
            intersections = []
            intersection_angles = []
            for i in range(len(X)):
                is_intersection, angles = self.intersection_detector.detect_intersection(X, labels, i)
                intersections.append(is_intersection)
                if angles is not None:
                    intersection_angles.extend(angles)
            results['intersections'] = np.array(intersections)
            results['intersection_angles'] = np.array(intersection_angles)
        
        # Compute harmonic analysis
        print("Computing harmonic analysis...")
        eigenvalues, eigenvectors = self.harmonic_analyzer.compute_laplacian_eigenmaps(X)
        spectral_gaps = self.harmonic_analyzer.detect_spectral_gaps(eigenvalues)
        results['eigenvalues'] = eigenvalues
        results['eigenvectors'] = eigenvectors
        results['spectral_gaps'] = spectral_gaps
        
        # Compute heat kernel signatures
        print("Computing heat kernel signatures...")
        hks = self.harmonic_analyzer.compute_heat_kernel_signature(X)
        results['heat_kernel_signatures'] = hks
        
        return results

    def analyze_stratum_properties(self, X, stratum_labels):
        """
        Analyze properties of each stratum.
        
        Args:
            X: Data points
            stratum_labels: Stratum labels
            
        Returns:
            Dictionary with stratum properties
        """
        stratum_properties = {}
        
        for stratum_id in np.unique(stratum_labels):
            mask = stratum_labels == stratum_id
            stratum_data = X[mask]
            
            if len(stratum_data) < 2:
                continue
                
            # Estimate dimension
            dim_info = self.dim_estimator.estimate_dimension(stratum_data)
            
            # Compute local metrics
            metrics = []
            for i in range(min(10, len(stratum_data))):  # Sample for efficiency
                metric = self.metric_learner.fit_local_metric(stratum_data, i)
                metrics.append(metric)
            
            stratum_properties[stratum_id] = {
                'dimension': dim_info,
                'size': len(stratum_data),
                'metrics': metrics,
                'data': stratum_data
            }
        
        return stratum_properties
