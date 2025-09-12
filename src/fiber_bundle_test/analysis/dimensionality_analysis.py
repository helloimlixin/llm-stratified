"""Dimensionality analysis for stratified manifold learning."""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.decomposition import PCA
import logging

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

logger = logging.getLogger(__name__)


class DimensionalityAnalyzer:
    """Analyze intrinsic dimensionality of embeddings and clusters."""
    
    def __init__(self, variance_threshold: float = 0.75):
        """
        Initialize dimensionality analyzer.
        
        Args:
            variance_threshold: Threshold for explained variance
        """
        self.variance_threshold = variance_threshold
    
    def compute_intrinsic_dimensions(self, embeddings: np.ndarray, 
                                   strata: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute intrinsic dimensions for embeddings and strata.
        
        Args:
            embeddings: Embedding matrix
            strata: Cluster/strata labels (optional)
            
        Returns:
            Dictionary with intrinsic dimension analysis
        """
        results = {}
        
        # Overall intrinsic dimension
        overall_dim = self._compute_pca_dimension(embeddings)
        results['overall_intrinsic_dimension'] = overall_dim
        
        # Per-stratum analysis
        if strata is not None:
            stratum_dims = {}
            unique_strata = np.unique(strata)
            
            for s in unique_strata:
                indices = np.where(strata == s)[0]
                data_cluster = embeddings[indices, :]
                
                if len(data_cluster) < 2:
                    stratum_dims[int(s)] = 1
                else:
                    stratum_dim = self._compute_pca_dimension(data_cluster)
                    stratum_dims[int(s)] = stratum_dim
            
            results['stratum_dimensions'] = stratum_dims
            results['dimension_variance'] = np.var(list(stratum_dims.values()))
            results['dimension_range'] = (min(stratum_dims.values()), max(stratum_dims.values()))
        
        return results
    
    def _compute_pca_dimension(self, data: np.ndarray) -> int:
        """
        Compute intrinsic dimension using PCA.
        
        Args:
            data: Data matrix
            
        Returns:
            Intrinsic dimension
        """
        if len(data) < 2:
            return 1
        
        try:
            pca = PCA()
            pca.fit(data)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            intrinsic_dim = int(np.searchsorted(cumulative_variance, self.variance_threshold) + 1)
            
            # Ensure dimension is reasonable
            intrinsic_dim = min(intrinsic_dim, data.shape[1], len(data) - 1)
            
            return max(1, intrinsic_dim)
            
        except Exception as e:
            logger.warning(f"PCA failed: {e}")
            return 1
    
    def analyze_dimension_progression(self, embeddings: np.ndarray, 
                                    max_components: int = 50) -> Dict[str, Any]:
        """
        Analyze how explained variance progresses with number of components.
        
        Args:
            embeddings: Embedding matrix
            max_components: Maximum number of components to analyze
            
        Returns:
            Dimension progression analysis
        """
        max_components = min(max_components, embeddings.shape[1], embeddings.shape[0] - 1)
        
        pca = PCA(n_components=max_components)
        pca.fit(embeddings)
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Find dimensions for different variance thresholds
        thresholds = [0.5, 0.75, 0.9, 0.95, 0.99]
        threshold_dims = {}
        
        for threshold in thresholds:
            dim = int(np.searchsorted(cumulative_variance, threshold) + 1)
            threshold_dims[threshold] = min(dim, max_components)
        
        return {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': cumulative_variance,
            'threshold_dimensions': threshold_dims,
            'effective_rank': self._compute_effective_rank(pca.explained_variance_ratio_),
            'participation_ratio': self._compute_participation_ratio(pca.explained_variance_ratio_)
        }
    
    def _compute_effective_rank(self, eigenvalues: np.ndarray) -> float:
        """Compute effective rank based on eigenvalue distribution."""
        eigenvalues = eigenvalues[eigenvalues > 0]
        if len(eigenvalues) == 0:
            return 0.0
        
        # Normalize
        eigenvalues = eigenvalues / np.sum(eigenvalues)
        
        # Compute entropy
        entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-12))
        
        return np.exp(entropy)
    
    def _compute_participation_ratio(self, eigenvalues: np.ndarray) -> float:
        """Compute participation ratio."""
        eigenvalues = eigenvalues[eigenvalues > 0]
        if len(eigenvalues) == 0:
            return 0.0
        
        return np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2)
    
    def reduce_dimensionality(self, embeddings: np.ndarray, 
                            target_dim: int = 64,
                            method: str = 'pca') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reduce dimensionality using specified method.
        
        Args:
            embeddings: Input embeddings
            target_dim: Target dimension
            method: Reduction method ('pca', 'umap')
            
        Returns:
            Tuple of (reduced_embeddings, reduction_info)
        """
        target_dim = min(target_dim, embeddings.shape[1], embeddings.shape[0] - 1)
        
        if method == 'pca':
            pca = PCA(n_components=target_dim)
            reduced_embeddings = pca.fit_transform(embeddings)
            
            info = {
                'method': 'pca',
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'total_explained_variance': np.sum(pca.explained_variance_ratio_),
                'components': pca.components_
            }
            
        elif method == 'umap' and HAS_UMAP:
            # Suppress UMAP parallelism warning
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="n_jobs value.*overridden.*by setting random_state")
                umap_reducer = umap.UMAP(n_components=target_dim, random_state=self.random_state, n_jobs=1)
                reduced_embeddings = umap_reducer.fit_transform(embeddings)
            
            info = {
                'method': 'umap',
                'n_neighbors': umap_reducer.n_neighbors,
                'min_dist': umap_reducer.min_dist
            }
            
        else:
            if method == 'umap':
                logger.warning("UMAP not available, falling back to PCA")
            
            # Fallback to PCA
            pca = PCA(n_components=target_dim)
            reduced_embeddings = pca.fit_transform(embeddings)
            
            info = {
                'method': 'pca',
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'total_explained_variance': np.sum(pca.explained_variance_ratio_)
            }
        
        return reduced_embeddings, info


def compute_intrinsic_dimensions(embeddings: np.ndarray, 
                               strata: np.ndarray,
                               variance_threshold: float = 0.75) -> Dict[int, int]:
    """
    Compute intrinsic dimensions for each stratum (notebook compatibility).
    
    Args:
        embeddings: Embedding matrix
        strata: Stratum labels
        variance_threshold: Variance threshold for dimension estimation
        
    Returns:
        Dictionary mapping stratum ID to intrinsic dimension
    """
    analyzer = DimensionalityAnalyzer(variance_threshold)
    results = analyzer.compute_intrinsic_dimensions(embeddings, strata)
    
    return results.get('stratum_dimensions', {})
