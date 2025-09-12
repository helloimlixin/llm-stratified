"""Distance analysis utilities for fiber bundle hypothesis testing."""

import numpy as np
from scipy.spatial import distance
from typing import Tuple


class DistanceAnalyzer:
    """Analyze distances between token embeddings."""
    
    @staticmethod
    def compute_distances(embeddings: np.ndarray, token_idx: int) -> np.ndarray:
        """
        Compute Euclidean distances from a token to all other tokens.
        
        Args:
            embeddings: Array of shape (n_tokens, embedding_dim)
            token_idx: Index of the reference token
            
        Returns:
            Sorted array of distances in ascending order
        """
        x = embeddings[token_idx]
        distances = distance.cdist([x], embeddings, 'euclidean')[0]
        return np.sort(distances)
    
    @staticmethod
    def compute_nx_r(distances: np.ndarray, r_values: np.ndarray) -> np.ndarray:
        """
        Compute N_x(r) - number of tokens within radius r.
        
        Args:
            distances: Sorted distances from reference token
            r_values: Array of radius values to evaluate
            
        Returns:
            Array of counts for each radius value
        """
        return np.array([np.sum(distances <= r) for r in r_values])
    
    @staticmethod
    def prepare_log_data(r_values: np.ndarray, nx_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare logarithmic data for slope analysis.
        
        Args:
            r_values: Array of radius values
            nx_r: Array of neighbor counts
            
        Returns:
            Tuple of (log_r, log_nx_r) arrays
        """
        log_r = np.log(r_values)
        log_nx_r = np.log(nx_r + 1e-10)  # Avoid log(0)
        return log_r, log_nx_r
