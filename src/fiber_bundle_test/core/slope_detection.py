"""Slope change detection for fiber bundle hypothesis testing."""

import numpy as np
from scipy.stats import ttest_ind
from typing import List, Tuple


class SlopeChangeDetector:
    """Detect significant changes in slopes for hypothesis testing."""
    
    @staticmethod
    def estimate_slopes(log_r: np.ndarray, log_nx_r: np.ndarray) -> np.ndarray:
        """
        Estimate slopes using three-point centered difference method.
        
        Args:
            log_r: Logarithmic radius values
            log_nx_r: Logarithmic neighbor counts
            
        Returns:
            Array of estimated slopes
        """
        slopes = np.zeros(len(log_r))
        
        # Central difference for interior points
        for i in range(1, len(log_r) - 1):
            slopes[i] = (log_nx_r[i + 1] - log_nx_r[i - 1]) / (log_r[i + 1] - log_r[i - 1])
        
        # Extrapolate edge values
        slopes[0] = slopes[1]
        slopes[-1] = slopes[-2]
        
        return slopes
    
    @staticmethod
    def detect_slope_changes(slopes: np.ndarray, 
                           window_size: int = 10, 
                           alpha: float = 0.05) -> Tuple[List[int], List[float]]:
        """
        Detect significant slope changes using statistical testing.
        
        Args:
            slopes: Array of slope values
            window_size: Size of the sliding window for comparison
            alpha: Significance level for statistical tests
            
        Returns:
            Tuple of (change_indices, p_values) for detected changes
        """
        changes = []
        p_values = []
        
        for i in range(window_size, len(slopes) - window_size):
            before = slopes[i - window_size:i]
            after = slopes[i:i + window_size]
            
            # Perform independent samples t-test
            t_stat, p_val = ttest_ind(before, after, equal_var=False)
            
            # Check for significant increase in slope
            if p_val < alpha and np.mean(after) > np.mean(before):
                changes.append(i)
                p_values.append(p_val)
        
        return changes, p_values
    
    @staticmethod
    def estimate_dimensions(slopes: np.ndarray, 
                          change_idx: int, 
                          window_size: int = 10) -> Tuple[float, float]:
        """
        Estimate base and fiber dimensions from slope data.
        
        Args:
            slopes: Array of slope values
            change_idx: Index of the detected change point
            window_size: Window size for dimension estimation
            
        Returns:
            Tuple of (base_dimension, fiber_dimension)
        """
        # Base dimension: mean slope before change point
        base_dim = np.mean(slopes[:change_idx]) if change_idx > 0 else 0
        
        # Fiber dimension: mean slope in window starting at change point
        end_idx = min(change_idx + window_size, len(slopes))
        fiber_dim = np.mean(slopes[change_idx:end_idx])
        
        return base_dim, fiber_dim
