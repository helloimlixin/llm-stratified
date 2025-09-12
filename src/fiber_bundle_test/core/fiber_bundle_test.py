"""Main fiber bundle hypothesis test implementation."""

import sys
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from statsmodels.stats.multitest import multipletests

from .distance_analysis import DistanceAnalyzer
from .slope_detection import SlopeChangeDetector


class FiberBundleTest:
    """Main class for conducting fiber bundle hypothesis tests."""
    
    def __init__(self, 
                 r_min: float = 0.01,
                 r_max: float = 20.0,
                 n_r: int = 200,
                 alpha: float = 0.05,
                 window_size: int = 10):
        """
        Initialize the fiber bundle test.
        
        Args:
            r_min: Minimum radius value
            r_max: Maximum radius value
            n_r: Number of radius values to evaluate
            alpha: Significance level for statistical tests
            window_size: Window size for slope change detection
        """
        self.r_min = r_min
        self.r_max = r_max
        self.n_r = n_r
        self.alpha = alpha
        self.window_size = window_size
        
        # Initialize analyzers
        self.distance_analyzer = DistanceAnalyzer()
        self.slope_detector = SlopeChangeDetector()
    
    def run_test(self, embeddings: np.ndarray, verbose: bool = False) -> Dict[str, Any]:
        """
        Run the complete fiber bundle hypothesis test.
        
        Args:
            embeddings: Array of shape (n_tokens, embedding_dim)
            verbose: Whether to print detailed diagnostics
            
        Returns:
            Dictionary containing test results and statistics
        """
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")
        
        n_tokens = len(embeddings)
        r_values = np.linspace(self.r_min, self.r_max, self.n_r)
        log_r = np.log(r_values)
        
        results = []
        all_p_values = []
        dimensions = []
        
        # Add progress bar for large datasets
        try:
            from tqdm import tqdm
            token_iterator = tqdm(range(n_tokens), desc="Analyzing tokens", 
                                disable=verbose or n_tokens < 20)
        except ImportError:
            token_iterator = range(n_tokens)
            if n_tokens >= 20:
                print(f"Analyzing {n_tokens} tokens...")
        
        for token_idx in token_iterator:
            result = self._test_single_token(
                embeddings, token_idx, r_values, log_r, verbose and token_idx < 3
            )
            
            results.append((token_idx, result['decision']))
            all_p_values.extend(result['p_values'])
            dimensions.append(result['dimensions'])
            
            # Print progress for large datasets without tqdm
            if 'tqdm' not in sys.modules and n_tokens >= 20 and (token_idx + 1) % (n_tokens // 10) == 0:
                progress = (token_idx + 1) / n_tokens * 100
                print(f"Progress: {progress:.1f}% ({token_idx + 1}/{n_tokens})")
        
        # Apply multiple testing correction
        corrected_p_values = []
        if all_p_values:
            corrected_p_values = multipletests(all_p_values, alpha=self.alpha, method='holm')[1]
        
        # Compile final results
        rejections = sum(1 for _, decision in results if decision == "Reject")
        
        return {
            'results': results,
            'dimensions': dimensions,
            'raw_p_values': all_p_values,
            'corrected_p_values': corrected_p_values,
            'total_rejections': rejections,
            'total_tokens': n_tokens,
            'rejection_rate': rejections / n_tokens if n_tokens > 0 else 0,
            'parameters': {
                'r_min': self.r_min,
                'r_max': self.r_max,
                'n_r': self.n_r,
                'alpha': self.alpha,
                'window_size': self.window_size
            }
        }
    
    def _test_single_token(self, 
                          embeddings: np.ndarray,
                          token_idx: int,
                          r_values: np.ndarray,
                          log_r: np.ndarray,
                          verbose: bool = False) -> Dict[str, Any]:
        """
        Test the fiber bundle hypothesis for a single token.
        
        Args:
            embeddings: All token embeddings
            token_idx: Index of the token to test
            r_values: Array of radius values
            log_r: Logarithmic radius values
            verbose: Whether to print diagnostics
            
        Returns:
            Dictionary with test results for this token
        """
        # Compute distances and neighbor counts
        distances = self.distance_analyzer.compute_distances(embeddings, token_idx)
        nx_r = self.distance_analyzer.compute_nx_r(distances, r_values)
        log_r, log_nx_r = self.distance_analyzer.prepare_log_data(r_values, nx_r)
        
        # Estimate slopes and detect changes
        slopes = self.slope_detector.estimate_slopes(log_r, log_nx_r)
        changes, p_vals = self.slope_detector.detect_slope_changes(
            slopes, self.window_size, self.alpha
        )
        
        # Make decision and estimate dimensions
        if changes:
            decision = "Reject"
            change_idx = changes[0]  # Use first detected change
            base_dim, fiber_dim = self.slope_detector.estimate_dimensions(
                slopes, change_idx, self.window_size
            )
            dimensions = (base_dim, fiber_dim)
        else:
            decision = "Fail to Reject"
            dimensions = (None, None)
        
        # Print diagnostics if requested
        if verbose:
            self._print_diagnostics(token_idx, distances, slopes, changes, dimensions)
        
        return {
            'decision': decision,
            'p_values': p_vals,
            'dimensions': dimensions,
            'changes': changes,
            'slopes': slopes,
            'distances': distances
        }
    
    def _print_diagnostics(self, 
                          token_idx: int,
                          distances: np.ndarray,
                          slopes: np.ndarray,
                          changes: List[int],
                          dimensions: Tuple[Optional[float], Optional[float]]):
        """Print diagnostic information for a token."""
        print(f"Token {token_idx} distances: {distances[:5]}... "
              f"(min: {distances.min():.4f}, max: {distances.max():.4f})")
        print(f"Token {token_idx} slope stats: min={slopes.min():.2f}, "
              f"max={slopes.max():.2f}, mean={slopes.mean():.2f}")
        
        base_dim, fiber_dim = dimensions
        if base_dim is not None and fiber_dim is not None:
            print(f"Token {token_idx} dimensions: Base={base_dim:.2f}, Fiber={fiber_dim:.2f}")
        else:
            print(f"Token {token_idx} dimensions: No significant change detected")
        
        if changes:
            print(f"Token {token_idx} change points: {changes}")
        else:
            print(f"Token {token_idx}: No significant slope changes detected")
        print("-" * 50)
