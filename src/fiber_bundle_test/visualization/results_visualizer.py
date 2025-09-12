"""Visualization utilities for fiber bundle hypothesis test results."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ResultsVisualizer:
    """Visualize results from fiber bundle hypothesis tests."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
    
    def plot_token_analysis(self,
                           log_r: np.ndarray,
                           log_nx_r: np.ndarray,
                           slopes: np.ndarray,
                           changes: List[int],
                           token_idx: int,
                           token_label: str = None) -> plt.Figure:
        """
        Plot log-log curve and slopes for a single token.
        
        Args:
            log_r: Logarithmic radius values
            log_nx_r: Logarithmic neighbor counts
            slopes: Estimated slopes
            changes: Detected change points
            token_idx: Token index
            token_label: Optional label for the token
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Plot log-log curve
        ax1.plot(log_r, log_nx_r, 'b-', linewidth=2, label="log N_x(r) vs log r")
        ax1.set_xlabel("log r", fontsize=12)
        ax1.set_ylabel("log N_x(r)", fontsize=12)
        ax1.set_title(f"Log-Log Plot (Token {token_idx})", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot slopes with change points
        ax2.plot(log_r, slopes, 'g-', linewidth=2, label="Slopes")
        
        # Mark change points
        for change in changes:
            ax2.axvline(log_r[change], color='red', linestyle='--', 
                       linewidth=2, alpha=0.7, label='Change Point' if change == changes[0] else "")
        
        ax2.set_xlabel("log r", fontsize=12)
        ax2.set_ylabel("Slope", fontsize=12)
        ax2.set_title(f"Slopes with Changes (Token {token_idx})", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add token label if provided
        if token_label:
            fig.suptitle(f"Analysis for Token: {token_label}", fontsize=16, y=0.98)
        
        plt.tight_layout()
        return fig
    
    def plot_results_summary(self, results: Dict[str, Any], 
                           token_labels: Optional[List[str]] = None) -> plt.Figure:
        """
        Plot summary of hypothesis test results.
        
        Args:
            results: Results dictionary from FiberBundleTest.run_test()
            token_labels: Optional labels for tokens
            
        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        decisions = [decision for _, decision in results['results']]
        dimensions = results['dimensions']
        
        # 1. Rejection counts
        rejection_counts = {'Reject': decisions.count('Reject'),
                          'Fail to Reject': decisions.count('Fail to Reject')}
        
        colors = ['#ff7f7f', '#7fbf7f']
        ax1.pie(rejection_counts.values(), labels=rejection_counts.keys(), 
                autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Hypothesis Test Results', fontsize=14)
        
        # 2. Base vs Fiber dimensions scatter plot
        base_dims = [dim[0] for dim in dimensions if dim[0] is not None]
        fiber_dims = [dim[1] for dim in dimensions if dim[1] is not None]
        
        if base_dims and fiber_dims:
            ax2.scatter(base_dims, fiber_dims, alpha=0.6, s=50)
            ax2.set_xlabel('Base Dimension', fontsize=12)
            ax2.set_ylabel('Fiber Dimension', fontsize=12)
            ax2.set_title('Base vs Fiber Dimensions', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # Add diagonal line
            min_dim = min(min(base_dims), min(fiber_dims))
            max_dim = max(max(base_dims), max(fiber_dims))
            ax2.plot([min_dim, max_dim], [min_dim, max_dim], 'r--', alpha=0.5, label='y=x')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No dimension data available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Base vs Fiber Dimensions', fontsize=14)
        
        # 3. P-value distribution
        p_values = results['raw_p_values']
        if p_values:
            ax3.hist(p_values, bins=20, alpha=0.7, edgecolor='black')
            ax3.axvline(results['parameters']['alpha'], color='red', linestyle='--', 
                       label=f'α = {results["parameters"]["alpha"]}')
            ax3.set_xlabel('P-values', fontsize=12)
            ax3.set_ylabel('Frequency', fontsize=12)
            ax3.set_title('P-value Distribution', fontsize=14)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No p-values available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('P-value Distribution', fontsize=14)
        
        # 4. Results by token (if labels provided)
        if token_labels:
            # Count by token type
            token_types = set(token_labels)
            type_results = {t: {'Reject': 0, 'Fail to Reject': 0} for t in token_types}
            
            for (_, decision), label in zip(results['results'], token_labels):
                type_results[label][decision] += 1
            
            types = list(token_types)
            reject_counts = [type_results[t]['Reject'] for t in types]
            fail_counts = [type_results[t]['Fail to Reject'] for t in types]
            
            x = np.arange(len(types))
            width = 0.35
            
            ax4.bar(x - width/2, reject_counts, width, label='Reject', color='#ff7f7f')
            ax4.bar(x + width/2, fail_counts, width, label='Fail to Reject', color='#7fbf7f')
            
            ax4.set_xlabel('Token Type', fontsize=12)
            ax4.set_ylabel('Count', fontsize=12)
            ax4.set_title('Results by Token Type', fontsize=14)
            ax4.set_xticks(x)
            ax4.set_xticklabels(types)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Token labels not provided', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Results by Token Type', fontsize=14)
        
        plt.tight_layout()
        return fig
    
    def plot_dimension_analysis(self, results: Dict[str, Any], 
                              token_labels: Optional[List[str]] = None) -> plt.Figure:
        """
        Plot detailed dimension analysis.
        
        Args:
            results: Results dictionary from FiberBundleTest.run_test()
            token_labels: Optional labels for tokens
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        dimensions = results['dimensions']
        base_dims = [dim[0] for dim in dimensions if dim[0] is not None]
        fiber_dims = [dim[1] for dim in dimensions if dim[1] is not None]
        
        if not base_dims or not fiber_dims:
            fig.text(0.5, 0.5, 'No dimension data available for plotting', 
                    ha='center', va='center', fontsize=16)
            return fig
        
        # Base dimension distribution
        ax1.hist(base_dims, bins=15, alpha=0.7, edgecolor='black', color='skyblue')
        ax1.set_xlabel('Base Dimension', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Base Dimension Distribution', fontsize=14)
        ax1.axvline(np.mean(base_dims), color='red', linestyle='--', 
                   label=f'Mean = {np.mean(base_dims):.2f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Fiber dimension distribution
        ax2.hist(fiber_dims, bins=15, alpha=0.7, edgecolor='black', color='lightcoral')
        ax2.set_xlabel('Fiber Dimension', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Fiber Dimension Distribution', fontsize=14)
        ax2.axvline(np.mean(fiber_dims), color='red', linestyle='--', 
                   label=f'Mean = {np.mean(fiber_dims):.2f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_plots(self, figures: List[plt.Figure], 
                   filenames: List[str], 
                   output_dir: str = "./plots",
                   dpi: int = 300):
        """
        Save multiple figures to files.
        
        Args:
            figures: List of matplotlib Figure objects
            filenames: List of filenames (without extension)
            output_dir: Output directory
            dpi: Resolution for saved figures
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for fig, filename in zip(figures, filenames):
            filepath = os.path.join(output_dir, f"{filename}.png")
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"Saved plot: {filepath}")
    
    @staticmethod
    def show_all():
        """Display all created plots."""
        plt.show()
