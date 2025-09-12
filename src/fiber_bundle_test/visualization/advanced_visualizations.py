"""Advanced visualization utilities for stratified manifold learning."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging

# Standard plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Interactive plotting
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Dimensionality reduction
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class AdvancedVisualizer:
    """Advanced visualization tools for stratified manifold analysis."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize advanced visualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        self.figsize = figsize
        
        # Set plotting style
        try:
            plt.style.use(style)
            sns.set_palette("husl")
        except:
            logger.warning(f"Style '{style}' not available, using default")
    
    def plot_3d_embeddings(self, embeddings: np.ndarray, 
                          domains: List[str],
                          strata: Optional[np.ndarray] = None,
                          labels: Optional[np.ndarray] = None,
                          method: str = 'pca',
                          title: str = "3D Embedding Visualization") -> Optional[Any]:
        """
        Create 3D visualization of embeddings.
        
        Args:
            embeddings: Embedding matrix
            domains: Domain labels
            strata: Cluster/strata labels
            labels: Class labels
            method: Dimensionality reduction method ('pca', 'umap')
            title: Plot title
            
        Returns:
            Plotly figure if available, None otherwise
        """
        if not HAS_PLOTLY:
            logger.warning("Plotly not available, skipping 3D visualization")
            return None
        
        # Reduce to 3D
        if method == 'pca':
            reducer = PCA(n_components=3)
            emb_3d = reducer.fit_transform(embeddings)
            variance_explained = reducer.explained_variance_ratio_.sum()
            method_info = f"PCA (explained variance: {variance_explained:.1%})"
            
        elif method == 'umap' and HAS_UMAP:
            # Suppress UMAP parallelism warning
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="n_jobs value.*overridden.*by setting random_state")
                reducer = umap.UMAP(n_components=3, random_state=42, n_jobs=1)
                emb_3d = reducer.fit_transform(embeddings)
            method_info = "UMAP"
            
        else:
            logger.warning("UMAP not available, falling back to PCA")
            reducer = PCA(n_components=3)
            emb_3d = reducer.fit_transform(embeddings)
            variance_explained = reducer.explained_variance_ratio_.sum()
            method_info = f"PCA (explained variance: {variance_explained:.1%})"
        
        # Create DataFrame for plotting
        df_plot = pd.DataFrame({
            "x": emb_3d[:, 0],
            "y": emb_3d[:, 1], 
            "z": emb_3d[:, 2],
            "domain": domains
        })
        
        # Add additional information if available
        if labels is not None:
            df_plot["label"] = labels
            df_plot["domain_label"] = [f"{d}_{l}" for d, l in zip(domains, labels)]
            color_col = "domain_label"
        else:
            color_col = "domain"
        
        if strata is not None:
            df_plot["stratum"] = strata
        
        # Create 3D scatter plot
        hover_data = ["domain"]
        if labels is not None:
            hover_data.append("label")
        if strata is not None:
            hover_data.append("stratum")
        
        fig = px.scatter_3d(
            df_plot,
            x="x", y="y", z="z",
            color=color_col,
            hover_data=hover_data,
            title=f"{title} ({method_info})",
            width=1000,
            height=700
        )
        
        return fig
    
    def plot_gating_heatmaps(self, gating_probs: np.ndarray,
                           strata: np.ndarray,
                           domains: List[str],
                           title: str = "Expert Gating Analysis") -> Optional[Any]:
        """
        Create heatmaps showing expert gating probabilities.
        
        Args:
            gating_probs: Gating probabilities matrix
            strata: Stratum labels
            domains: Domain labels
            title: Plot title
            
        Returns:
            Plotly figure if available
        """
        if not HAS_PLOTLY:
            logger.warning("Plotly not available, skipping gating heatmaps")
            return None
        
        K = gating_probs.shape[1]
        
        # Create DataFrame
        df_gating = pd.DataFrame(gating_probs, columns=[f"Expert_{i}" for i in range(K)])
        df_gating["Stratum"] = strata
        df_gating["Domain"] = domains
        
        # Compute averages
        expert_cols = [f"Expert_{i}" for i in range(K)]
        avg_gating_per_stratum = df_gating.groupby("Stratum")[expert_cols].mean()
        avg_gating_per_domain = df_gating.groupby("Domain")[expert_cols].mean()
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Average Gating per Stratum", "Average Gating per Domain"],
            column_widths=[0.5, 0.5]
        )
        
        # Add heatmaps
        fig.add_trace(
            go.Heatmap(
                z=avg_gating_per_stratum.values,
                x=expert_cols,
                y=avg_gating_per_stratum.index,
                colorscale="plasma",
                name="Stratum"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=avg_gating_per_domain.values,
                x=expert_cols,
                y=avg_gating_per_domain.index,
                colorscale="plasma",
                name="Domain"
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=title,
            width=1200,
            height=500,
            showlegend=False
        )
        
        return fig
    
    def plot_umap_visualization(self, embeddings: np.ndarray,
                              domains: List[str],
                              strata: np.ndarray,
                              title: str = "UMAP Visualization") -> Optional[Any]:
        """
        Create UMAP visualization.
        
        Args:
            embeddings: Embedding matrix
            domains: Domain labels
            strata: Stratum labels
            title: Plot title
            
        Returns:
            Plotly figure if available
        """
        if not HAS_PLOTLY or not HAS_UMAP:
            logger.warning("Plotly or UMAP not available, skipping UMAP visualization")
            return None
        
        # UMAP reduction to 3D
        umap_3d = umap.UMAP(n_components=3, random_state=42).fit_transform(embeddings)
        
        df_umap = pd.DataFrame(umap_3d, columns=["component_0", "component_1", "component_2"])
        df_umap["Domain"] = domains
        df_umap["Stratum"] = strata
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            df_umap,
            x="component_0", y="component_1", z="component_2",
            color="Domain",
            symbol="Stratum",
            title=title,
            hover_data=["Domain", "Stratum"],
            width=1000,
            height=700
        )
        
        return fig
    
    def plot_stratification_summary(self, analysis_results: Dict[str, Any]) -> plt.Figure:
        """
        Create comprehensive stratification summary plot.
        
        Args:
            analysis_results: Results from StratificationAnalyzer
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Fiber bundle rejection rates
        fiber_results = analysis_results['fiber_bundle']
        decisions = [decision for _, decision in fiber_results['results']]
        rejection_counts = {'Reject': decisions.count('Reject'),
                          'Fail to Reject': decisions.count('Fail to Reject')}
        
        axes[0, 0].pie(rejection_counts.values(), labels=rejection_counts.keys(), 
                      autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Overall Fiber Bundle Results')
        
        # 2. Domain distribution
        if 'domain_analysis' in analysis_results:
            domain_analysis = analysis_results['domain_analysis']
            if 'contingency_table' in domain_analysis:
                contingency = pd.DataFrame(domain_analysis['contingency_table'])
                sns.heatmap(contingency, annot=True, fmt='d', ax=axes[0, 1], cmap='Blues')
                axes[0, 1].set_title('Domain-Stratum Distribution')
        
        # 3. Intrinsic dimensions per stratum
        if 'dimensionality' in analysis_results:
            dim_results = analysis_results['dimensionality']
            if 'stratum_dimensions' in dim_results:
                stratum_dims = dim_results['stratum_dimensions']
                strata = list(stratum_dims.keys())
                dims = list(stratum_dims.values())
                
                axes[0, 2].bar(strata, dims)
                axes[0, 2].set_title('Intrinsic Dimensions per Stratum')
                axes[0, 2].set_xlabel('Stratum')
                axes[0, 2].set_ylabel('Intrinsic Dimension')
        
        # 4. Clustering quality metrics
        if 'clustering' in analysis_results:
            clustering = analysis_results['clustering']
            metrics = clustering['metrics']
            
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            axes[1, 0].bar(range(len(metric_names)), metric_values)
            axes[1, 0].set_title('Clustering Quality Metrics')
            axes[1, 0].set_xticks(range(len(metric_names)))
            axes[1, 0].set_xticklabels(metric_names, rotation=45)
        
        # 5. Fiber bundle per stratum
        if 'stratum_fiber_analysis' in analysis_results:
            stratum_fiber = analysis_results['stratum_fiber_analysis']
            
            stratum_ids = []
            rejection_rates = []
            
            for stratum_id, results in stratum_fiber.items():
                if 'rejection_rate' in results:
                    stratum_ids.append(stratum_id)
                    rejection_rates.append(results['rejection_rate'])
            
            if stratum_ids:
                axes[1, 1].bar(stratum_ids, rejection_rates)
                axes[1, 1].set_title('Fiber Bundle Rejection Rate per Stratum')
                axes[1, 1].set_xlabel('Stratum')
                axes[1, 1].set_ylabel('Rejection Rate')
        
        # 6. Summary statistics
        if 'summary' in analysis_results:
            summary = analysis_results['summary']
            
            summary_text = f"""
            Overall Rejection Rate: {summary.get('overall_fiber_rejection_rate', 0):.1%}
            Clustering Quality: {summary.get('clustering_quality', 0):.3f}
            Avg Stratum Dimension: {summary.get('avg_stratum_dimension', 0):.1f}
            Dimension Heterogeneity: {summary.get('dimension_heterogeneity', 0):.2f}
            Stratum Purity: {summary.get('avg_stratum_purity', 0):.1%}
            """
            
            axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                           fontsize=12, verticalalignment='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[1, 2].set_title('Summary Statistics')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def save_interactive_plots(self, figures: List[Any], 
                             filenames: List[str],
                             output_dir: str = "./plots"):
        """
        Save interactive Plotly figures as HTML.
        
        Args:
            figures: List of Plotly figures
            filenames: List of filenames (without extension)
            output_dir: Output directory
        """
        if not HAS_PLOTLY:
            logger.warning("Plotly not available, cannot save interactive plots")
            return
        
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for fig, filename in zip(figures, filenames):
            if fig is not None:
                filepath = output_path / f"{filename}.html"
                fig.write_html(str(filepath))
                logger.info(f"Saved interactive plot: {filepath}")


def create_notebook_visualizations(embeddings: np.ndarray,
                                 domains: List[str], 
                                 strata: np.ndarray,
                                 gating_probs: Optional[np.ndarray] = None,
                                 labels: Optional[np.ndarray] = None,
                                 output_dir: str = "./plots") -> Dict[str, Any]:
    """
    Create all visualizations from the notebook (compatibility function).
    
    Args:
        embeddings: Embedding matrix
        domains: Domain labels
        strata: Stratum labels
        gating_probs: Expert gating probabilities
        labels: Class labels
        output_dir: Output directory for plots
        
    Returns:
        Dictionary with created figures
    """
    visualizer = AdvancedVisualizer()
    figures = {}
    
    # 3D PCA visualization
    pca_fig = visualizer.plot_3d_embeddings(
        embeddings, domains, strata, labels, 
        method='pca', 
        title="3D PCA Visualization of Embeddings"
    )
    figures['pca_3d'] = pca_fig
    
    # 3D UMAP visualization
    umap_fig = visualizer.plot_3d_embeddings(
        embeddings, domains, strata, labels,
        method='umap',
        title="3D UMAP Visualization of Embeddings"
    )
    figures['umap_3d'] = umap_fig
    
    # Gating heatmaps
    if gating_probs is not None:
        gating_fig = visualizer.plot_gating_heatmaps(
            gating_probs, strata, domains,
            title="Expert Gating Analysis"
        )
        figures['gating_heatmaps'] = gating_fig
    
    # Save interactive plots
    plot_list = [fig for fig in figures.values() if fig is not None]
    filenames = [name for name, fig in figures.items() if fig is not None]
    
    visualizer.save_interactive_plots(plot_list, filenames, output_dir)
    
    return figures


def plot_expert_utilization(gating_probs: np.ndarray, 
                           expert_names: Optional[List[str]] = None) -> plt.Figure:
    """
    Plot expert utilization statistics.
    
    Args:
        gating_probs: Gating probabilities matrix
        expert_names: Optional expert names
        
    Returns:
        Matplotlib figure
    """
    K = gating_probs.shape[1]
    if expert_names is None:
        expert_names = [f"Expert_{i}" for i in range(K)]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Average utilization
    avg_utilization = gating_probs.mean(axis=0)
    axes[0].bar(range(K), avg_utilization)
    axes[0].set_title('Average Expert Utilization')
    axes[0].set_xlabel('Expert')
    axes[0].set_ylabel('Average Probability')
    axes[0].set_xticks(range(K))
    axes[0].set_xticklabels(expert_names, rotation=45)
    
    # Hard assignment distribution
    hard_assignments = gating_probs.argmax(axis=1)
    assignment_counts = np.bincount(hard_assignments, minlength=K)
    assignment_freq = assignment_counts / len(hard_assignments)
    
    axes[1].bar(range(K), assignment_freq)
    axes[1].set_title('Hard Assignment Frequency')
    axes[1].set_xlabel('Expert')
    axes[1].set_ylabel('Assignment Frequency')
    axes[1].set_xticks(range(K))
    axes[1].set_xticklabels(expert_names, rotation=45)
    
    plt.tight_layout()
    return fig


def plot_dimension_analysis(intrinsic_dims: Dict[int, int]) -> plt.Figure:
    """
    Plot intrinsic dimension analysis.
    
    Args:
        intrinsic_dims: Dictionary mapping stratum to intrinsic dimension
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    strata = list(intrinsic_dims.keys())
    dims = list(intrinsic_dims.values())
    
    # Bar plot of dimensions
    axes[0].bar(strata, dims, color='skyblue', edgecolor='navy')
    axes[0].set_title('Intrinsic Dimensions per Stratum')
    axes[0].set_xlabel('Stratum')
    axes[0].set_ylabel('Intrinsic Dimension')
    
    # Add value labels
    for i, (stratum, dim) in enumerate(zip(strata, dims)):
        axes[0].text(i, dim + 0.1, str(dim), ha='center', va='bottom', fontweight='bold')
    
    # Histogram of dimensions
    axes[1].hist(dims, bins=max(1, len(set(dims))), alpha=0.7, color='lightcoral', edgecolor='darkred')
    axes[1].set_title('Distribution of Intrinsic Dimensions')
    axes[1].set_xlabel('Intrinsic Dimension')
    axes[1].set_ylabel('Number of Strata')
    
    plt.tight_layout()
    return fig
