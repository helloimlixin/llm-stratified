"""
Interactive visualizations for stratified manifold learning.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, UMAP
from sklearn.cluster import KMeans
import json
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class InteractiveVisualizer:
    """Interactive visualization tools for stratified manifolds."""
    
    def __init__(self, data, labels=None, domains=None, strata=None, texts=None):
        self.data = np.array(data)
        self.labels = labels
        self.domains = domains
        self.strata = strata
        self.texts = texts
        self.n_samples, self.n_features = self.data.shape
        
        # Prepare data for visualization
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for interactive visualization."""
        # Apply PCA for 2D visualization
        self.pca_2d = PCA(n_components=2)
        self.data_2d = self.pca_2d.fit_transform(self.data)
        
        # Apply t-SNE for 2D visualization
        try:
            self.tsne_2d = TSNE(n_components=2, random_state=42, perplexity=min(30, self.n_samples//4))
            self.data_tsne = self.tsne_2d.fit_transform(self.data)
        except:
            self.data_tsne = self.data_2d  # Fallback to PCA
        
        # Apply UMAP for 2D visualization
        try:
            import umap
            self.umap_2d = umap.UMAP(n_components=2, random_state=42)
            self.data_umap = self.umap_2d.fit_transform(self.data)
        except:
            self.data_umap = self.data_2d  # Fallback to PCA
        
        # Create hover text
        self.hover_text = []
        for i in range(self.n_samples):
            text_parts = [f"Sample {i}"]
            if self.texts is not None and i < len(self.texts):
                text_parts.append(f"Text: {self.texts[i][:100]}...")
            if self.domains is not None and i < len(self.domains):
                text_parts.append(f"Domain: {self.domains[i]}")
            if self.strata is not None and i < len(self.strata):
                text_parts.append(f"Stratum: {self.strata[i]}")
            if self.labels is not None and i < len(self.labels):
                text_parts.append(f"Label: {self.labels[i]}")
            
            self.hover_text.append("<br>".join(text_parts))
    
    def create_interactive_scatter(self, method='pca', color_by='strata', title=None):
        """Create interactive scatter plot."""
        if method == 'pca':
            x_data, y_data = self.data_2d[:, 0], self.data_2d[:, 1]
            method_name = "PCA"
        elif method == 'tsne':
            x_data, y_data = self.data_tsne[:, 0], self.data_tsne[:, 1]
            method_name = "t-SNE"
        elif method == 'umap':
            x_data, y_data = self.data_umap[:, 0], self.data_umap[:, 1]
            method_name = "UMAP"
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Determine color data
        if color_by == 'strata' and self.strata is not None:
            color_data = self.strata
            color_title = "Stratum"
        elif color_by == 'domains' and self.domains is not None:
            color_data = self.domains
            color_title = "Domain"
        elif color_by == 'labels' and self.labels is not None:
            color_data = self.labels
            color_title = "Label"
        else:
            color_data = np.arange(self.n_samples)
            color_title = "Sample Index"
        
        # Create scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            marker=dict(
                size=8,
                color=color_data,
                colorscale='viridis',
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=self.hover_text,
            hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
            name='Data Points'
        ))
        
        fig.update_layout(
            title=title or f"Interactive {method_name} Visualization",
            xaxis_title=f"{method_name} Component 1",
            yaxis_title=f"{method_name} Component 2",
            showlegend=False,
            width=800,
            height=600,
            hovermode='closest'
        )
        
        return fig
    
    def create_3d_scatter(self, method='pca', color_by='strata', title=None):
        """Create interactive 3D scatter plot."""
        # Apply PCA for 3D visualization
        pca_3d = PCA(n_components=3)
        data_3d = pca_3d.fit_transform(self.data)
        
        # Determine color data
        if color_by == 'strata' and self.strata is not None:
            color_data = self.strata
            color_title = "Stratum"
        elif color_by == 'domains' and self.domains is not None:
            color_data = self.domains
            color_title = "Domain"
        elif color_by == 'labels' and self.labels is not None:
            color_data = self.labels
            color_title = "Label"
        else:
            color_data = np.arange(self.n_samples)
            color_title = "Sample Index"
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(
            x=data_3d[:, 0],
            y=data_3d[:, 1],
            z=data_3d[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=color_data,
                colorscale='viridis',
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=self.hover_text,
            hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
            name='Data Points'
        ))
        
        fig.update_layout(
            title=title or "Interactive 3D Visualization",
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3"
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_comparison_dashboard(self, title="Stratified Manifold Dashboard"):
        """Create comprehensive comparison dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('PCA Visualization', 't-SNE Visualization', 
                          'UMAP Visualization', '3D PCA Visualization'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter3d"}]]
        )
        
        # PCA plot
        fig.add_trace(
            go.Scatter(
                x=self.data_2d[:, 0],
                y=self.data_2d[:, 1],
                mode='markers',
                marker=dict(
                    size=6,
                    color=self.strata if self.strata is not None else np.arange(self.n_samples),
                    colorscale='viridis',
                    opacity=0.7
                ),
                text=self.hover_text,
                hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                name='PCA'
            ),
            row=1, col=1
        )
        
        # t-SNE plot
        fig.add_trace(
            go.Scatter(
                x=self.data_tsne[:, 0],
                y=self.data_tsne[:, 1],
                mode='markers',
                marker=dict(
                    size=6,
                    color=self.strata if self.strata is not None else np.arange(self.n_samples),
                    colorscale='viridis',
                    opacity=0.7
                ),
                text=self.hover_text,
                hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                name='t-SNE'
            ),
            row=1, col=2
        )
        
        # UMAP plot
        fig.add_trace(
            go.Scatter(
                x=self.data_umap[:, 0],
                y=self.data_umap[:, 1],
                mode='markers',
                marker=dict(
                    size=6,
                    color=self.strata if self.strata is not None else np.arange(self.n_samples),
                    colorscale='viridis',
                    opacity=0.7
                ),
                text=self.hover_text,
                hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                name='UMAP'
            ),
            row=2, col=1
        )
        
        # 3D PCA plot
        pca_3d = PCA(n_components=3)
        data_3d = pca_3d.fit_transform(self.data)
        
        fig.add_trace(
            go.Scatter3d(
                x=data_3d[:, 0],
                y=data_3d[:, 1],
                z=data_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=self.strata if self.strata is not None else np.arange(self.n_samples),
                    colorscale='viridis',
                    opacity=0.7
                ),
                text=self.hover_text,
                hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
                name='3D PCA'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_stratified_analysis(self, title="Stratified Analysis"):
        """Create stratified analysis visualization."""
        if self.strata is None:
            raise ValueError("Strata information required for stratified analysis")
        
        unique_strata = np.unique(self.strata)
        n_strata = len(unique_strata)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Stratum Distribution', 'Domain vs Stratum', 
                          'Stratum Separation', 'Stratum Properties'),
            specs=[[{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Stratum distribution
        stratum_counts = [np.sum(self.strata == s) for s in unique_strata]
        fig.add_trace(
            go.Bar(x=[f"Stratum {s}" for s in unique_strata], y=stratum_counts, name="Count"),
            row=1, col=1
        )
        
        # Domain vs Stratum heatmap
        if self.domains is not None:
            domain_stratum_matrix = np.zeros((len(set(self.domains)), n_strata))
            domain_list = list(set(self.domains))
            
            for i, domain in enumerate(domain_list):
                for j, stratum in enumerate(unique_strata):
                    domain_stratum_matrix[i, j] = np.sum((np.array(self.domains) == domain) & (self.strata == stratum))
            
            fig.add_trace(
                go.Heatmap(
                    z=domain_stratum_matrix,
                    x=[f"Stratum {s}" for s in unique_strata],
                    y=domain_list,
                    colorscale='Blues'
                ),
                row=1, col=2
            )
        
        # Stratum separation (PCA)
        colors = px.colors.qualitative.Set1[:n_strata]
        for i, stratum in enumerate(unique_strata):
            mask = self.strata == stratum
            fig.add_trace(
                go.Scatter(
                    x=self.data_2d[mask, 0],
                    y=self.data_2d[mask, 1],
                    mode='markers',
                    marker=dict(size=8, color=colors[i]),
                    name=f"Stratum {stratum}"
                ),
                row=2, col=1
            )
        
        # Stratum properties
        stratum_properties = []
        for stratum in unique_strata:
            mask = self.strata == stratum
            stratum_data = self.data[mask]
            
            # Compute properties
            mean_norm = np.mean(np.linalg.norm(stratum_data, axis=1))
            std_norm = np.std(np.linalg.norm(stratum_data, axis=1))
            
            stratum_properties.append([mean_norm, std_norm])
        
        stratum_properties = np.array(stratum_properties)
        
        fig.add_trace(
            go.Bar(x=[f"Stratum {s}" for s in unique_strata], y=stratum_properties[:, 0], name="Mean Norm"),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def save_interactive_html(self, fig, filename):
        """Save interactive plot as HTML."""
        pyo.plot(fig, filename=filename, auto_open=False)
        print(f"Interactive visualization saved as {filename}")

def create_interactive_dashboard(data, labels=None, domains=None, strata=None, texts=None, 
                                output_dir="interactive_visualizations"):
    """Create comprehensive interactive dashboard."""
    print("🎨 Creating Interactive Visualizations")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = InteractiveVisualizer(data, labels, domains, strata, texts)
    
    # Create various visualizations
    visualizations = {}
    
    # 1. PCA scatter plot
    print("Creating PCA scatter plot...")
    fig_pca = visualizer.create_interactive_scatter(method='pca', color_by='strata')
    visualizer.save_interactive_html(fig_pca, f"{output_dir}/pca_scatter.html")
    visualizations['pca'] = fig_pca
    
    # 2. t-SNE scatter plot
    print("Creating t-SNE scatter plot...")
    fig_tsne = visualizer.create_interactive_scatter(method='tsne', color_by='strata')
    visualizer.save_interactive_html(fig_tsne, f"{output_dir}/tsne_scatter.html")
    visualizations['tsne'] = fig_tsne
    
    # 3. UMAP scatter plot
    print("Creating UMAP scatter plot...")
    fig_umap = visualizer.create_interactive_scatter(method='umap', color_by='strata')
    visualizer.save_interactive_html(fig_umap, f"{output_dir}/umap_scatter.html")
    visualizations['umap'] = fig_umap
    
    # 4. 3D scatter plot
    print("Creating 3D scatter plot...")
    fig_3d = visualizer.create_3d_scatter(color_by='strata')
    visualizer.save_interactive_html(fig_3d, f"{output_dir}/3d_scatter.html")
    visualizations['3d'] = fig_3d
    
    # 5. Comparison dashboard
    print("Creating comparison dashboard...")
    fig_dashboard = visualizer.create_comparison_dashboard()
    visualizer.save_interactive_html(fig_dashboard, f"{output_dir}/comparison_dashboard.html")
    visualizations['dashboard'] = fig_dashboard
    
    # 6. Stratified analysis
    if strata is not None:
        print("Creating stratified analysis...")
        fig_stratified = visualizer.create_stratified_analysis()
        visualizer.save_interactive_html(fig_stratified, f"{output_dir}/stratified_analysis.html")
        visualizations['stratified'] = fig_stratified
    
    print(f"\n✅ Interactive visualizations created successfully!")
    print(f"   Files saved in '{output_dir}/' directory")
    print(f"   Open any .html file in your browser to view interactive plots")
    
    return visualizations

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_swiss_roll
    
    # Generate sample data
    data, _ = make_swiss_roll(n_samples=300, noise=0.1)
    
    # Create some strata and domains
    strata = np.zeros(300, dtype=int)
    strata[100:200] = 1
    strata[200:300] = 2
    
    domains = ['domain_A'] * 100 + ['domain_B'] * 100 + ['domain_C'] * 100
    labels = np.random.randint(0, 2, 300)
    
    # Create interactive dashboard
    visualizations = create_interactive_dashboard(
        data, labels=labels, domains=domains, strata=strata
    )
