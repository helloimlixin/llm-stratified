"""Clustering analysis for stratified manifold learning."""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import logging

logger = logging.getLogger(__name__)


class ClusteringAnalyzer:
    """Comprehensive clustering analysis for embeddings."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize clustering analyzer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
    
    def analyze_optimal_clusters(self, embeddings: np.ndarray, 
                               max_clusters: int = 10,
                               min_clusters: int = 2) -> Dict[str, Any]:
        """
        Analyze optimal number of clusters using multiple metrics.
        
        Args:
            embeddings: Embedding matrix
            max_clusters: Maximum number of clusters to test
            min_clusters: Minimum number of clusters to test
            
        Returns:
            Analysis results with optimal cluster numbers
        """
        cluster_range = range(min_clusters, min(max_clusters + 1, len(embeddings)))
        
        results = {
            'n_clusters': list(cluster_range),
            'silhouette_scores': [],
            'davies_bouldin_scores': [],
            'calinski_harabasz_scores': [],
            'inertias': []
        }
        
        for n_clusters in cluster_range:
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            labels = kmeans.fit_predict(embeddings)
            
            # Compute metrics
            silhouette = silhouette_score(embeddings, labels)
            davies_bouldin = davies_bouldin_score(embeddings, labels)
            calinski_harabasz = calinski_harabasz_score(embeddings, labels)
            inertia = kmeans.inertia_
            
            results['silhouette_scores'].append(silhouette)
            results['davies_bouldin_scores'].append(davies_bouldin)
            results['calinski_harabasz_scores'].append(calinski_harabasz)
            results['inertias'].append(inertia)
        
        # Find optimal numbers
        optimal_clusters = {
            'silhouette': cluster_range[np.argmax(results['silhouette_scores'])],
            'davies_bouldin': cluster_range[np.argmin(results['davies_bouldin_scores'])],
            'calinski_harabasz': cluster_range[np.argmax(results['calinski_harabasz_scores'])]
        }
        
        # Elbow method for inertia
        inertias = np.array(results['inertias'])
        if len(inertias) > 2:
            # Compute second derivative to find elbow
            second_deriv = np.diff(inertias, n=2)
            elbow_idx = np.argmax(second_deriv) + 2  # Adjust for diff operations
            optimal_clusters['elbow'] = cluster_range[min(elbow_idx, len(cluster_range) - 1)]
        
        results['optimal_clusters'] = optimal_clusters
        
        return results
    
    def perform_clustering(self, embeddings: np.ndarray, 
                          n_clusters: int = 5,
                          method: str = 'kmeans') -> Dict[str, Any]:
        """
        Perform clustering with specified method.
        
        Args:
            embeddings: Embedding matrix
            n_clusters: Number of clusters
            method: Clustering method ('kmeans', 'dbscan', 'hierarchical')
            
        Returns:
            Clustering results
        """
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            labels = clusterer.fit_predict(embeddings)
            
        elif method == 'dbscan':
            # Auto-determine eps using k-distance
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=n_clusters)
            neighbors_fit = neighbors.fit(embeddings)
            distances, indices = neighbors_fit.kneighbors(embeddings)
            distances = np.sort(distances, axis=0)
            distances = distances[:, n_clusters-1]
            eps = np.percentile(distances, 90)  # Use 90th percentile
            
            clusterer = DBSCAN(eps=eps, min_samples=n_clusters)
            labels = clusterer.fit_predict(embeddings)
            
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(embeddings)
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Compute metrics
        unique_labels = np.unique(labels)
        n_clusters_found = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
        
        if n_clusters_found > 1:
            silhouette = silhouette_score(embeddings, labels)
            davies_bouldin = davies_bouldin_score(embeddings, labels)
            calinski_harabasz = calinski_harabasz_score(embeddings, labels)
        else:
            silhouette = davies_bouldin = calinski_harabasz = 0.0
        
        return {
            'labels': labels,
            'n_clusters_found': n_clusters_found,
            'method': method,
            'metrics': {
                'silhouette_score': silhouette,
                'davies_bouldin_score': davies_bouldin,
                'calinski_harabasz_score': calinski_harabasz
            },
            'cluster_sizes': np.bincount(labels[labels >= 0]) if n_clusters_found > 0 else []
        }
    
    def compare_clustering_methods(self, embeddings: np.ndarray, 
                                  n_clusters: int = 5) -> Dict[str, Any]:
        """
        Compare different clustering methods.
        
        Args:
            embeddings: Embedding matrix
            n_clusters: Number of clusters
            
        Returns:
            Comparison results
        """
        methods = ['kmeans', 'hierarchical']
        if len(embeddings) > 100:  # DBSCAN can be slow on large datasets
            methods.append('dbscan')
        
        results = {}
        
        for method in methods:
            try:
                result = self.perform_clustering(embeddings, n_clusters, method)
                results[method] = result
                logger.info(f"✓ {method}: {result['n_clusters_found']} clusters, "
                          f"silhouette={result['metrics']['silhouette_score']:.3f}")
            except Exception as e:
                logger.warning(f"Failed clustering with {method}: {e}")
        
        # Find best method based on silhouette score
        if results:
            best_method = max(results.keys(), 
                            key=lambda m: results[m]['metrics']['silhouette_score'])
            results['best_method'] = best_method
            results['best_result'] = results[best_method]
        
        return results
    
    def analyze_cluster_stability(self, embeddings: np.ndarray, 
                                n_clusters: int = 5,
                                n_runs: int = 10) -> Dict[str, Any]:
        """
        Analyze clustering stability across multiple runs.
        
        Args:
            embeddings: Embedding matrix
            n_clusters: Number of clusters
            n_runs: Number of runs for stability analysis
            
        Returns:
            Stability analysis results
        """
        all_labels = []
        all_metrics = {'silhouette': [], 'davies_bouldin': [], 'calinski_harabasz': []}
        
        for run in range(n_runs):
            # Use different random states
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state + run)
            labels = kmeans.fit_predict(embeddings)
            all_labels.append(labels)
            
            # Compute metrics
            all_metrics['silhouette'].append(silhouette_score(embeddings, labels))
            all_metrics['davies_bouldin'].append(davies_bouldin_score(embeddings, labels))
            all_metrics['calinski_harabasz'].append(calinski_harabasz_score(embeddings, labels))
        
        # Compute pairwise ARI between runs
        ari_scores = []
        for i in range(n_runs):
            for j in range(i+1, n_runs):
                ari = adjusted_rand_score(all_labels[i], all_labels[j])
                ari_scores.append(ari)
        
        return {
            'mean_ari': np.mean(ari_scores),
            'std_ari': np.std(ari_scores),
            'metric_stability': {
                metric: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
                for metric, values in all_metrics.items()
            },
            'all_labels': all_labels,
            'stability_score': np.mean(ari_scores)  # Overall stability measure
        }


def compute_clustering_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive clustering metrics.
    
    Args:
        embeddings: Embedding matrix
        labels: Cluster labels
        
    Returns:
        Dictionary of clustering metrics
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels >= 0])
    
    if n_clusters <= 1:
        return {
            'silhouette_score': 0.0,
            'davies_bouldin_score': float('inf'),
            'calinski_harabasz_score': 0.0,
            'n_clusters': n_clusters
        }
    
    return {
        'silhouette_score': silhouette_score(embeddings, labels),
        'davies_bouldin_score': davies_bouldin_score(embeddings, labels),
        'calinski_harabasz_score': calinski_harabasz_score(embeddings, labels),
        'n_clusters': n_clusters
    }
