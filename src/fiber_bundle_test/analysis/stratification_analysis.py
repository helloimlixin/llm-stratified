"""Stratification analysis combining fiber bundle testing with manifold learning."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging

from ..core import FiberBundleTest
from .clustering_analysis import ClusteringAnalyzer
from .dimensionality_analysis import DimensionalityAnalyzer

logger = logging.getLogger(__name__)


class StratificationAnalyzer:
    """Comprehensive stratification analysis combining multiple approaches."""
    
    def __init__(self, 
                 fiber_test_params: Optional[Dict[str, Any]] = None,
                 variance_threshold: float = 0.75):
        """
        Initialize stratification analyzer.
        
        Args:
            fiber_test_params: Parameters for fiber bundle test
            variance_threshold: Threshold for intrinsic dimension estimation
        """
        # Default fiber bundle test parameters
        if fiber_test_params is None:
            fiber_test_params = {
                'r_min': 0.01,
                'r_max': 25.0,
                'n_r': 200,
                'alpha': 0.01,
                'window_size': 12
            }
        
        self.fiber_test = FiberBundleTest(**fiber_test_params)
        self.clustering_analyzer = ClusteringAnalyzer()
        self.dim_analyzer = DimensionalityAnalyzer(variance_threshold)
    
    def analyze_stratification(self, 
                             embeddings: np.ndarray,
                             domains: List[str],
                             labels: Optional[np.ndarray] = None,
                             n_clusters: int = 5) -> Dict[str, Any]:
        """
        Perform comprehensive stratification analysis.
        
        Args:
            embeddings: Embedding matrix
            domains: Domain labels for each embedding
            labels: Optional class labels
            n_clusters: Number of clusters for stratification
            
        Returns:
            Comprehensive analysis results
        """
        logger.info("Starting comprehensive stratification analysis...")
        
        results = {
            'input_info': {
                'n_samples': len(embeddings),
                'embedding_dim': embeddings.shape[1],
                'n_domains': len(set(domains)),
                'domains': list(set(domains))
            }
        }
        
        # 1. Fiber Bundle Analysis
        logger.info("Running fiber bundle hypothesis test...")
        print(f"   📊 Step 1/5: Fiber bundle analysis on {len(embeddings)} samples...")
        fiber_results = self.fiber_test.run_test(embeddings, verbose=False)
        results['fiber_bundle'] = fiber_results
        
        # 2. Clustering Analysis
        logger.info("Performing clustering analysis...")
        print(f"   🔍 Step 2/5: Clustering analysis...")
        clustering_results = self.clustering_analyzer.perform_clustering(
            embeddings, n_clusters=n_clusters
        )
        strata = clustering_results['labels']
        results['clustering'] = clustering_results
        
        # 3. Intrinsic Dimensionality Analysis
        logger.info("Computing intrinsic dimensions...")
        print(f"   📐 Step 3/5: Computing intrinsic dimensions...")
        dim_results = self.dim_analyzer.compute_intrinsic_dimensions(embeddings, strata)
        results['dimensionality'] = dim_results
        
        # 4. Domain-Stratum Analysis
        logger.info("Analyzing domain-stratum relationships...")
        print(f"   🌍 Step 4/5: Analyzing domain-stratum relationships...")
        domain_analysis = self._analyze_domain_stratification(domains, strata, labels)
        results['domain_analysis'] = domain_analysis
        
        # 5. Fiber Bundle per Stratum
        logger.info("Running fiber bundle test per stratum...")
        print(f"   🔬 Step 5/5: Running fiber bundle test per stratum...")
        stratum_fiber_results = self._analyze_fiber_bundle_per_stratum(embeddings, strata)
        results['stratum_fiber_analysis'] = stratum_fiber_results
        
        # 6. Summary Statistics
        summary = self._compute_summary_statistics(results)
        results['summary'] = summary
        
        logger.info("Stratification analysis completed")
        return results
    
    def _analyze_domain_stratification(self, 
                                     domains: List[str], 
                                     strata: np.ndarray,
                                     labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze relationship between domains and strata."""
        # Create contingency table
        domain_stratum_table = pd.crosstab(
            domains, strata, 
            rownames=['Domain'], 
            colnames=['Stratum']
        )
        
        # Compute domain purity in each stratum
        stratum_purity = {}
        for stratum_id in np.unique(strata):
            stratum_mask = strata == stratum_id
            stratum_domains = np.array(domains)[stratum_mask]
            
            if len(stratum_domains) > 0:
                unique_domains, counts = np.unique(stratum_domains, return_counts=True)
                max_count = np.max(counts)
                purity = max_count / len(stratum_domains)
                dominant_domain = unique_domains[np.argmax(counts)]
                
                stratum_purity[int(stratum_id)] = {
                    'purity': purity,
                    'dominant_domain': dominant_domain,
                    'domain_distribution': dict(zip(unique_domains, counts.tolist()))
                }
        
        # Compute domain concentration in strata
        domain_concentration = {}
        for domain in set(domains):
            domain_mask = np.array(domains) == domain
            domain_strata = strata[domain_mask]
            
            if len(domain_strata) > 0:
                unique_strata, counts = np.unique(domain_strata, return_counts=True)
                max_count = np.max(counts)
                concentration = max_count / len(domain_strata)
                primary_stratum = unique_strata[np.argmax(counts)]
                
                domain_concentration[domain] = {
                    'concentration': concentration,
                    'primary_stratum': int(primary_stratum),
                    'stratum_distribution': dict(zip(unique_strata.astype(int), counts.tolist()))
                }
        
        analysis = {
            'contingency_table': domain_stratum_table.to_dict(),
            'stratum_purity': stratum_purity,
            'domain_concentration': domain_concentration,
            'avg_stratum_purity': np.mean([s['purity'] for s in stratum_purity.values()]),
            'avg_domain_concentration': np.mean([d['concentration'] for d in domain_concentration.values()])
        }
        
        # Add label analysis if available
        if labels is not None:
            label_analysis = self._analyze_label_stratification(labels, strata)
            analysis['label_analysis'] = label_analysis
        
        return analysis
    
    def _analyze_label_stratification(self, labels: np.ndarray, strata: np.ndarray) -> Dict[str, Any]:
        """Analyze relationship between labels and strata."""
        # Label purity in each stratum
        stratum_label_purity = {}
        for stratum_id in np.unique(strata):
            stratum_mask = strata == stratum_id
            stratum_labels = labels[stratum_mask]
            
            if len(stratum_labels) > 0:
                unique_labels, counts = np.unique(stratum_labels, return_counts=True)
                max_count = np.max(counts)
                purity = max_count / len(stratum_labels)
                dominant_label = unique_labels[np.argmax(counts)]
                
                stratum_label_purity[int(stratum_id)] = {
                    'purity': purity,
                    'dominant_label': int(dominant_label),
                    'label_distribution': dict(zip(unique_labels.astype(int), counts.tolist()))
                }
        
        return {
            'stratum_label_purity': stratum_label_purity,
            'avg_label_purity': np.mean([s['purity'] for s in stratum_label_purity.values()])
        }
    
    def _analyze_fiber_bundle_per_stratum(self, 
                                        embeddings: np.ndarray, 
                                        strata: np.ndarray) -> Dict[str, Any]:
        """Run fiber bundle analysis for each stratum separately."""
        stratum_results = {}
        
        for stratum_id in np.unique(strata):
            stratum_mask = strata == stratum_id
            stratum_embeddings = embeddings[stratum_mask]
            
            if len(stratum_embeddings) < 10:  # Need minimum samples
                stratum_results[int(stratum_id)] = {
                    'error': 'Insufficient samples',
                    'n_samples': len(stratum_embeddings)
                }
                continue
            
            try:
                # Run fiber bundle test on this stratum
                stratum_fiber_results = self.fiber_test.run_test(
                    stratum_embeddings, verbose=False
                )
                
                stratum_results[int(stratum_id)] = {
                    'n_samples': len(stratum_embeddings),
                    'rejection_rate': stratum_fiber_results['rejection_rate'],
                    'total_rejections': stratum_fiber_results['total_rejections'],
                    'avg_base_dim': self._compute_avg_dimension(
                        stratum_fiber_results['dimensions'], 0
                    ),
                    'avg_fiber_dim': self._compute_avg_dimension(
                        stratum_fiber_results['dimensions'], 1
                    )
                }
                
            except Exception as e:
                logger.warning(f"Fiber bundle test failed for stratum {stratum_id}: {e}")
                stratum_results[int(stratum_id)] = {
                    'error': str(e),
                    'n_samples': len(stratum_embeddings)
                }
        
        return stratum_results
    
    def _compute_avg_dimension(self, dimensions: List[Tuple], dim_index: int) -> Optional[float]:
        """Compute average dimension from dimension tuples."""
        dims = [d[dim_index] for d in dimensions if d[dim_index] is not None]
        return np.mean(dims) if dims else None
    
    def _compute_summary_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute summary statistics from all analyses."""
        summary = {}
        
        # Fiber bundle summary
        fiber_results = results['fiber_bundle']
        summary['overall_fiber_rejection_rate'] = fiber_results['rejection_rate']
        summary['total_fiber_rejections'] = fiber_results['total_rejections']
        
        # Clustering summary
        clustering_results = results['clustering']
        summary['clustering_quality'] = clustering_results['metrics']['silhouette_score']
        summary['n_clusters_found'] = clustering_results['n_clusters_found']
        
        # Dimensionality summary
        dim_results = results['dimensionality']
        summary['overall_intrinsic_dimension'] = dim_results['overall_intrinsic_dimension']
        
        if 'stratum_dimensions' in dim_results:
            stratum_dims = list(dim_results['stratum_dimensions'].values())
            summary['avg_stratum_dimension'] = np.mean(stratum_dims)
            summary['dimension_heterogeneity'] = np.std(stratum_dims)
        
        # Domain analysis summary
        domain_analysis = results['domain_analysis']
        summary['avg_stratum_purity'] = domain_analysis.get('avg_stratum_purity', 0)
        summary['avg_domain_concentration'] = domain_analysis.get('avg_domain_concentration', 0)
        
        return summary


def analyze_stratification(embeddings: np.ndarray,
                         domains: List[str], 
                         labels: Optional[np.ndarray] = None,
                         n_clusters: int = 5,
                         fiber_test_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function for complete stratification analysis (notebook compatibility).
    
    Args:
        embeddings: Embedding matrix
        domains: Domain labels
        labels: Optional class labels
        n_clusters: Number of clusters
        fiber_test_params: Fiber bundle test parameters
        
    Returns:
        Complete analysis results
    """
    analyzer = StratificationAnalyzer(fiber_test_params)
    return analyzer.analyze_stratification(embeddings, domains, labels, n_clusters)
