"""
Utility functions for stratified manifold learning experiments.

This module contains common utility functions used across different models
and experiments, including data loading, preprocessing, and visualization.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, concatenate_datasets
from datasets import Value
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from typing import List, Dict, Tuple, Optional, Union


def unify_dataset(ds, domain_name, samples_per_domain=100, text_field="text"):
    """
    Unify dataset format for multi-domain sentiment analysis.
    
    Args:
        ds: Dataset object
        domain_name: Name of the domain
        samples_per_domain: Number of samples to use per domain
        text_field: Field name containing text
        
    Returns:
        Unified dataset
    """
    if text_field != "text":
        ds = ds.map(lambda x: {"text": x[text_field], "label": x["label"]})
    keep_cols = ["text", "label"]
    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    ds = ds.remove_columns(remove_cols)
    ds = ds.map(lambda x: {"label": int(x["label"])})
    ds = ds.cast_column("label", Value("int64"))
    ds_small = ds.select(range(min(samples_per_domain, len(ds))))
    ds_small = ds_small.add_column("domain", [domain_name] * len(ds_small))
    return ds_small


def load_multidomain_sentiment(samples_per_domain=100):
    """
    Load multi-domain sentiment datasets.
    
    Args:
        samples_per_domain: Number of samples per domain
        
    Returns:
        Concatenated dataset from multiple domains
    """
    imdb_ds = unify_dataset(load_dataset("imdb", split=f"train[:{samples_per_domain}]"), "imdb", samples_per_domain)
    rt_ds = unify_dataset(load_dataset("rotten_tomatoes", split=f"train[:{samples_per_domain}]"), "rotten", samples_per_domain)

    ap_raw = load_dataset("amazon_polarity", split=f"train[:{int(2 * samples_per_domain)}]")
    ap_raw = ap_raw.map(lambda x: {"text": f"{x['title']} {x['content']}".strip()})
    ap_ds = unify_dataset(ap_raw, "amazon", samples_per_domain)

    sst2_ds = load_dataset("glue", "sst2", split=f"train[:{samples_per_domain}]")
    sst2_ds = unify_dataset(sst2_ds, "sst2", samples_per_domain, text_field="sentence")

    tweet_ds = load_dataset("tweet_eval", "sentiment", split=f"train[:{samples_per_domain}]")
    tweet_ds = unify_dataset(tweet_ds, "tweet", samples_per_domain, text_field="text")

    ag_news_ds = load_dataset("ag_news", split=f"train[:{samples_per_domain}]")
    ag_news_ds = unify_dataset(ag_news_ds, "ag_news", samples_per_domain, text_field="text")

    return concatenate_datasets([imdb_ds, rt_ds, ap_ds, sst2_ds, tweet_ds, ag_news_ds])


def compute_clustering_metrics(embeddings, labels):
    """
    Compute clustering quality metrics.
    
    Args:
        embeddings: Embedding matrix
        labels: Cluster labels
        
    Returns:
        Dictionary containing clustering metrics
    """
    silhouette = silhouette_score(embeddings, labels)
    db_index = davies_bouldin_score(embeddings, labels)
    ch_index = calinski_harabasz_score(embeddings, labels)
    
    return {
        'silhouette_score': silhouette,
        'davies_bouldin_index': db_index,
        'calinski_harabasz_index': ch_index
    }


def compute_intrinsic_dimensions(embeddings, labels, variance_threshold=0.75):
    """
    Compute intrinsic dimensions for each cluster.
    
    Args:
        embeddings: Embedding matrix
        labels: Cluster labels
        variance_threshold: Variance threshold for dimension estimation
        
    Returns:
        Dictionary mapping cluster labels to intrinsic dimensions
    """
    intrinsic_dims = {}
    
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        cluster_data = embeddings[indices, :]
        
        if len(cluster_data) < 2:
            intrinsic_dims[label] = 1
        else:
            pca_cluster = PCA()
            pca_cluster.fit(cluster_data)
            cumulative_variance = np.cumsum(pca_cluster.explained_variance_ratio_)
            intrinsic_dims[label] = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)
    
    return intrinsic_dims


def analyze_expert_usage(gating_probs, labels, domains=None):
    """
    Analyze expert usage patterns.
    
    Args:
        gating_probs: Gating probabilities matrix
        labels: Cluster labels
        domains: Domain labels (optional)
        
    Returns:
        Dictionary containing expert usage analysis
    """
    K = gating_probs.shape[1]
    df_gating = pd.DataFrame(gating_probs, columns=[f"Expert_{i}" for i in range(K)])
    df_gating["Label"] = labels
    
    if domains is not None:
        df_gating["Domain"] = domains
    
    expert_cols = [f"Expert_{i}" for i in range(K)]
    avg_gating_per_label = df_gating.groupby("Label")[expert_cols].mean()
    
    if domains is not None:
        avg_gating_per_domain = df_gating.groupby("Domain")[expert_cols].mean()
    else:
        avg_gating_per_domain = None
    
    # Expert assignments
    expert_assignment = gating_probs.argmax(axis=1)
    df_expert_labels = pd.crosstab(expert_assignment, labels, 
                                 rownames=["Expert Assignment"], colnames=["Label"])
    
    return {
        'avg_gating_per_label': avg_gating_per_label,
        'avg_gating_per_domain': avg_gating_per_domain,
        'expert_assignment': expert_assignment,
        'expert_label_crosstab': df_expert_labels,
        'gating_df': df_gating
    }


def create_visualizations(results, model_name, save_plots=False, output_dir="results"):
    """
    Create comprehensive visualizations for experiment results.
    
    Args:
        results: Results dictionary from experiment
        model_name: Name of the model (for file naming)
        save_plots: Whether to save plots to files
        output_dir: Directory to save plots
    """
    if save_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    avg_gating_per_stratum = results['avg_gating_per_stratum']
    avg_gating_per_domain = results['avg_gating_per_domain']
    emb_64d = results['emb_64d']
    strata = results['strata']
    domains = results['domains']
    model = results['model']
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1) Heatmaps of gating probabilities
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Stratum heatmap
    im0 = axes[0].imshow(
        avg_gating_per_stratum.values,
        aspect='auto',
        interpolation='nearest',
        cmap='viridis'
    )
    axes[0].set_title(f"Average Gating Probabilities per Stratum ({model_name})", fontsize=14)
    axes[0].set_xlabel("Expert", fontsize=12)
    axes[0].set_ylabel("Stratum", fontsize=12)
    axes[0].set_xticks(range(model.K))
    axes[0].set_xticklabels(avg_gating_per_stratum.columns, rotation=45, ha='right')
    axes[0].set_yticks(range(len(avg_gating_per_stratum.index)))
    axes[0].set_yticklabels(avg_gating_per_stratum.index)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Domain heatmap
    im1 = axes[1].imshow(
        avg_gating_per_domain.values,
        aspect='auto',
        interpolation='nearest',
        cmap='viridis'
    )
    axes[1].set_title(f"Average Gating Probabilities per Domain ({model_name})", fontsize=14)
    axes[1].set_xlabel("Expert", fontsize=12)
    axes[1].set_ylabel("Domain", fontsize=12)
    axes[1].set_xticks(range(model.K))
    axes[1].set_xticklabels(avg_gating_per_domain.columns, rotation=45, ha='right')
    axes[1].set_yticks(range(len(avg_gating_per_domain.index)))
    axes[1].set_yticklabels(avg_gating_per_domain.index)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_plots:
        plt.savefig(f'{output_dir}/{model_name}_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2) PCA visualization
    pca2 = PCA(n_components=2)
    emb_pca2 = pca2.fit_transform(emb_64d)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        emb_pca2[:, 0],
        emb_pca2[:, 1],
        c=strata,
        cmap='tab10',
        alpha=0.7,
        s=30,
        edgecolors='black',
        linewidth=0.5
    )
    plt.title(f"2D PCA of {model_name} Embeddings", fontsize=16)
    plt.xlabel("PC1", fontsize=14)
    plt.ylabel("PC2", fontsize=14)
    plt.colorbar(scatter, label="Stratum")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plots:
        plt.savefig(f'{output_dir}/{model_name}_pca_2d.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3) UMAP visualization
    umap2 = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(emb_64d)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        umap2[:, 0],
        umap2[:, 1],
        c=strata,
        cmap='tab10',
        alpha=0.7,
        s=30,
        edgecolors='black',
        linewidth=0.5
    )
    plt.title(f"2D UMAP of {model_name} Embeddings", fontsize=16)
    plt.xlabel("UMAP1", fontsize=14)
    plt.ylabel("UMAP2", fontsize=14)
    plt.colorbar(scatter, label="Stratum")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plots:
        plt.savefig(f'{output_dir}/{model_name}_umap_2d.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4) Domain-specific visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    unique_domains = np.unique(domains)
    for i, domain in enumerate(unique_domains):
        if i >= 6:  # Limit to 6 subplots
            break
            
        mask = domains == domain
        domain_pca = emb_pca2[mask]
        domain_strata = strata[mask]
        
        scatter = axes[i].scatter(
            domain_pca[:, 0],
            domain_pca[:, 1],
            c=domain_strata,
            cmap='tab10',
            alpha=0.7,
            s=30
        )
        axes[i].set_title(f"{domain.title()} Domain", fontsize=14)
        axes[i].set_xlabel("PC1", fontsize=12)
        axes[i].set_ylabel("PC2", fontsize=12)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(f"Domain-Specific PCA Visualizations ({model_name})", fontsize=16)
    plt.tight_layout()
    if save_plots:
        plt.savefig(f'{output_dir}/{model_name}_domain_specific.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_experiment_summary(results, model_name):
    """
    Print a comprehensive summary of experiment results.
    
    Args:
        results: Results dictionary from experiment
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Basic statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(results['embeddings'])}")
    print(f"  Embedding dimension: {results['embeddings'].shape[1]}")
    print(f"  Reduced dimension: {results['emb_64d'].shape[1]}")
    print(f"  Number of strata: {len(np.unique(results['strata']))}")
    print(f"  Number of domains: {len(np.unique(results['domains']))}")
    
    # Clustering metrics
    print(f"\nClustering Quality Metrics:")
    metrics = results['clustering_metrics']
    print(f"  Silhouette Score: {metrics['silhouette']:.4f} (higher is better)")
    print(f"  Davies-Bouldin Index: {metrics['davies_bouldin']:.4f} (lower is better)")
    print(f"  Calinski-Harabasz Index: {metrics['calinski_harabasz']:.4f} (higher is better)")
    
    # Intrinsic dimensions
    print(f"\nIntrinsic Dimensions per Stratum:")
    for stratum, dim in results['intrinsic_dims'].items():
        print(f"  Stratum {stratum}: {dim}")
    
    # Expert usage
    print(f"\nExpert Usage Analysis:")
    avg_sparsity = results['avg_sparsity_per_stratum']
    print(f"  Average weighted sparsity per stratum:")
    for stratum, sparsity in avg_sparsity.iterrows():
        print(f"    Stratum {stratum}: {sparsity['Weighted_Sparsity']:.2f}")
    
    # Expert assignment distribution
    expert_assignment = results['expert_assignment']
    unique, counts = np.unique(expert_assignment, return_counts=True)
    print(f"  Expert assignment distribution:")
    for expert, count in zip(unique, counts):
        percentage = (count / len(expert_assignment)) * 100
        print(f"    Expert {expert}: {count} samples ({percentage:.1f}%)")
    
    print(f"\n{'='*60}")


def save_results(results, model_name, output_dir="results"):
    """
    Save experiment results to files.
    
    Args:
        results: Results dictionary from experiment
        model_name: Name of the model
        output_dir: Directory to save results
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save numerical results
    np.save(f"{output_dir}/{model_name}_embeddings.npy", results['embeddings'])
    np.save(f"{output_dir}/{model_name}_emb_64d.npy", results['emb_64d'])
    np.save(f"{output_dir}/{model_name}_strata.npy", results['strata'])
    np.save(f"{output_dir}/{model_name}_gating_probs.npy", results['gating_probs'])
    
    # Save DataFrames
    results['avg_gating_per_stratum'].to_csv(f"{output_dir}/{model_name}_gating_per_stratum.csv")
    results['avg_gating_per_domain'].to_csv(f"{output_dir}/{model_name}_gating_per_domain.csv")
    results['avg_sparsity_per_stratum'].to_csv(f"{output_dir}/{model_name}_sparsity_per_stratum.csv")
    results['df_expert_strata'].to_csv(f"{output_dir}/{model_name}_expert_strata_crosstab.csv")
    
    # Save model
    torch.save(results['model'].state_dict(), f"{output_dir}/{model_name}_model.pth")
    
    # Save summary
    summary = {
        'model_name': model_name,
        'clustering_metrics': results['clustering_metrics'],
        'intrinsic_dims': results['intrinsic_dims'],
        'n_samples': len(results['embeddings']),
        'n_strata': len(np.unique(results['strata'])),
        'n_domains': len(np.unique(results['domains']))
    }
    
    import json
    with open(f"{output_dir}/{model_name}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to {output_dir}/")


def load_results(model_name, output_dir="results"):
    """
    Load previously saved experiment results.
    
    Args:
        model_name: Name of the model
        output_dir: Directory containing results
        
    Returns:
        Dictionary containing loaded results
    """
    results = {}
    
    # Load numerical results
    results['embeddings'] = np.load(f"{output_dir}/{model_name}_embeddings.npy")
    results['emb_64d'] = np.load(f"{output_dir}/{model_name}_emb_64d.npy")
    results['strata'] = np.load(f"{output_dir}/{model_name}_strata.npy")
    results['gating_probs'] = np.load(f"{output_dir}/{model_name}_gating_probs.npy")
    
    # Load DataFrames
    results['avg_gating_per_stratum'] = pd.read_csv(f"{output_dir}/{model_name}_gating_per_stratum.csv", index_col=0)
    results['avg_gating_per_domain'] = pd.read_csv(f"{output_dir}/{model_name}_gating_per_domain.csv", index_col=0)
    results['avg_sparsity_per_stratum'] = pd.read_csv(f"{output_dir}/{model_name}_sparsity_per_stratum.csv", index_col=0)
    results['df_expert_strata'] = pd.read_csv(f"{output_dir}/{model_name}_expert_strata_crosstab.csv", index_col=0)
    
    # Load summary
    import json
    with open(f"{output_dir}/{model_name}_summary.json", 'r') as f:
        summary = json.load(f)
    
    results.update(summary)
    
    return results
