"""
GPT-3 Model Implementation for Stratified Manifold Learning

This module contains the GPT-3-based embedding generation and model training
for experiments on stratified manifolds in large language models.
"""

import os
import openai
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, concatenate_datasets
from datasets import Value
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


def gpt3_embed_texts(texts, batch_size=100):
    """
    Generate sentence embeddings using GPT-3 (text-embedding-ada-002) in batches.
    
    Args:
        texts: List of text strings
        batch_size: Batch size for processing
        
    Returns:
        numpy array of embeddings
    """
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Filter out empty strings
        batch = [txt for txt in batch if isinstance(txt, str) and len(txt.strip()) > 0]
        if not batch:
            continue
        response = openai.embeddings.create(input=batch, model="text-embedding-ada-002")
        batch_embeddings = [d.embedding for d in response.data]
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)


def gpt3_embed_text(text):
    """
    Generate embedding for a single text using GPT-3.
    
    Args:
        text: Input text string
        
    Returns:
        GPT-3 embedding as numpy array
    """
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    embedding = response.data[0].embedding
    return np.array(embedding)


def embed_sample(sample):
    """Embed a single sample from dataset."""
    return gpt3_embed_text(sample["text"])


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


def run_gpt3_experiment(samples_per_domain=500, num_epochs=100, lr=1e-3, margin=1.0):
    """
    Run the complete GPT-3 experiment.
    
    Args:
        samples_per_domain: Number of samples per domain
        num_epochs: Number of training epochs
        lr: Learning rate
        margin: Margin for contrastive loss
        
    Returns:
        Dictionary containing results and model
    """
    from src.geometric_tools.moe_models import TopKSTE, topk_st, LISTALayer, DictionaryExpertLISTA, GatingNetworkAttention, MixtureOfDictionaryExperts
    from src.geometric_tools.training import contrastive_loss_with_labels, train_contrastive_moe_with_labels
    
    print("Using device:", device)
    combined_ds = load_multidomain_sentiment(samples_per_domain=samples_per_domain)
    texts = combined_ds["text"]
    domains = combined_ds["domain"]
    labels = torch.tensor(combined_ds["label"], dtype=torch.long)
    
    print("Generating GPT-3 embeddings...")
    all_embeddings = gpt3_embed_texts(texts, batch_size=100)
    print("All embeddings shape:", all_embeddings.shape)
    
    # Use PCA for dimensionality reduction
    pca = PCA(n_components=64)
    emb_64d = pca.fit_transform(all_embeddings)
    print("Reduced shape:", emb_64d.shape)
    
    X_tensor = torch.tensor(emb_64d, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize model
    model = MixtureOfDictionaryExperts(
        input_dim=64,
        query_dim=128,
        code_dim=32,
        K=7,
        projection_dim=64,
        num_lista_layers=5,
        sparsity_levels=[8, 12, 16, 20, 24, 28, 32],
        threshold=0.5
    )
    
    # Train model
    train_contrastive_moe_with_labels(model, loader, num_epochs=num_epochs, lr=lr, margin=margin, device=device)
    
    # Perform clustering analysis
    kmeans = KMeans(n_clusters=5, random_state=42)
    strata = kmeans.fit_predict(emb_64d)
    
    print("\nIntrinsic Dimensions of Each Stratum (75% Variance Explained):")
    intrinsic_dims = {}
    for s in np.unique(strata):
        indices = np.where(strata == s)[0]
        data_cluster = emb_64d[indices, :]
        if len(data_cluster) < 2:
            intrinsic_dims[s] = 1
        else:
            pca_cluster = PCA()
            pca_cluster.fit(data_cluster)
            cumulative_variance = np.cumsum(pca_cluster.explained_variance_ratio_)
            intrinsic_dims[s] = int(np.searchsorted(cumulative_variance, 0.75) + 1)
        print(f"Stratum {s}: Intrinsic dimension = {intrinsic_dims[s]}")
    
    # Calculate clustering metrics
    silhouette = silhouette_score(emb_64d, strata)
    db_index = davies_bouldin_score(emb_64d, strata)
    ch_index = calinski_harabasz_score(emb_64d, strata)
    
    print("\nStratum Separation Measures:")
    print(f"Silhouette Score: {silhouette:.4f} (higher is better)")
    print(f"Davies-Bouldin Index: {db_index:.4f} (lower is better)")
    print(f"Calinski-Harabasz Index: {ch_index:.4f} (higher is better)")
    
    # Analyze gating probabilities
    with torch.no_grad():
        gating_probs_all = model.gating_net(X_tensor.to(device)).cpu().numpy()
    
    df_gating = pd.DataFrame(gating_probs_all, columns=[f"Expert_{i}" for i in range(model.K)])
    df_gating["Stratum"] = strata
    df_gating["Domain"] = domains
    
    expert_cols = [f"Expert_{i}" for i in range(model.K)]
    avg_gating_per_stratum = df_gating.groupby("Stratum")[expert_cols].mean()
    avg_gating_per_domain = df_gating.groupby("Domain")[expert_cols].mean()
    
    # Calculate weighted sparsity
    sparsity_params = np.array([expert.sparsity_level for expert in model.experts])
    weighted_sparsity = np.dot(gating_probs_all, sparsity_params)
    df_weighted = pd.DataFrame({"Weighted_Sparsity": weighted_sparsity, "Stratum": strata})
    avg_sparsity_per_stratum = df_weighted.groupby("Stratum").mean()
    
    print("\nAverage weighted sparsity level per stratum:")
    print(avg_sparsity_per_stratum)
    
    # Expert assignment analysis
    expert_assignment = gating_probs_all.argmax(axis=1)
    df_expert_strata = pd.crosstab(expert_assignment, strata, rownames=["Expert Assignment"], colnames=["Stratum"])
    print("\nExpert Assignment vs Stratum (Contingency Table):")
    print(df_expert_strata)
    
    # Gating network keys analysis
    keys = model.gating_net.keys.detach().cpu().numpy()
    print("\nGating Network Keys (norms):")
    for i, key in enumerate(keys):
        print(f"Expert {i} key norm: {np.linalg.norm(key):.4f}")
    
    # Prepare visualization data
    pca_3d = PCA(n_components=3)
    emb_3d = pca_3d.fit_transform(emb_64d)
    domain_label = [f"{d}_{l}" for d, l in zip(domains, labels)]
    
    df_plot = pd.DataFrame({
        "x": emb_3d[:, 0],
        "y": emb_3d[:, 1],
        "z": emb_3d[:, 2],
        "domain": domains,
        "label": labels,
        "Stratum": strata,
        "domain_label": domain_label
    })
    
    return {
        'model': model,
        'embeddings': all_embeddings,
        'emb_64d': emb_64d,
        'emb_3d': emb_3d,
        'strata': strata,
        'domains': domains,
        'labels': labels,
        'intrinsic_dims': intrinsic_dims,
        'clustering_metrics': {
            'silhouette': silhouette,
            'davies_bouldin': db_index,
            'calinski_harabasz': ch_index
        },
        'gating_probs': gating_probs_all,
        'avg_gating_per_stratum': avg_gating_per_stratum,
        'avg_gating_per_domain': avg_gating_per_domain,
        'avg_sparsity_per_stratum': avg_sparsity_per_stratum,
        'expert_assignment': expert_assignment,
        'df_expert_strata': df_expert_strata,
        'df_plot': df_plot
    }


def visualize_gpt3_results(results, save_plots=False):
    """
    Create visualizations for GPT-3 experiment results.
    
    Args:
        results: Results dictionary from run_gpt3_experiment
        save_plots: Whether to save plots to files
    """
    avg_gating_per_stratum = results['avg_gating_per_stratum']
    avg_gating_per_domain = results['avg_gating_per_domain']
    emb_64d = results['emb_64d']
    strata = results['strata']
    model = results['model']
    
    # 1) Heatmaps of gating probabilities (2D)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Stratum heatmap
    im0 = axes[0].imshow(
        avg_gating_per_stratum.values,
        aspect='auto',
        interpolation='nearest'
    )
    axes[0].set_title("Avg Gating per Stratum")
    axes[0].set_xlabel("Expert")
    axes[0].set_ylabel("Stratum")
    axes[0].set_xticks(range(model.K))
    axes[0].set_xticklabels(avg_gating_per_stratum.columns, rotation=45, ha='right')
    axes[0].set_yticks(range(len(avg_gating_per_stratum.index)))
    axes[0].set_yticklabels(avg_gating_per_stratum.index)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Domain heatmap
    im1 = axes[1].imshow(
        avg_gating_per_domain.values,
        aspect='auto',
        interpolation='nearest'
    )
    axes[1].set_title("Avg Gating per Domain")
    axes[1].set_xlabel("Expert")
    axes[1].set_ylabel("Domain")
    axes[1].set_xticks(range(model.K))
    axes[1].set_xticklabels(avg_gating_per_domain.columns, rotation=45, ha='right')
    axes[1].set_yticks(range(len(avg_gating_per_domain.index)))
    axes[1].set_yticklabels(avg_gating_per_domain.index)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_plots:
        plt.savefig('gpt3_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2) PCA → 2D scatter
    pca2 = PCA(n_components=2)
    emb_pca2 = pca2.fit_transform(emb_64d)

    plt.figure(figsize=(6, 6))
    sc = plt.scatter(
        emb_pca2[:, 0],
        emb_pca2[:, 1],
        c=strata,
        cmap='tab10',
        alpha=0.7,
        s=20
    )
    plt.title("2D PCA of GPT-3 Embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(sc, label="Stratum")
    plt.tight_layout()
    if save_plots:
        plt.savefig('gpt3_pca_2d.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Run the experiment
    results = run_gpt3_experiment()
    
    # Create visualizations
    visualize_gpt3_results(results, save_plots=True)
