"""
RoBERTa Model Implementation for Stratified Manifold Learning

This module contains the RoBERTa-based embedding generation and model training
for experiments on stratified manifolds in large language models.
"""

import os
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
import plotly.subplots as sp
from transformers import RobertaTokenizer, RobertaModel
from tqdm.auto import tqdm
import umap
import matplotlib.pyplot as plt

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize RoBERTa model and tokenizer (lazy loading)
roberta_tokenizer = None
roberta_model = None

def get_roberta_model():
    """Get RoBERTa model and tokenizer (lazy initialization)."""
    global roberta_tokenizer, roberta_model
    if roberta_tokenizer is None:
        roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        roberta_model = RobertaModel.from_pretrained("roberta-base").to(device)
        roberta_model.eval()
    return roberta_tokenizer, roberta_model


def roberta_embed_texts(texts, batch_size=100, max_length=512):
    """
    Generates sentence embeddings using RoBERTa.
    For each batch, tokenizes the texts and computes the mean of the last hidden states.
    
    Args:
        texts: List of text strings
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        
    Returns:
        numpy array of embeddings
    """
    tokenizer, model = get_roberta_model()
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)


def roberta_embed_text(text):
    """
    Generate embedding for a single text using RoBERTa.
    
    Args:
        text: Single text string
        
    Returns:
        numpy array of embedding
    """
    tokenizer, model = get_roberta_model()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze(0)


def embed_sample(sample):
    """Embed a single sample from dataset."""
    return roberta_embed_text(sample["text"])


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


def run_roberta_experiment(samples_per_domain=500, num_epochs=10, lr=1e-3, margin=1.0):
    """
    Run the complete RoBERTa experiment.
    
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
    
    print("Generating RoBERTa based embeddings (batched)...")
    all_embeddings = roberta_embed_texts(texts, batch_size=100, max_length=512)
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
    
    # UMAP 3D projection
    umap_3d = umap.UMAP(n_components=3, random_state=None).fit_transform(emb_64d)
    df_umap = pd.DataFrame(umap_3d, columns=["component_0", "component_1", "component_2"])
    df_umap["Domain"] = domains
    df_umap["Stratum"] = strata
    
    return {
        'model': model,
        'embeddings': all_embeddings,
        'emb_64d': emb_64d,
        'emb_3d': emb_3d,
        'umap_3d': umap_3d,
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
        'df_plot': df_plot,
        'df_umap': df_umap
    }


def visualize_roberta_results(results, save_plots=False):
    """
    Create visualizations for RoBERTa experiment results.
    
    Args:
        results: Results dictionary from run_roberta_experiment
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
        plt.savefig('roberta_heatmap.png', dpi=300, bbox_inches='tight')
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
    plt.title("2D PCA of RoBERTa Embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(sc, label="Stratum")
    plt.tight_layout()
    if save_plots:
        plt.savefig('roberta_pca_2d.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3) UMAP → 2D scatter
    umap2 = umap.UMAP(n_components=2, random_state=42).fit_transform(emb_64d)

    plt.figure(figsize=(6, 6))
    sc = plt.scatter(
        umap2[:, 0],
        umap2[:, 1],
        c=strata,
        cmap='tab10',
        alpha=0.7,
        s=20
    )
    plt.title("2D UMAP of RoBERTa Embeddings")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.colorbar(sc, label="Stratum")
    plt.tight_layout()
    if save_plots:
        plt.savefig('roberta_umap_2d.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Run the experiment
    results = run_roberta_experiment()
    
    # Create visualizations
    visualize_roberta_results(results, save_plots=True)
