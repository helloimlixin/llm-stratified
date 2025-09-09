"""
Working experiment script that avoids segmentation faults.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from datasets import load_dataset, concatenate_datasets
from datasets import Value

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def load_multidomain_sentiment(samples_per_domain=100):
    """Load multi-domain sentiment datasets."""
    def unify_dataset(ds, domain_name, samples_per_domain=100, text_field="text"):
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

    # Load datasets
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

def roberta_embed_texts(texts, batch_size=10):
    """Generate RoBERTa embeddings with lazy loading."""
    from transformers import RobertaTokenizer, RobertaModel
    from tqdm.auto import tqdm
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base").to(device)
    model.eval()
    
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

def run_working_experiment(samples_per_domain=50, num_clusters=5):
    """Run a working experiment."""
    print("🚀 Running Working Experiment")
    print("=" * 40)
    
    # Load data
    print("Loading multi-domain sentiment data...")
    combined_ds = load_multidomain_sentiment(samples_per_domain=samples_per_domain)
    texts = combined_ds["text"]
    domains = combined_ds["domain"]
    labels = combined_ds["label"]
    
    print(f"Loaded {len(texts)} samples from {len(set(domains))} domains")
    print(f"Domains: {set(domains)}")
    
    # Generate embeddings
    print("Generating RoBERTa embeddings...")
    all_embeddings = roberta_embed_texts(texts, batch_size=10)
    print(f"Embeddings shape: {all_embeddings.shape}")
    
    # Apply PCA
    print("Applying PCA dimensionality reduction...")
    n_components = min(64, all_embeddings.shape[0], all_embeddings.shape[1])
    pca = PCA(n_components=n_components)
    emb_64d = pca.fit_transform(all_embeddings)
    print(f"Reduced embeddings shape: {emb_64d.shape}")
    
    # Clustering
    print("Performing clustering...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    strata = kmeans.fit_predict(emb_64d)
    
    # Calculate metrics
    silhouette = silhouette_score(emb_64d, strata)
    db_index = davies_bouldin_score(emb_64d, strata)
    ch_index = calinski_harabasz_score(emb_64d, strata)
    
    print(f"\nClustering Results:")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Davies-Bouldin Index: {db_index:.4f}")
    print(f"  Calinski-Harabasz Index: {ch_index:.4f}")
    
    # Intrinsic dimensions
    print(f"\nIntrinsic Dimensions per Stratum:")
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
        print(f"  Stratum {s}: {intrinsic_dims[s]} dimensions")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # PCA 2D visualization
    pca_2d = PCA(n_components=2)
    emb_2d = pca_2d.fit_transform(emb_64d)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: PCA colored by strata
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=strata, cmap='tab10', alpha=0.7, s=30)
    plt.title("PCA: Colored by Strata")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter1, label="Stratum")
    
    # Plot 2: PCA colored by domains
    plt.subplot(1, 2, 2)
    domain_colors = {domain: i for i, domain in enumerate(set(domains))}
    domain_numeric = [domain_colors[d] for d in domains]
    scatter2 = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=domain_numeric, cmap='viridis', alpha=0.7, s=30)
    plt.title("PCA: Colored by Domain")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter2, label="Domain")
    
    plt.tight_layout()
    plt.savefig('working_experiment.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Domain-stratum analysis
    print(f"\nDomain-Stratum Analysis:")
    df_analysis = pd.DataFrame({
        'domain': domains,
        'stratum': strata,
        'label': labels
    })
    
    domain_stratum_crosstab = pd.crosstab(df_analysis['domain'], df_analysis['stratum'])
    print("\nDomain vs Stratum Crosstab:")
    print(domain_stratum_crosstab)
    
    # Save results
    results = {
        'embeddings': all_embeddings,
        'emb_64d': emb_64d,
        'strata': strata,
        'domains': domains,
        'labels': labels,
        'intrinsic_dims': intrinsic_dims,
        'clustering_metrics': {
            'silhouette': silhouette,
            'davies_bouldin': db_index,
            'calinski_harabasz': ch_index
        },
        'domain_stratum_crosstab': domain_stratum_crosstab
    }
    
    print(f"\n✅ Experiment completed successfully!")
    print(f"   Results saved to memory")
    print(f"   Visualization saved as 'working_experiment.png'")
    
    return results

if __name__ == "__main__":
    # Run experiment with small dataset
    results = run_working_experiment(samples_per_domain=30, num_clusters=3)
