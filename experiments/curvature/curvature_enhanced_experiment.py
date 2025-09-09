"""
Enhanced experiment with curvature-based geometric analysis.
Integrates Mixed-curvature VAE concepts for latent embedding evaluation.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, concatenate_datasets
from datasets import Value
from tqdm.auto import tqdm
import json
from datetime import datetime

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

    print("Loading datasets...")
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

def roberta_embed_texts(texts, batch_size=32):
    """Generate RoBERTa embeddings."""
    from transformers import RobertaTokenizer, RobertaModel
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base").to(device)
    model.eval()
    
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="RoBERTa embeddings"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

class SimpleMoE(nn.Module):
    """Simplified Mixture of Experts for demonstration."""
    def __init__(self, input_dim=768, hidden_dim=256, num_experts=8, top_k=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
        
    def forward(self, x):
        # Gating
        gate_scores = self.gate(x)
        gate_probs = torch.softmax(gate_scores, dim=-1)
        
        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Expert outputs
        expert_outputs = []
        for i in range(self.num_experts):
            expert_outputs.append(self.experts[i](x))
        
        # Weighted combination
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_weight = top_k_probs[:, i:i+1]
            expert_output = torch.stack([expert_outputs[j][k] for k, j in enumerate(expert_idx)])
            output += expert_weight * expert_output
        
        return output, gate_probs

def train_moe_model(embeddings, labels, domains, num_epochs=20, lr=1e-3):
    """Train MoE model on embeddings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Convert to tensors
    X = torch.FloatTensor(embeddings).to(device)
    y = torch.LongTensor(labels).to(device)
    
    # Initialize model
    model = SimpleMoE(input_dim=embeddings.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    losses = []
    
    for epoch in tqdm(range(num_epochs), desc="Training MoE"):
        optimizer.zero_grad()
        
        # Forward pass
        output, gate_probs = model(X)
        
        # Simple reconstruction loss
        reconstruction_loss = nn.MSELoss()(output, X)
        
        # Diversity loss to encourage expert specialization
        diversity_loss = -torch.mean(torch.sum(gate_probs * torch.log(gate_probs + 1e-8), dim=-1))
        
        # Total loss
        total_loss = reconstruction_loss + 0.1 * diversity_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        losses.append(total_loss.item())
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")
    
    return model, losses

def run_curvature_enhanced_experiment(samples_per_domain=200, num_clusters=6, num_epochs=15):
    """
    Run enhanced experiment with curvature-based geometric analysis.
    
    Inspired by: Mixed-curvature Variational Autoencoders (arXiv:1911.08411)
    """
    print("🔬 Running Curvature-Enhanced Experiment")
    print("=" * 60)
    print("Inspired by: Mixed-curvature Variational Autoencoders (arXiv:1911.08411)")
    print("=" * 60)
    
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
    all_embeddings = roberta_embed_texts(texts, batch_size=32)
    print(f"Embeddings shape: {all_embeddings.shape}")
    
    # Apply PCA
    print("Applying PCA dimensionality reduction...")
    n_components = min(128, all_embeddings.shape[0], all_embeddings.shape[1])
    pca = PCA(n_components=n_components)
    emb_reduced = pca.fit_transform(all_embeddings)
    print(f"Reduced embeddings shape: {emb_reduced.shape}")
    
    # Clustering
    print("Performing clustering...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    strata = kmeans.fit_predict(emb_reduced)
    
    # Calculate metrics
    silhouette = silhouette_score(emb_reduced, strata)
    db_index = davies_bouldin_score(emb_reduced, strata)
    ch_index = calinski_harabasz_score(emb_reduced, strata)
    
    print(f"\nClustering Results:")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Davies-Bouldin Index: {db_index:.4f}")
    print(f"  Calinski-Harabasz Index: {ch_index:.4f}")
    
    # Train MoE model
    print(f"\nTraining MoE model...")
    moe_model, training_losses = train_moe_model(all_embeddings, labels, domains, num_epochs=num_epochs)
    
    # Get MoE-enhanced embeddings
    print("Generating MoE-enhanced embeddings...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    moe_model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(all_embeddings).to(device)
        moe_embeddings, gate_probs = moe_model(X_tensor)
        moe_embeddings = moe_embeddings.cpu().numpy()
        gate_probs = gate_probs.cpu().numpy()
    
    # Apply PCA to MoE embeddings
    n_components_moe = min(128, moe_embeddings.shape[0], moe_embeddings.shape[1])
    pca_moe = PCA(n_components=n_components_moe)
    emb_moe_reduced = pca_moe.fit_transform(moe_embeddings)
    
    # Clustering on MoE embeddings
    kmeans_moe = KMeans(n_clusters=num_clusters, random_state=42)
    strata_moe = kmeans_moe.fit_predict(emb_moe_reduced)
    
    # Calculate MoE metrics
    silhouette_moe = silhouette_score(emb_moe_reduced, strata_moe)
    db_index_moe = davies_bouldin_score(emb_moe_reduced, strata_moe)
    ch_index_moe = calinski_harabasz_score(emb_moe_reduced, strata_moe)
    
    print(f"\nMoE-Enhanced Clustering Results:")
    print(f"  Silhouette Score: {silhouette_moe:.4f}")
    print(f"  Davies-Bouldin Index: {db_index_moe:.4f}")
    print(f"  Calinski-Harabasz Index: {ch_index_moe:.4f}")
    
    # CURVATURE ANALYSIS
    print(f"\n{'='*60}")
    print("🔬 CURVATURE-BASED GEOMETRIC ANALYSIS")
    print(f"{'='*60}")
    
    from src.geometric_tools.curvature_analysis import CurvatureAnalyzer, run_curvature_analysis
    
    # Analyze original embeddings
    print("Analyzing curvature of original embeddings...")
    original_analyzer = CurvatureAnalyzer(emb_reduced, labels=labels, domains=domains)
    original_curvature_signature = original_analyzer.compute_mixed_curvature_signature()
    
    # Analyze MoE-enhanced embeddings
    print("Analyzing curvature of MoE-enhanced embeddings...")
    moe_analyzer = CurvatureAnalyzer(emb_moe_reduced, labels=labels, domains=domains)
    moe_curvature_signature = moe_analyzer.compute_mixed_curvature_signature()
    
    # Analyze curvature by strata
    print("Analyzing curvature by strata...")
    original_stratum_curvature = original_analyzer.analyze_curvature_by_strata(strata)
    moe_stratum_curvature = moe_analyzer.analyze_curvature_by_strata(strata_moe)
    
    # Print curvature analysis results
    print(f"\nCurvature Analysis Results:")
    print(f"  Original Embeddings:")
    print(f"    Riemannian Curvature: {original_curvature_signature['curvature_mean']:.4f} ± {original_curvature_signature['curvature_std']:.4f}")
    print(f"    Sectional Curvature: {original_curvature_signature['sectional_curvature'].mean():.4f} ± {original_curvature_signature['sectional_curvature'].std():.4f}")
    print(f"    Gaussian Curvature: {original_curvature_signature['gaussian_curvature'].mean():.4f} ± {original_curvature_signature['gaussian_curvature'].std():.4f}")
    print(f"    Mean Curvature: {original_curvature_signature['mean_curvature'].mean():.4f} ± {original_curvature_signature['mean_curvature'].std():.4f}")
    
    print(f"\n  MoE-Enhanced Embeddings:")
    print(f"    Riemannian Curvature: {moe_curvature_signature['curvature_mean']:.4f} ± {moe_curvature_signature['curvature_std']:.4f}")
    print(f"    Sectional Curvature: {moe_curvature_signature['sectional_curvature'].mean():.4f} ± {moe_curvature_signature['sectional_curvature'].std():.4f}")
    print(f"    Gaussian Curvature: {moe_curvature_signature['gaussian_curvature'].mean():.4f} ± {moe_curvature_signature['gaussian_curvature'].std():.4f}")
    print(f"    Mean Curvature: {moe_curvature_signature['mean_curvature'].mean():.4f} ± {moe_curvature_signature['mean_curvature'].std():.4f}")
    
    # Curvature by stratum analysis
    print(f"\nCurvature Analysis by Stratum:")
    print(f"  Original Embeddings:")
    for stratum, analysis in original_stratum_curvature.items():
        print(f"    Stratum {stratum}: Riemannian = {analysis['curvature_mean']:.4f} ± {analysis['curvature_std']:.4f}")
    
    print(f"\n  MoE-Enhanced Embeddings:")
    for stratum, analysis in moe_stratum_curvature.items():
        print(f"    Stratum {stratum}: Riemannian = {analysis['curvature_mean']:.4f} ± {analysis['curvature_std']:.4f}")
    
    # Create comprehensive visualizations
    print("\nCreating comprehensive visualizations...")
    
    # PCA 2D visualization
    pca_2d = PCA(n_components=2)
    emb_2d = pca_2d.fit_transform(emb_reduced)
    emb_moe_2d = pca_2d.fit_transform(emb_moe_reduced)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    # Original embeddings
    scatter1 = axes[0, 0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=strata, cmap='tab10', alpha=0.7, s=30)
    axes[0, 0].set_title("Original: Colored by Strata")
    axes[0, 0].set_xlabel("PC1")
    axes[0, 0].set_ylabel("PC2")
    plt.colorbar(scatter1, ax=axes[0, 0], label="Stratum")
    
    domain_colors = {domain: i for i, domain in enumerate(set(domains))}
    domain_numeric = [domain_colors[d] for d in domains]
    scatter2 = axes[0, 1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=domain_numeric, cmap='viridis', alpha=0.7, s=30)
    axes[0, 1].set_title("Original: Colored by Domain")
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")
    plt.colorbar(scatter2, ax=axes[0, 1], label="Domain")
    
    scatter3 = axes[0, 2].scatter(emb_2d[:, 0], emb_2d[:, 1], c=original_curvature_signature['riemannian_curvature'], cmap='plasma', alpha=0.7, s=30)
    axes[0, 2].set_title("Original: Riemannian Curvature")
    axes[0, 2].set_xlabel("PC1")
    axes[0, 2].set_ylabel("PC2")
    plt.colorbar(scatter3, ax=axes[0, 2], label="Curvature")
    
    # MoE-enhanced embeddings
    scatter4 = axes[1, 0].scatter(emb_moe_2d[:, 0], emb_moe_2d[:, 1], c=strata_moe, cmap='tab10', alpha=0.7, s=30)
    axes[1, 0].set_title("MoE-Enhanced: Colored by Strata")
    axes[1, 0].set_xlabel("PC1")
    axes[1, 0].set_ylabel("PC2")
    plt.colorbar(scatter4, ax=axes[1, 0], label="Stratum")
    
    scatter5 = axes[1, 1].scatter(emb_moe_2d[:, 0], emb_moe_2d[:, 1], c=domain_numeric, cmap='viridis', alpha=0.7, s=30)
    axes[1, 1].set_title("MoE-Enhanced: Colored by Domain")
    axes[1, 1].set_xlabel("PC1")
    axes[1, 1].set_ylabel("PC2")
    plt.colorbar(scatter5, ax=axes[1, 1], label="Domain")
    
    scatter6 = axes[1, 2].scatter(emb_moe_2d[:, 0], emb_moe_2d[:, 1], c=moe_curvature_signature['riemannian_curvature'], cmap='plasma', alpha=0.7, s=30)
    axes[1, 2].set_title("MoE-Enhanced: Riemannian Curvature")
    axes[1, 2].set_xlabel("PC1")
    axes[1, 2].set_ylabel("PC2")
    plt.colorbar(scatter6, ax=axes[1, 2], label="Curvature")
    
    # Curvature comparison plots
    scatter7 = axes[2, 0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=original_curvature_signature['gaussian_curvature'], cmap='coolwarm', alpha=0.7, s=30)
    axes[2, 0].set_title("Original: Gaussian Curvature")
    axes[2, 0].set_xlabel("PC1")
    axes[2, 0].set_ylabel("PC2")
    plt.colorbar(scatter7, ax=axes[2, 0], label="Curvature")
    
    scatter8 = axes[2, 1].scatter(emb_moe_2d[:, 0], emb_moe_2d[:, 1], c=moe_curvature_signature['gaussian_curvature'], cmap='coolwarm', alpha=0.7, s=30)
    axes[2, 1].set_title("MoE-Enhanced: Gaussian Curvature")
    axes[2, 1].set_xlabel("PC1")
    axes[2, 1].set_ylabel("PC2")
    plt.colorbar(scatter8, ax=axes[2, 1], label="Curvature")
    
    # Curvature difference
    curvature_diff = moe_curvature_signature['riemannian_curvature'] - original_curvature_signature['riemannian_curvature']
    scatter9 = axes[2, 2].scatter(emb_2d[:, 0], emb_2d[:, 1], c=curvature_diff, cmap='RdBu_r', alpha=0.7, s=30)
    axes[2, 2].set_title("Curvature Change (MoE - Original)")
    axes[2, 2].set_xlabel("PC1")
    axes[2, 2].set_ylabel("PC2")
    plt.colorbar(scatter9, ax=axes[2, 2], label="Curvature Change")
    
    plt.tight_layout()
    plt.savefig('curvature_enhanced_experiment.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Training loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses)
    plt.title("MoE Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig('moe_training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Expert usage analysis
    plt.figure(figsize=(12, 8))
    expert_usage = np.mean(gate_probs, axis=0)
    plt.bar(range(len(expert_usage)), expert_usage)
    plt.title("Expert Usage Distribution")
    plt.xlabel("Expert Index")
    plt.ylabel("Average Usage Probability")
    plt.grid(True, alpha=0.3)
    plt.savefig('expert_usage.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save comprehensive results
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_size': len(texts),
        'num_domains': len(set(domains)),
        'domains': list(set(domains)),
        'original_embeddings': all_embeddings.tolist(),
        'moe_embeddings': moe_embeddings.tolist(),
        'original_strata': strata.tolist(),
        'moe_strata': strata_moe.tolist(),
        'original_metrics': {
            'silhouette': float(silhouette),
            'davies_bouldin': float(db_index),
            'calinski_harabasz': float(ch_index)
        },
        'moe_metrics': {
            'silhouette': float(silhouette_moe),
            'davies_bouldin': float(db_index_moe),
            'calinski_harabasz': float(ch_index_moe)
        },
        'original_curvature_signature': {
            'riemannian_curvature_mean': float(original_curvature_signature['curvature_mean']),
            'riemannian_curvature_std': float(original_curvature_signature['curvature_std']),
            'sectional_curvature_mean': float(original_curvature_signature['sectional_curvature'].mean()),
            'gaussian_curvature_mean': float(original_curvature_signature['gaussian_curvature'].mean()),
            'mean_curvature_mean': float(original_curvature_signature['mean_curvature'].mean())
        },
        'moe_curvature_signature': {
            'riemannian_curvature_mean': float(moe_curvature_signature['curvature_mean']),
            'riemannian_curvature_std': float(moe_curvature_signature['curvature_std']),
            'sectional_curvature_mean': float(moe_curvature_signature['sectional_curvature'].mean()),
            'gaussian_curvature_mean': float(moe_curvature_signature['gaussian_curvature'].mean()),
            'mean_curvature_mean': float(moe_curvature_signature['mean_curvature'].mean())
        },
        'training_losses': [float(x) for x in training_losses],
        'expert_usage': expert_usage.tolist()
    }
    
    # Save results to JSON
    with open('curvature_enhanced_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Curvature-enhanced experiment completed successfully!")
    print(f"   Results saved to 'curvature_enhanced_results.json'")
    print(f"   Visualizations saved:")
    print(f"     - curvature_enhanced_experiment.png")
    print(f"     - moe_training_loss.png")
    print(f"     - expert_usage.png")
    
    return results

if __name__ == "__main__":
    # Run curvature-enhanced experiment
    results = run_curvature_enhanced_experiment(samples_per_domain=150, num_clusters=5, num_epochs=20)
