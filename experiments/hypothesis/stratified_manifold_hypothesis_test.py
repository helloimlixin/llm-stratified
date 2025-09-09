"""
Enhanced experiment testing the stratified manifold hypothesis.
Specifically tests for abrupt curvature changes between strata.
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

def run_stratified_manifold_hypothesis_test(samples_per_domain=200, num_clusters=6, num_epochs=15):
    """
    Run experiment testing the stratified manifold hypothesis.
    
    Specifically tests for abrupt curvature changes between strata.
    """
    print("🔬 Testing Stratified Manifold Hypothesis")
    print("=" * 60)
    print("Hypothesis: There should be abrupt changes in curvature between strata")
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
    
    # CURVATURE DISCONTINUITY ANALYSIS
    print(f"\n{'='*60}")
    print("🔬 CURVATURE DISCONTINUITY ANALYSIS")
    print(f"{'='*60}")
    
    from src.geometric_tools.curvature_discontinuity_analysis import run_curvature_discontinuity_analysis
    
    # Test hypothesis on original embeddings
    print("Testing stratified manifold hypothesis on original embeddings...")
    original_hypothesis_results = run_curvature_discontinuity_analysis(
        emb_reduced, strata, labels=labels, domains=domains,
        save_path='original_curvature_discontinuity.png'
    )
    
    # Test hypothesis on MoE-enhanced embeddings
    print("\nTesting stratified manifold hypothesis on MoE-enhanced embeddings...")
    moe_hypothesis_results = run_curvature_discontinuity_analysis(
        emb_moe_reduced, strata_moe, labels=labels, domains=domains,
        save_path='moe_curvature_discontinuity.png'
    )
    
    # Compare hypothesis test results
    print(f"\n{'='*60}")
    print("📊 HYPOTHESIS TEST COMPARISON")
    print(f"{'='*60}")
    
    print(f"Original Embeddings:")
    print(f"  Hypothesis Supported: {original_hypothesis_results['hypothesis_supported']}")
    print(f"  Effect Size (Cohen's d): {original_hypothesis_results['statistics']['cohens_d']:.4f}")
    print(f"  T-test p-value: {original_hypothesis_results['statistics']['t_pvalue']:.6f}")
    
    print(f"\nMoE-Enhanced Embeddings:")
    print(f"  Hypothesis Supported: {moe_hypothesis_results['hypothesis_supported']}")
    print(f"  Effect Size (Cohen's d): {moe_hypothesis_results['statistics']['cohens_d']:.4f}")
    print(f"  T-test p-value: {moe_hypothesis_results['statistics']['t_pvalue']:.6f}")
    
    # Analyze discontinuity scores
    print(f"\nDiscontinuity Scores Comparison:")
    if original_hypothesis_results['discontinuity_scores'] and moe_hypothesis_results['discontinuity_scores']:
        print(f"  Original Average: {np.mean(list(original_hypothesis_results['discontinuity_scores'].values())):.4f}")
        print(f"  MoE Average: {np.mean(list(moe_hypothesis_results['discontinuity_scores'].values())):.4f}")
        
        # Check if MoE increases or decreases discontinuities
        original_avg = np.mean(list(original_hypothesis_results['discontinuity_scores'].values()))
        moe_avg = np.mean(list(moe_hypothesis_results['discontinuity_scores'].values()))
        
        if moe_avg > original_avg:
            print(f"  MoE Effect: INCREASES discontinuities (+{((moe_avg/original_avg - 1) * 100):.1f}%)")
        else:
            print(f"  MoE Effect: DECREASES discontinuities ({((moe_avg/original_avg - 1) * 100):.1f}%)")
    
    # Create comprehensive comparison visualization
    print("\nCreating comprehensive comparison visualization...")
    
    # PCA 2D visualization
    pca_2d = PCA(n_components=2)
    emb_2d = pca_2d.fit_transform(emb_reduced)
    emb_moe_2d = pca_2d.fit_transform(emb_moe_reduced)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
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
    
    # MoE-enhanced embeddings
    scatter3 = axes[0, 2].scatter(emb_moe_2d[:, 0], emb_moe_2d[:, 1], c=strata_moe, cmap='tab10', alpha=0.7, s=30)
    axes[0, 2].set_title("MoE-Enhanced: Colored by Strata")
    axes[0, 2].set_xlabel("PC1")
    axes[0, 2].set_ylabel("PC2")
    plt.colorbar(scatter3, ax=axes[0, 2], label="Stratum")
    
    scatter4 = axes[0, 3].scatter(emb_moe_2d[:, 0], emb_moe_2d[:, 1], c=domain_numeric, cmap='viridis', alpha=0.7, s=30)
    axes[0, 3].set_title("MoE-Enhanced: Colored by Domain")
    axes[0, 3].set_xlabel("PC1")
    axes[0, 3].set_ylabel("PC2")
    plt.colorbar(scatter4, ax=axes[0, 3], label="Domain")
    
    # Hypothesis test results visualization
    models = ['Original', 'MoE-Enhanced']
    hypothesis_supported = [original_hypothesis_results['hypothesis_supported'], 
                           moe_hypothesis_results['hypothesis_supported']]
    effect_sizes = [original_hypothesis_results['statistics']['cohens_d'],
                   moe_hypothesis_results['statistics']['cohens_d']]
    
    # Hypothesis support bar chart
    colors = ['green' if h else 'red' for h in hypothesis_supported]
    bars = axes[1, 0].bar(models, [1 if h else 0 for h in hypothesis_supported], color=colors, alpha=0.7)
    axes[1, 0].set_title("Hypothesis Support")
    axes[1, 0].set_ylabel("Supported (1) / Not Supported (0)")
    axes[1, 0].set_ylim(0, 1.2)
    
    # Add text labels
    for i, (bar, supported) in enumerate(zip(bars, hypothesis_supported)):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f"{'Supported' if supported else 'Not Supported'}",
                       ha='center', va='bottom', fontweight='bold')
    
    # Effect size comparison
    bars2 = axes[1, 1].bar(models, effect_sizes, color=['blue', 'orange'], alpha=0.7)
    axes[1, 1].set_title("Effect Size (Cohen's d)")
    axes[1, 1].set_ylabel("Cohen's d")
    axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Medium Effect')
    axes[1, 1].legend()
    
    # Add value labels
    for bar, effect in zip(bars2, effect_sizes):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f"{effect:.3f}", ha='center', va='bottom')
    
    # Discontinuity scores comparison
    if original_hypothesis_results['discontinuity_scores'] and moe_hypothesis_results['discontinuity_scores']:
        orig_scores = list(original_hypothesis_results['discontinuity_scores'].values())
        moe_scores = list(moe_hypothesis_results['discontinuity_scores'].values())
        
        axes[1, 2].boxplot([orig_scores, moe_scores], labels=models)
        axes[1, 2].set_title("Discontinuity Scores Distribution")
        axes[1, 2].set_ylabel("Discontinuity Score")
        axes[1, 2].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[1, 3].text(0.1, 0.8, f"Original Embeddings:", fontweight='bold', transform=axes[1, 3].transAxes)
    axes[1, 3].text(0.1, 0.7, f"  Hypothesis: {'✓ Supported' if original_hypothesis_results['hypothesis_supported'] else '✗ Not Supported'}", 
                   transform=axes[1, 3].transAxes)
    axes[1, 3].text(0.1, 0.6, f"  Effect Size: {original_hypothesis_results['statistics']['cohens_d']:.3f}", 
                   transform=axes[1, 3].transAxes)
    axes[1, 3].text(0.1, 0.5, f"  P-value: {original_hypothesis_results['statistics']['t_pvalue']:.6f}", 
                   transform=axes[1, 3].transAxes)
    
    axes[1, 3].text(0.1, 0.3, f"MoE-Enhanced Embeddings:", fontweight='bold', transform=axes[1, 3].transAxes)
    axes[1, 3].text(0.1, 0.2, f"  Hypothesis: {'✓ Supported' if moe_hypothesis_results['hypothesis_supported'] else '✗ Not Supported'}", 
                   transform=axes[1, 3].transAxes)
    axes[1, 3].text(0.1, 0.1, f"  Effect Size: {moe_hypothesis_results['statistics']['cohens_d']:.3f}", 
                   transform=axes[1, 3].transAxes)
    axes[1, 3].text(0.1, 0.0, f"  P-value: {moe_hypothesis_results['statistics']['t_pvalue']:.6f}", 
                   transform=axes[1, 3].transAxes)
    
    axes[1, 3].set_xlim(0, 1)
    axes[1, 3].set_ylim(0, 1)
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('stratified_manifold_hypothesis_test.png', dpi=300, bbox_inches='tight')
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
        'original_hypothesis_test': original_hypothesis_results,
        'moe_hypothesis_test': moe_hypothesis_results,
        'training_losses': [float(x) for x in training_losses],
        'expert_usage': np.mean(gate_probs, axis=0).tolist()
    }
    
    # Save results to JSON
    with open('stratified_manifold_hypothesis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Stratified manifold hypothesis test completed successfully!")
    print(f"   Results saved to 'stratified_manifold_hypothesis_results.json'")
    print(f"   Visualizations saved:")
    print(f"     - stratified_manifold_hypothesis_test.png")
    print(f"     - original_curvature_discontinuity.png")
    print(f"     - moe_curvature_discontinuity.png")
    
    return results

if __name__ == "__main__":
    # Run stratified manifold hypothesis test
    results = run_stratified_manifold_hypothesis_test(samples_per_domain=150, num_clusters=5, num_epochs=20)
