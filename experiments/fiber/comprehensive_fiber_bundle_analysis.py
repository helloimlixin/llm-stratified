"""
Comprehensive experiment integrating fiber bundle hypothesis test.
Combines stratified manifold analysis with token embedding irregularities.
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

class GeometricAwareMoE(nn.Module):
    """MoE with geometric awareness inspired by fiber bundle analysis."""
    def __init__(self, input_dim=768, hidden_dim=256, num_experts=8, top_k=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Expert networks with geometric regularization
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(num_experts)
        ])
        
        # Advanced gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts)
        )
        
        # Fiber bundle awareness layer
        self.fiber_bundle_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
        # Manifold smoothness layer
        self.smoothness_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output between 0 and 1
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
        
        # Geometric awareness
        fiber_bundle_score = self.fiber_bundle_layer(x)
        smoothness_score = self.smoothness_layer(x)
        
        return output, gate_probs, fiber_bundle_score, smoothness_score

def train_geometric_aware_moe(embeddings, labels, domains, num_epochs=20, lr=1e-3):
    """Train geometric-aware MoE model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Convert to tensors
    X = torch.FloatTensor(embeddings).to(device)
    y = torch.LongTensor(labels).to(device)
    
    # Initialize model
    model = GeometricAwareMoE(input_dim=embeddings.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    losses = []
    
    for epoch in tqdm(range(num_epochs), desc="Training Geometric-Aware MoE"):
        optimizer.zero_grad()
        
        # Forward pass
        output, gate_probs, fiber_bundle_score, smoothness_score = model(X)
        
        # Reconstruction loss
        reconstruction_loss = nn.MSELoss()(output, X)
        
        # Diversity loss
        diversity_loss = -torch.mean(torch.sum(gate_probs * torch.log(gate_probs + 1e-8), dim=-1))
        
        # Fiber bundle consistency loss (encourage smooth structure)
        fiber_bundle_loss = torch.mean(fiber_bundle_score**2)
        
        # Smoothness loss (encourage manifold smoothness)
        smoothness_loss = torch.mean((smoothness_score - 1.0)**2)  # Encourage high smoothness
        
        # Total loss
        total_loss = (reconstruction_loss + 
                     0.1 * diversity_loss + 
                     0.01 * fiber_bundle_loss + 
                     0.01 * smoothness_loss)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        losses.append(total_loss.item())
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")
    
    return model, losses

def run_comprehensive_fiber_bundle_analysis(samples_per_domain=200, num_clusters=6, num_epochs=15):
    """
    Run comprehensive analysis integrating fiber bundle hypothesis test.
    
    Combines stratified manifold analysis with token embedding irregularities.
    """
    print("🔬 Running Comprehensive Fiber Bundle Analysis")
    print("=" * 60)
    print("Integrating stratified manifold analysis with fiber bundle hypothesis test")
    print("Based on: Robinson, Dey, & Chiang (2025) - arXiv:2504.01002v2")
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
    
    # Train geometric-aware MoE model
    print(f"\nTraining geometric-aware MoE model...")
    moe_model, training_losses = train_geometric_aware_moe(all_embeddings, labels, domains, num_epochs=num_epochs)
    
    # Get MoE-enhanced embeddings
    print("Generating MoE-enhanced embeddings...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    moe_model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(all_embeddings).to(device)
        moe_embeddings, gate_probs, fiber_bundle_scores, smoothness_scores = moe_model(X_tensor)
        moe_embeddings = moe_embeddings.cpu().numpy()
        gate_probs = gate_probs.cpu().numpy()
        fiber_bundle_scores = fiber_bundle_scores.cpu().numpy()
        smoothness_scores = smoothness_scores.cpu().numpy()
    
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
    
    # FIBER BUNDLE HYPOTHESIS TEST
    print(f"\n{'='*60}")
    print("🔬 FIBER BUNDLE HYPOTHESIS TEST")
    print(f"{'='*60}")
    
    from src.geometric_tools.fiber_bundle_analysis import run_fiber_bundle_analysis
    
    # Test original embeddings
    print("Testing fiber bundle hypothesis on original embeddings...")
    original_fiber_results = run_fiber_bundle_analysis(
        emb_reduced, strata=strata, labels=labels, domains=domains, texts=texts,
        save_path='original_fiber_bundle_analysis.png'
    )
    
    # Test MoE-enhanced embeddings
    print("\nTesting fiber bundle hypothesis on MoE-enhanced embeddings...")
    moe_fiber_results = run_fiber_bundle_analysis(
        emb_moe_reduced, strata=strata_moe, labels=labels, domains=domains, texts=texts,
        save_path='moe_fiber_bundle_analysis.png'
    )
    
    # DEEP GEOMETRIC ANALYSIS
    print(f"\n{'='*60}")
    print("🔬 DEEP GEOMETRIC ANALYSIS")
    print(f"{'='*60}")
    
    from src.geometric_tools.deep_geometric_analysis import run_deep_geometric_analysis
    
    # Analyze original embeddings
    print("Analyzing original embeddings...")
    original_deep_analysis = run_deep_geometric_analysis(
        emb_reduced, strata, labels=labels, domains=domains,
        save_path='original_deep_geometric_analysis.png'
    )
    
    # Analyze MoE-enhanced embeddings
    print("\nAnalyzing MoE-enhanced embeddings...")
    moe_deep_analysis = run_deep_geometric_analysis(
        emb_moe_reduced, strata_moe, labels=labels, domains=domains,
        save_path='moe_deep_geometric_analysis.png'
    )
    
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
    
    # COMPREHENSIVE ANALYSIS
    print(f"\n{'='*60}")
    print("📊 COMPREHENSIVE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    # Compare fiber bundle results
    print(f"Fiber Bundle Hypothesis Test Comparison:")
    print(f"  Original Irregularity Rate: {original_fiber_results['irregularity_rate']:.3f}")
    print(f"  MoE Irregularity Rate: {moe_fiber_results['irregularity_rate']:.3f}")
    print(f"  Irregularity Change: {((moe_fiber_results['irregularity_rate'] / original_fiber_results['irregularity_rate'] - 1) * 100):+.1f}%")
    
    # Compare deep analysis results
    print(f"\nDeep Geometric Analysis Comparison:")
    print(f"  Original Scalar Curvature: {original_deep_analysis['scalar_curvature']:.4f}")
    print(f"  MoE Scalar Curvature: {moe_deep_analysis['scalar_curvature']:.4f}")
    print(f"  Original Mean Intrinsic Dimension: {np.mean(original_deep_analysis['intrinsic_dims']):.4f}")
    print(f"  MoE Mean Intrinsic Dimension: {np.mean(moe_deep_analysis['intrinsic_dims']):.4f}")
    
    # Compare hypothesis test results
    print(f"\nStratified Manifold Hypothesis Comparison:")
    print(f"  Original Hypothesis Supported: {original_hypothesis_results['hypothesis_supported']}")
    print(f"  MoE Hypothesis Supported: {moe_hypothesis_results['hypothesis_supported']}")
    print(f"  Original Effect Size: {original_hypothesis_results['statistics']['cohens_d']:.4f}")
    print(f"  MoE Effect Size: {moe_hypothesis_results['statistics']['cohens_d']:.4f}")
    
    # Geometric awareness analysis
    print(f"\nGeometric Awareness Analysis:")
    print(f"  Mean Fiber Bundle Score: {np.mean(fiber_bundle_scores):.4f}")
    print(f"  Mean Smoothness Score: {np.mean(smoothness_scores):.4f}")
    print(f"  Fiber Bundle Score Std: {np.std(fiber_bundle_scores):.4f}")
    print(f"  Smoothness Score Std: {np.std(smoothness_scores):.4f}")
    
    # Expert analysis
    print(f"\nExpert Analysis:")
    expert_usage = np.mean(gate_probs, axis=0)
    print(f"  Most Used Expert: {np.argmax(expert_usage)} (usage: {np.max(expert_usage):.4f})")
    print(f"  Least Used Expert: {np.argmin(expert_usage)} (usage: {np.min(expert_usage):.4f})")
    print(f"  Expert Usage Variance: {np.var(expert_usage):.4f}")
    
    # Create comprehensive comparison visualization
    print("\nCreating comprehensive comparison visualization...")
    
    # PCA 2D visualization
    pca_2d = PCA(n_components=2)
    emb_2d = pca_2d.fit_transform(emb_reduced)
    emb_moe_2d = pca_2d.fit_transform(emb_moe_reduced)
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Original embeddings
    scatter1 = axes[0, 0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=strata, cmap='tab10', alpha=0.7, s=30)
    axes[0, 0].set_title("Original: Colored by Strata")
    axes[0, 0].set_xlabel("PC1")
    axes[0, 0].set_ylabel("PC2")
    plt.colorbar(scatter1, ax=axes[0, 0], label="Stratum")
    
    # Fiber bundle irregularities
    colors = ['red' if irr else 'blue' for irr in original_fiber_results['irregularities']]
    scatter2 = axes[0, 1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, alpha=0.7, s=30)
    axes[0, 1].set_title("Original: Fiber Bundle Irregularities")
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Irregular'),
                      Patch(facecolor='blue', label='Regular')]
    axes[0, 1].legend(handles=legend_elements)
    
    scatter3 = axes[0, 2].scatter(emb_2d[:, 0], emb_2d[:, 1], c=original_deep_analysis['ricci_curvatures'], cmap='plasma', alpha=0.7, s=30)
    axes[0, 2].set_title("Original: Ricci Curvature")
    axes[0, 2].set_xlabel("PC1")
    axes[0, 2].set_ylabel("PC2")
    plt.colorbar(scatter3, ax=axes[0, 2], label="Ricci Curvature")
    
    scatter4 = axes[0, 3].scatter(emb_2d[:, 0], emb_2d[:, 1], c=original_deep_analysis['intrinsic_dims'], cmap='viridis', alpha=0.7, s=30)
    axes[0, 3].set_title("Original: Intrinsic Dimension")
    axes[0, 3].set_xlabel("PC1")
    axes[0, 3].set_ylabel("PC2")
    plt.colorbar(scatter4, ax=axes[0, 3], label="Intrinsic Dimension")
    
    # MoE-enhanced embeddings
    scatter5 = axes[1, 0].scatter(emb_moe_2d[:, 0], emb_moe_2d[:, 1], c=strata_moe, cmap='tab10', alpha=0.7, s=30)
    axes[1, 0].set_title("MoE-Enhanced: Colored by Strata")
    axes[1, 0].set_xlabel("PC1")
    axes[1, 0].set_ylabel("PC2")
    plt.colorbar(scatter5, ax=axes[1, 0], label="Stratum")
    
    # MoE fiber bundle irregularities
    colors_moe = ['red' if irr else 'blue' for irr in moe_fiber_results['irregularities']]
    scatter6 = axes[1, 1].scatter(emb_moe_2d[:, 0], emb_moe_2d[:, 1], c=colors_moe, alpha=0.7, s=30)
    axes[1, 1].set_title("MoE-Enhanced: Fiber Bundle Irregularities")
    axes[1, 1].set_xlabel("PC1")
    axes[1, 1].set_ylabel("PC2")
    axes[1, 1].legend(handles=legend_elements)
    
    scatter7 = axes[1, 2].scatter(emb_moe_2d[:, 0], emb_moe_2d[:, 1], c=moe_deep_analysis['ricci_curvatures'], cmap='plasma', alpha=0.7, s=30)
    axes[1, 2].set_title("MoE-Enhanced: Ricci Curvature")
    axes[1, 2].set_xlabel("PC1")
    axes[1, 2].set_ylabel("PC2")
    plt.colorbar(scatter7, ax=axes[1, 2], label="Ricci Curvature")
    
    scatter8 = axes[1, 3].scatter(emb_moe_2d[:, 0], emb_moe_2d[:, 1], c=moe_deep_analysis['intrinsic_dims'], cmap='viridis', alpha=0.7, s=30)
    axes[1, 3].set_title("MoE-Enhanced: Intrinsic Dimension")
    axes[1, 3].set_xlabel("PC1")
    axes[1, 3].set_ylabel("PC2")
    plt.colorbar(scatter8, ax=axes[1, 3], label="Intrinsic Dimension")
    
    # Comparison plots
    # Irregularity rate comparison
    models = ['Original', 'MoE-Enhanced']
    irregularity_rates = [original_fiber_results['irregularity_rate'], moe_fiber_results['irregularity_rate']]
    
    bars1 = axes[2, 0].bar(models, irregularity_rates, color=['blue', 'orange'], alpha=0.7)
    axes[2, 0].set_title("Fiber Bundle Irregularity Rate")
    axes[2, 0].set_ylabel("Irregularity Rate")
    axes[2, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, rate in zip(bars1, irregularity_rates):
        axes[2, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f"{rate:.3f}", ha='center', va='bottom')
    
    # Geometric awareness scores
    axes[2, 1].scatter(fiber_bundle_scores.flatten(), smoothness_scores.flatten(), alpha=0.6, s=30)
    axes[2, 1].set_title("Geometric Awareness Scores")
    axes[2, 1].set_xlabel("Fiber Bundle Score")
    axes[2, 1].set_ylabel("Smoothness Score")
    axes[2, 1].grid(True, alpha=0.3)
    
    # Expert usage
    axes[2, 2].bar(range(len(expert_usage)), expert_usage, alpha=0.7)
    axes[2, 2].set_title("Expert Usage Distribution")
    axes[2, 2].set_xlabel("Expert Index")
    axes[2, 2].set_ylabel("Usage Probability")
    axes[2, 2].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[2, 3].text(0.1, 0.9, f"Comprehensive Analysis Summary", 
                    fontweight='bold', transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.8, f"Original Irregularity Rate: {original_fiber_results['irregularity_rate']:.3f}", 
                    transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.7, f"MoE Irregularity Rate: {moe_fiber_results['irregularity_rate']:.3f}", 
                    transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.6, f"Original Hypothesis: {'✓' if original_hypothesis_results['hypothesis_supported'] else '✗'}", 
                    transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.5, f"MoE Hypothesis: {'✓' if moe_hypothesis_results['hypothesis_supported'] else '✗'}", 
                    transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.4, f"Mean Fiber Bundle Score: {np.mean(fiber_bundle_scores):.3f}", 
                    transform=axes[2, 3].transAxes)
    axes[2, 3].text(0.1, 0.3, f"Mean Smoothness Score: {np.mean(smoothness_scores):.3f}", 
                    transform=axes[2, 3].transAxes)
    
    axes[2, 3].set_xlim(0, 1)
    axes[2, 3].set_ylim(0, 1)
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('comprehensive_fiber_bundle_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Training loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses)
    plt.title("Geometric-Aware MoE Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig('geometric_aware_moe_training_loss.png', dpi=300, bbox_inches='tight')
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
        'original_fiber_bundle_results': original_fiber_results,
        'moe_fiber_bundle_results': moe_fiber_results,
        'original_deep_analysis': {
            'scalar_curvature': float(original_deep_analysis['scalar_curvature']),
            'mean_ricci_curvature': float(np.mean(original_deep_analysis['ricci_curvatures'])),
            'mean_intrinsic_dimension': float(np.mean(original_deep_analysis['intrinsic_dims'])),
            'stratum_analysis': {int(k): v for k, v in original_deep_analysis['stratum_analysis'].items()}
        },
        'moe_deep_analysis': {
            'scalar_curvature': float(moe_deep_analysis['scalar_curvature']),
            'mean_ricci_curvature': float(np.mean(moe_deep_analysis['ricci_curvatures'])),
            'mean_intrinsic_dimension': float(np.mean(moe_deep_analysis['intrinsic_dims'])),
            'stratum_analysis': {int(k): v for k, v in moe_deep_analysis['stratum_analysis'].items()}
        },
        'original_hypothesis_test': original_hypothesis_results,
        'moe_hypothesis_test': moe_hypothesis_results,
        'training_losses': [float(x) for x in training_losses],
        'expert_usage': expert_usage.tolist(),
        'fiber_bundle_scores': fiber_bundle_scores.flatten().tolist(),
        'smoothness_scores': smoothness_scores.flatten().tolist()
    }
    
    # Save results to JSON
    with open('comprehensive_fiber_bundle_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Comprehensive fiber bundle analysis completed successfully!")
    print(f"   Results saved to 'comprehensive_fiber_bundle_analysis_results.json'")
    print(f"   Visualizations saved:")
    print(f"     - comprehensive_fiber_bundle_analysis.png")
    print(f"     - original_fiber_bundle_analysis.png")
    print(f"     - moe_fiber_bundle_analysis.png")
    print(f"     - original_deep_geometric_analysis.png")
    print(f"     - moe_deep_geometric_analysis.png")
    print(f"     - original_curvature_discontinuity.png")
    print(f"     - moe_curvature_discontinuity.png")
    print(f"     - geometric_aware_moe_training_loss.png")
    
    return results

if __name__ == "__main__":
    # Run comprehensive fiber bundle analysis
    results = run_comprehensive_fiber_bundle_analysis(samples_per_domain=150, num_clusters=5, num_epochs=20)
