#!/usr/bin/env python3
"""
Advanced Fiber Bundle Analysis with LLaMA-3.2-1B and Stratified Manifold Learning

This script combines the fiber bundle hypothesis test with advanced stratified manifold
learning using mixture-of-experts and dictionary learning, applied to LLaMA-3.2-1B embeddings.
"""

import sys
import os
from pathlib import Path
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
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import umap
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fiber_bundle_test.core import FiberBundleTest
from fiber_bundle_test.visualization import ResultsVisualizer
from fiber_bundle_test.utils import DataUtils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"

#############################################################################
# 1. LLaMA-3.2-1B Model for Sentence Embeddings
#############################################################################
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B"

def setup_llama_model():
    """Setup LLaMA-3.2-1B model for embedding extraction."""
    try:
        llama3_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME, trust_remote_code=True)
        llama3_model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_NAME, 
            device_map="auto", 
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        llama3_model.eval()
        
        # Set pad_token if not already set
        if llama3_tokenizer.pad_token is None:
            llama3_tokenizer.pad_token = llama3_tokenizer.eos_token
        
        return llama3_tokenizer, llama3_model
        
    except Exception as e:
        logger.error(f"Failed to load LLaMA-3.2-1B: {e}")
        logger.info("Make sure you have access to the model and are logged in to HuggingFace")
        raise

def llama3_embed_texts(texts, tokenizer, model, batch_size=25):
    """
    Generates sentence embeddings using LLaMA-3.2-1B model in batches.
    """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting LLaMA embeddings"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Use last hidden state and average over sequence length
        hidden_states = outputs.hidden_states[-1]  # shape: (batch, seq_len, hidden_dim)
        batch_embeddings = hidden_states.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

#############################################################################
# 2. Data Loading & Processing (Multi-domain Text Datasets)
#############################################################################
def unify_dataset(ds, domain_name, samples_per_domain=100, text_field="text"):
    """Unify dataset format across different sources."""
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
    Loads and unifies six diverse text datasets for fiber bundle analysis:
      - IMDB (long reviews)
      - Rotten Tomatoes (moderate-length reviews)
      - Amazon Polarity (complex reviews)
      - GLUE/SST2 (simpler, shorter sentences)
      - TweetEval (tweets)
      - AG News (news articles)
    """
    datasets_list = []
    
    try:
        # IMDB dataset
        imdb_ds = unify_dataset(
            load_dataset("imdb", split=f"train[:{samples_per_domain}]"), 
            "imdb", samples_per_domain
        )
        datasets_list.append(imdb_ds)
        logger.info(f"Loaded IMDB dataset: {len(imdb_ds)} samples")
    except Exception as e:
        logger.warning(f"Failed to load IMDB: {e}")
    
    try:
        # Rotten Tomatoes
        rt_ds = unify_dataset(
            load_dataset("rotten_tomatoes", split=f"train[:{samples_per_domain}]"), 
            "rotten", samples_per_domain
        )
        datasets_list.append(rt_ds)
        logger.info(f"Loaded Rotten Tomatoes dataset: {len(rt_ds)} samples")
    except Exception as e:
        logger.warning(f"Failed to load Rotten Tomatoes: {e}")
    
    try:
        # Amazon Polarity
        ap_raw = load_dataset("amazon_polarity", split=f"train[:{int(2 * samples_per_domain)}]")
        ap_raw = ap_raw.map(lambda x: {"text": f"{x['title']} {x['content']}".strip()})
        ap_ds = unify_dataset(ap_raw, "amazon", samples_per_domain)
        datasets_list.append(ap_ds)
        logger.info(f"Loaded Amazon Polarity dataset: {len(ap_ds)} samples")
    except Exception as e:
        logger.warning(f"Failed to load Amazon Polarity: {e}")
    
    try:
        # SST2
        sst2_ds = load_dataset("glue", "sst2", split=f"train[:{samples_per_domain}]")
        sst2_ds = unify_dataset(sst2_ds, "sst2", samples_per_domain, text_field="sentence")
        datasets_list.append(sst2_ds)
        logger.info(f"Loaded SST2 dataset: {len(sst2_ds)} samples")
    except Exception as e:
        logger.warning(f"Failed to load SST2: {e}")
    
    # If no datasets loaded, create sample data
    if not datasets_list:
        logger.warning("No datasets loaded successfully, creating sample data")
        sample_texts = [
            "This movie was absolutely fantastic and I loved every moment of it.",
            "The film was terrible and completely boring throughout.",
            "Great product, highly recommend to everyone looking for quality.",
            "Poor quality item, would not buy again from this seller.",
            "The news report was very informative and well researched.",
            "Breaking news story lacks important details and context."
        ]
        sample_labels = [1, 0, 1, 0, 1, 0]
        sample_domains = ["sample"] * len(sample_texts)
        
        # Expand sample data
        expanded_texts = sample_texts * (samples_per_domain // len(sample_texts) + 1)
        expanded_labels = sample_labels * (samples_per_domain // len(sample_labels) + 1)
        expanded_domains = sample_domains * (samples_per_domain // len(sample_domains) + 1)
        
        from datasets import Dataset
        sample_ds = Dataset.from_dict({
            "text": expanded_texts[:samples_per_domain],
            "label": expanded_labels[:samples_per_domain],
            "domain": expanded_domains[:samples_per_domain]
        })
        datasets_list.append(sample_ds)
    
    return concatenate_datasets(datasets_list)

#############################################################################
# 3. Top-K with Straight-Through Estimator
#############################################################################
class TopKSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        values, indices = torch.topk(torch.abs(x), k, dim=1)
        mask = torch.zeros_like(x)
        mask.scatter_(1, indices, 1.0)
        return x * mask
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def topk_st(x, k):
    return TopKSTE.apply(x, k)

#############################################################################
# 4. LISTA-based Dictionary Experts with Varying Sparsity Levels
#############################################################################
class LISTALayer(nn.Module):
    def __init__(self, input_dim, code_dim, sparsity_level):
        super().__init__()
        self.W = nn.Linear(input_dim, code_dim, bias=False)
        self.S = nn.Linear(code_dim, code_dim, bias=False)
        self.sparsity_level = sparsity_level
    
    def forward(self, x, z_prev):
        update = self.W(x) + self.S(z_prev)
        new_z = topk_st(update, self.sparsity_level)
        return new_z

class DictionaryExpertLISTA(nn.Module):
    def __init__(self, input_dim, code_dim, num_layers=5, sparsity_level=5):
        super().__init__()
        self.sparsity_level = sparsity_level
        self.dictionary = nn.Parameter(torch.randn(input_dim, code_dim) * 0.01)
        self.num_layers = num_layers
        self.lista_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            layer = LISTALayer(input_dim, code_dim, sparsity_level)
            self.lista_layers.append(layer)
    
    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        z = torch.zeros(batch_size, self.dictionary.shape[1], device=device)
        
        for layer in self.lista_layers:
            z = layer(x, z)
        
        recon = torch.matmul(z, self.dictionary.T)
        return recon, z

#############################################################################
# 5. Gating Network & Mixture-of-Experts
#############################################################################
class GatingNetworkAttention(nn.Module):
    def __init__(self, input_dim, query_dim, K):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, query_dim)
        self.keys = nn.Parameter(torch.randn(K, query_dim))
    
    def forward(self, x):
        query = self.query_proj(x)
        logits = torch.matmul(query, self.keys.T) / (query.shape[-1] ** 0.5)
        probs = torch.softmax(logits, dim=1)
        return probs

class MixtureOfDictionaryExperts(nn.Module):
    def __init__(self, input_dim, query_dim, code_dim, K, projection_dim=64, 
                 num_lista_layers=5, sparsity_levels=None, threshold=0.9):
        super().__init__()
        self.K = K
        self.threshold = threshold
        self.gating_net = GatingNetworkAttention(input_dim, query_dim, K)
        
        if sparsity_levels is None:
            sparsity_levels = list(map(int, np.linspace(5, code_dim, K)))
        
        self.experts = nn.ModuleList([
            DictionaryExpertLISTA(input_dim, code_dim, 
                                num_layers=num_lista_layers, 
                                sparsity_level=sparsity_levels[i])
            for i in range(K)
        ])
        
        self.projection_head = nn.Sequential(
            nn.Linear(code_dim, code_dim),
            nn.ReLU(),
            nn.Linear(code_dim, projection_dim)
        )
        
        self.register_buffer("sparsity_levels", torch.tensor(sparsity_levels, dtype=torch.float32))
    
    def forward(self, x):
        gating_probs = self.gating_net(x)
        batch_size = gating_probs.size(0)
        
        # Get outputs from all experts
        zs = []
        for expert in self.experts:
            _, z = expert(x)
            zs.append(z)
        zs = torch.stack(zs, dim=0)  # (K, batch_size, code_dim)
        
        # Hard expert assignment based on gating probabilities
        selected_z = torch.empty(batch_size, zs.size(2), device=x.device)
        for i in range(batch_size):
            p = gating_probs[i]
            p_max = torch.max(p)
            eligible = (p >= self.threshold * p_max).nonzero(as_tuple=True)[0]
            
            if eligible.numel() == 0:
                idx = torch.argmax(p)
            else:
                eligible_sparsity = self.sparsity_levels[eligible]
                min_idx = torch.argmin(eligible_sparsity)
                idx = eligible[min_idx]
            
            selected_z[i] = zs[idx, i, :]
        
        projection = self.projection_head(selected_z)
        return projection

#############################################################################
# 6. Contrastive Loss with Labels
#############################################################################
def contrastive_loss_with_labels(embeddings, labels, margin=1.0):
    """Contrastive loss using actual labels."""
    batch_size = embeddings.size(0)
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    
    labels = labels.unsqueeze(1)
    sim_matrix = (labels == labels.T).float()
    
    pos_loss = sim_matrix * (dist_matrix ** 2)
    neg_loss = (1 - sim_matrix) * torch.clamp(margin - dist_matrix, min=0.0) ** 2
    
    return (pos_loss + neg_loss).mean()

#############################################################################
# 7. Training Function
#############################################################################
def train_contrastive_moe_with_labels(model, data_loader, num_epochs=100, lr=1e-3, margin=1.0, device="cpu"):
    """Train the mixture-of-experts model with contrastive loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_samples = 0
        
        for batch in data_loader:
            x, batch_labels = batch[0].to(device), batch[1].to(device)
            z = model(x)
            loss = contrastive_loss_with_labels(z, batch_labels, margin)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
        
        avg_loss = total_loss / total_samples
        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Contrastive Loss: {avg_loss:.6f}")
    
    return model

#############################################################################
# 8. Fiber Bundle Analysis Integration
#############################################################################
def run_fiber_bundle_analysis(embeddings, domains, output_dir):
    """Run fiber bundle hypothesis test on the embeddings."""
    logger.info("Running fiber bundle hypothesis test...")
    
    # Create target tokens from domains
    unique_domains = list(set(domains))
    target_tokens = []
    sentences = []
    
    # Create synthetic sentences for each domain
    domain_templates = {
        'imdb': 'This movie review from IMDB discusses the film quality.',
        'rotten': 'This Rotten Tomatoes review evaluates the movie performance.',
        'amazon': 'This Amazon product review describes the item quality.',
        'sst2': 'This Stanford sentiment analysis sentence expresses an opinion.',
        'tweet': 'This tweet contains social media sentiment.',
        'ag_news': 'This news article reports on current events.',
        'sample': 'This sample text demonstrates sentiment analysis.'
    }
    
    for i, domain in enumerate(domains):
        template = domain_templates.get(domain, f'This {domain} text contains sentiment.')
        sentences.append(template)
        target_tokens.append(domain)
    
    # Run fiber bundle test
    test = FiberBundleTest(
        r_min=0.01,
        r_max=30.0,
        n_r=250,
        alpha=0.01,
        window_size=15
    )
    
    results = test.run_test(embeddings, verbose=False)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    DataUtils.save_results(results, output_path / 'fiber_bundle_results.json')
    
    # Create visualizations
    visualizer = ResultsVisualizer()
    
    summary_fig = visualizer.plot_results_summary(results, target_tokens)
    dimension_fig = visualizer.plot_dimension_analysis(results, target_tokens)
    
    plots_dir = output_path / 'plots'
    visualizer.save_plots(
        [summary_fig, dimension_fig],
        ['llama_fiber_summary', 'llama_fiber_dimensions'],
        str(plots_dir)
    )
    
    logger.info(f"Fiber bundle analysis results saved to {output_path}")
    
    return results

#############################################################################
# 9. Main Execution Function
#############################################################################
def main():
    """Main execution function combining all analyses."""
    print("🦙 Advanced LLaMA-3.2-1B Fiber Bundle Analysis")
    print("=" * 60)
    print(f"Using device: {device}")
    
    try:
        # 1. Setup LLaMA model
        print("\n1. Setting up LLaMA-3.2-1B model...")
        llama3_tokenizer, llama3_model = setup_llama_model()
        print("✅ LLaMA-3.2-1B model loaded successfully")
        
        # 2. Load multi-domain datasets
        print("\n2. Loading multi-domain sentiment datasets...")
        combined_ds = load_multidomain_sentiment(samples_per_domain=200)
        texts = combined_ds["text"]
        domains = combined_ds["domain"]
        labels = torch.tensor(combined_ds["label"], dtype=torch.long)
        
        print(f"✅ Loaded {len(texts)} samples from {len(set(domains))} domains")
        print(f"Domains: {list(set(domains))}")
        
        # 3. Generate LLaMA embeddings
        print("\n3. Generating LLaMA-3.2-1B embeddings...")
        all_embeddings = llama3_embed_texts(texts, llama3_tokenizer, llama3_model, batch_size=8)
        print(f"✅ Generated embeddings shape: {all_embeddings.shape}")
        
        # 4. Prepare embeddings for analysis
        print("\n4. Preparing embeddings for analysis...")
        
        # Pad to consistent dimension if needed
        target_dim = 2048
        if all_embeddings.shape[1] < target_dim:
            padding = np.zeros((all_embeddings.shape[0], target_dim - all_embeddings.shape[1]))
            padded_embeddings = np.concatenate([all_embeddings, padding], axis=1)
        else:
            padded_embeddings = all_embeddings[:, :target_dim]
        
        print(f"✅ Padded embeddings shape: {padded_embeddings.shape}")
        
        # Reduce dimensionality using PCA
        pca = PCA(n_components=64)
        emb_64d = pca.fit_transform(padded_embeddings)
        print(f"✅ PCA reduced shape: {emb_64d.shape}")
        
        # 5. Run Fiber Bundle Analysis
        print("\n5. Running Fiber Bundle Hypothesis Test...")
        output_dir = "./advanced_llama_analysis_output"
        fiber_results = run_fiber_bundle_analysis(emb_64d, domains, output_dir)
        
        print(f"\n📊 Fiber Bundle Results:")
        print(f"  Total tokens analyzed: {fiber_results['total_tokens']}")
        print(f"  Hypothesis rejected: {fiber_results['total_rejections']}")
        print(f"  Rejection rate: {fiber_results['rejection_rate']:.2%}")
        
        # 6. Stratified Analysis with MoE
        print("\n6. Training Mixture-of-Experts for Stratified Analysis...")
        
        X_tensor = torch.tensor(emb_64d, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, labels)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Initialize MoE model
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
        
        # Train the model
        trained_model = train_contrastive_moe_with_labels(
            model, loader, num_epochs=50, lr=1e-3, margin=1.0, device=device
        )
        
        print("✅ MoE training completed")
        
        # 7. Stratification Analysis
        print("\n7. Analyzing stratification...")
        
        # K-means clustering for stratification
        kmeans = KMeans(n_clusters=5, random_state=42)
        strata = kmeans.fit_predict(emb_64d)
        
        # Compute intrinsic dimensions
        print("\nIntrinsic Dimensions per Stratum (75% variance):")
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
        
        # Clustering metrics
        silhouette = silhouette_score(emb_64d, strata)
        db_index = davies_bouldin_score(emb_64d, strata)
        ch_index = calinski_harabasz_score(emb_64d, strata)
        
        print(f"\nStratum Separation Metrics:")
        print(f"  Silhouette Score: {silhouette:.4f} (higher is better)")
        print(f"  Davies-Bouldin Index: {db_index:.4f} (lower is better)")
        print(f"  Calinski-Harabasz Index: {ch_index:.4f} (higher is better)")
        
        # 8. Visualizations
        print("\n8. Creating visualizations...")
        
        # Get gating probabilities
        with torch.no_grad():
            gating_probs_all = trained_model.gating_net(X_tensor.to(device)).cpu().numpy()
        
        # Create dataframes for analysis
        df_gating = pd.DataFrame(gating_probs_all, columns=[f"Expert_{i}" for i in range(model.K)])
        df_gating["Stratum"] = strata
        df_gating["Domain"] = domains
        
        # Average gating probabilities
        expert_cols = [f"Expert_{i}" for i in range(model.K)]
        avg_gating_per_stratum = df_gating.groupby("Stratum")[expert_cols].mean()
        avg_gating_per_domain = df_gating.groupby("Domain")[expert_cols].mean()
        
        # Create heatmaps
        fig_heatmaps = sp.make_subplots(
            rows=1, cols=2,
            subplot_titles=["Gating Probabilities per Stratum", "Gating Probabilities per Domain"],
            column_widths=[0.5, 0.5]
        )
        
        # 3D visualization
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
        
        fig_3d = px.scatter_3d(
            df_plot,
            x="x", y="y", z="z",
            color="domain_label",
            hover_data=["domain", "label", "Stratum"],
            title="3D Visualization of LLaMA-3.2-1B Embeddings with Fiber Bundle Analysis",
            width=1000,
            height=700
        )
        
        # UMAP visualization
        umap_3d = umap.UMAP(n_components=3, random_state=42).fit_transform(emb_64d)
        df_umap = pd.DataFrame(umap_3d, columns=["component_0", "component_1", "component_2"])
        df_umap["Domain"] = domains
        df_umap["Stratum"] = strata
        
        fig_umap_3d = px.scatter_3d(
            df_umap,
            x="component_0", y="component_1", z="component_2",
            color="Domain",
            symbol="Stratum",
            title="UMAP 3D Visualization of LLaMA-3.2-1B Embeddings",
            hover_data=["Domain", "Stratum"]
        )
        
        # Save visualizations
        output_path = Path(output_dir)
        fig_3d.write_html(output_path / "llama_3d_visualization.html")
        fig_umap_3d.write_html(output_path / "llama_umap_3d.html")
        
        print("✅ Visualizations saved")
        
        # 9. Final Summary
        print(f"\n🎉 Analysis Complete!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"\n📊 Summary:")
        print(f"  • Analyzed {len(texts)} texts from {len(set(domains))} domains")
        print(f"  • LLaMA-3.2-1B embedding dimension: {all_embeddings.shape[1]}")
        print(f"  • Fiber bundle rejection rate: {fiber_results['rejection_rate']:.1%}")
        print(f"  • Identified {len(np.unique(strata))} distinct strata")
        print(f"  • Silhouette score: {silhouette:.3f}")
        
        return {
            'fiber_results': fiber_results,
            'embeddings': emb_64d,
            'strata': strata,
            'domains': domains,
            'intrinsic_dims': intrinsic_dims,
            'clustering_metrics': {
                'silhouette': silhouette,
                'davies_bouldin': db_index,
                'calinski_harabasz': ch_index
            }
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\n❌ Analysis failed: {e}")
        print("\nTroubleshooting:")
        print("• Ensure you have access to LLaMA-3.2-1B model")
        print("• Login to HuggingFace: huggingface-cli login")
        print("• Install required dependencies: pip install -r requirements_extended.txt")
        raise

if __name__ == "__main__":
    main()
