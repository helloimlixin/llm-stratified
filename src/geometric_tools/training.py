"""
Training utilities for stratified manifold learning experiments.

This module contains loss functions and training procedures for the
Mixture of Dictionary Experts models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


def contrastive_loss_with_labels(embeddings, labels, margin=1.0):
    """
    Contrastive loss function using actual labels.
    
    This loss encourages embeddings of samples with the same label to be close
    and embeddings of samples with different labels to be far apart.
    
    Args:
        embeddings: Tensor of embeddings (batch_size, embedding_dim)
        labels: Tensor of labels (batch_size,)
        margin: Margin for negative pairs
        
    Returns:
        Contrastive loss value
    """
    batch_size = embeddings.size(0)
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    labels = labels.unsqueeze(1)
    sim_matrix = (labels == labels.T).float()
    
    # Positive loss: minimize distance for same labels
    pos_loss = sim_matrix * (dist_matrix ** 2)
    
    # Negative loss: maximize distance for different labels (with margin)
    neg_loss = (1 - sim_matrix) * torch.clamp(margin - dist_matrix, min=0.0) ** 2
    
    return (pos_loss + neg_loss).mean()


def train_contrastive_moe_with_labels(model, data_loader, num_epochs=10, lr=1e-3, 
                                    margin=1.0, device="cpu", verbose=True):
    """
    Train Mixture of Dictionary Experts model with contrastive loss.
    
    Args:
        model: MixtureOfDictionaryExperts model to train
        data_loader: DataLoader for training data
        num_epochs: Number of training epochs
        lr: Learning rate
        margin: Margin for contrastive loss
        device: Device to train on
        verbose: Whether to print training progress
        
    Returns:
        Trained model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_samples = 0
        
        for batch in data_loader:
            x, batch_labels = batch[0].to(device), batch[1].to(device)
            
            # Forward pass
            z = model(x)
            loss = contrastive_loss_with_labels(z, batch_labels, margin)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
        
        avg_loss = total_loss / total_samples
        
        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Contrastive Loss: {avg_loss:.9f}")
    
    return model


def evaluate_model_performance(model, data_loader, device="cpu"):
    """
    Evaluate model performance on a dataset.
    
    Args:
        model: Trained model
        data_loader: DataLoader for evaluation data
        device: Device to evaluate on
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_embeddings = []
    all_labels = []
    all_gating_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            x, batch_labels = batch[0].to(device), batch[1].to(device)
            
            # Forward pass
            z = model(x)
            loss = contrastive_loss_with_labels(z, batch_labels)
            
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            
            # Store results
            all_embeddings.append(z.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())
            all_gating_probs.append(model.get_gating_probabilities(x))
    
    # Concatenate results
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_gating_probs = np.concatenate(all_gating_probs, axis=0)
    
    avg_loss = total_loss / total_samples
    
    return {
        'avg_loss': avg_loss,
        'embeddings': all_embeddings,
        'labels': all_labels,
        'gating_probs': all_gating_probs,
        'total_samples': total_samples
    }


def compute_clustering_metrics(embeddings, labels):
    """
    Compute clustering quality metrics.
    
    Args:
        embeddings: Embedding matrix
        labels: Cluster labels
        
    Returns:
        Dictionary containing clustering metrics
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
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
    from sklearn.decomposition import PCA
    
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
    import pandas as pd
    
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
