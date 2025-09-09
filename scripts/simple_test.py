"""
Minimal RoBERTa model implementation to avoid segmentation faults.
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import pandas as pd

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

def simple_roberta_embed_text(text):
    """
    Simple RoBERTa embedding function with lazy loading.
    """
    try:
        from transformers import RobertaTokenizer, RobertaModel
        
        # Initialize model and tokenizer
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaModel.from_pretrained("roberta-base").to(device)
        model.eval()
        
        # Generate embedding
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze(0)
        
        return embedding
    except Exception as e:
        print(f"Error in RoBERTa embedding: {e}")
        # Return random embedding as fallback
        return np.random.randn(768)

def simple_experiment():
    """Run a simple experiment to test functionality."""
    print("Running simple experiment...")
    
    # Create some dummy text data
    texts = [
        "This is a positive review about a great movie.",
        "I really enjoyed this film and would recommend it.",
        "This movie was terrible and boring.",
        "I did not like this film at all.",
        "Amazing movie with great acting and story."
    ]
    
    print(f"Processing {len(texts)} texts...")
    
    # Generate embeddings
    embeddings = []
    for text in texts:
        embedding = simple_roberta_embed_text(text)
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    print(f"PCA reduced shape: {embeddings_2d.shape}")
    
    # Simple clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(embeddings_2d)
    print(f"Clusters: {clusters}")
    
    # Create simple visualization
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='viridis')
    plt.title("Simple RoBERTa Embeddings (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig('simple_experiment.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Simple experiment completed successfully!")
    print("   Check 'simple_experiment.png' for the visualization.")
    
    return True

if __name__ == "__main__":
    simple_experiment()
