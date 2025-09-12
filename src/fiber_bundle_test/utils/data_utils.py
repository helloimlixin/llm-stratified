"""Data utility functions."""

import numpy as np
import json
import pickle
from typing import List, Tuple, Dict, Any
from pathlib import Path


class DataUtils:
    """Utility functions for data handling and processing."""
    
    @staticmethod
    def save_embeddings(embeddings: np.ndarray, 
                       filepath: str,
                       metadata: Dict[str, Any] = None):
        """
        Save embeddings to file with optional metadata.
        
        Args:
            embeddings: Numpy array of embeddings
            filepath: Path to save file
            metadata: Optional metadata dictionary
        """
        data = {
            'embeddings': embeddings,
            'metadata': metadata or {}
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.suffix == '.npz':
            np.savez_compressed(filepath, **data)
        elif filepath.suffix == '.pkl':
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError("Unsupported file format. Use .npz or .pkl")
    
    @staticmethod
    def load_embeddings(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load embeddings from file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            Tuple of (embeddings, metadata)
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.npz':
            data = np.load(filepath, allow_pickle=True)
            embeddings = data['embeddings']
            metadata = data.get('metadata', {})
            if hasattr(metadata, 'item'):  # Handle numpy scalar
                metadata = metadata.item()
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            embeddings = data['embeddings']
            metadata = data.get('metadata', {})
        else:
            raise ValueError("Unsupported file format. Use .npz or .pkl")
        
        return embeddings, metadata
    
    @staticmethod
    def save_results(results: Dict[str, Any], filepath: str):
        """
        Save test results to JSON file.
        
        Args:
            results: Results dictionary from FiberBundleTest
            filepath: Path to save file
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = DataUtils._make_json_serializable(results)
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    @staticmethod
    def load_results(filepath: str) -> Dict[str, Any]:
        """
        Load test results from JSON file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            Results dictionary
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        return results
    
    @staticmethod
    def _make_json_serializable(obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            # Convert keys to strings if they're numpy integers
            new_dict = {}
            for key, value in obj.items():
                if isinstance(key, (np.integer, np.int64, np.int32)):
                    new_key = str(int(key))
                else:
                    new_key = key
                new_dict[new_key] = DataUtils._make_json_serializable(value)
            return new_dict
        elif isinstance(obj, (list, tuple)):
            return [DataUtils._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    @staticmethod
    def create_sample_random_embeddings(n_tokens: int = 100, 
                                      embedding_dim: int = 512,
                                      seed: int = 42) -> np.ndarray:
        """
        Create sample random embeddings for testing.
        
        Args:
            n_tokens: Number of token embeddings
            embedding_dim: Dimension of each embedding
            seed: Random seed for reproducibility
            
        Returns:
            Array of random embeddings
        """
        np.random.seed(seed)
        return np.random.randn(n_tokens, embedding_dim)
    
    @staticmethod
    def extract_token_labels(sentences: List[str], 
                           target_tokens: List[str]) -> List[str]:
        """
        Create descriptive labels for tokens based on sentences.
        
        Args:
            sentences: List of sentences
            target_tokens: List of target tokens
            
        Returns:
            List of descriptive labels
        """
        labels = []
        for sentence, token in zip(sentences, target_tokens):
            # Extract a few context words
            words = sentence.split()
            try:
                token_idx = words.index(token)
                context = words[token_idx + 1] if token_idx + 1 < len(words) else sentence[:20]
                labels.append(f"{token} ({context})")
            except ValueError:
                labels.append(f"{token} (context_not_found)")
        
        return labels
    
    @staticmethod
    def print_summary_statistics(embeddings: np.ndarray):
        """
        Print summary statistics for embeddings.
        
        Args:
            embeddings: Array of embeddings
        """
        print(f"Embedding Statistics:")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Mean: {embeddings.mean():.4f}")
        print(f"  Std: {embeddings.std():.4f}")
        print(f"  Min: {embeddings.min():.4f}")
        print(f"  Max: {embeddings.max():.4f}")
        
        # Compute pairwise distances for additional stats
        from scipy.spatial.distance import pdist
        distances = pdist(embeddings, metric='euclidean')
        print(f"  Pairwise distance mean: {distances.mean():.4f}")
        print(f"  Pairwise distance std: {distances.std():.4f}")
