"""
Minimal working example to test the core functionality.
"""

import sys
import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from datasets import load_dataset

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_functionality():
    """Test basic functionality without complex imports."""
    print("Testing basic functionality...")
    
    # Test PyTorch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create some dummy data
    X = torch.randn(100, 64)
    print(f"Created dummy data: {X.shape}")
    
    # Test PCA
    pca = PCA(n_components=32)
    X_reduced = pca.fit_transform(X.numpy())
    print(f"PCA reduction: {X_reduced.shape}")
    
    # Test clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X_reduced)
    print(f"Clustering result: {len(np.unique(clusters))} clusters")
    
    return True

def test_dataset_loading():
    """Test dataset loading functionality."""
    print("\nTesting dataset loading...")
    
    try:
        # Load a small sample
        ds = load_dataset("imdb", split="train[:10]")
        print(f"✅ Loaded IMDB dataset: {len(ds)} samples")
        
        # Check structure
        print(f"   Columns: {ds.column_names}")
        print(f"   Sample text: {ds[0]['text'][:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

def test_geometric_tools():
    """Test geometric tools imports."""
    print("\nTesting geometric tools...")
    
    try:
        # Test MoE models import
        from src.geometric_tools.moe_models import TopKSTE, MixtureOfDictionaryExperts
        print("✅ MoE models imported successfully")
        
        # Test training utilities
        from src.geometric_tools.training import contrastive_loss_with_labels
        print("✅ Training utilities imported successfully")
        
        # Test geometric analysis
        from src.geometric_tools.geometric_analysis import LocalDimensionEstimator
        print("✅ Geometric analysis imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Geometric tools import failed: {e}")
        return False

def test_model_imports():
    """Test model imports."""
    print("\nTesting model imports...")
    
    try:
        # Test RoBERTa model
        from src.models.roberta_model import roberta_embed_text, load_multidomain_sentiment
        print("✅ RoBERTa model imported successfully")
        
        # Test BERT model
        from src.models.bert_model import bert_embed_text
        print("✅ BERT model imported successfully")
        
        # Test GPT-3 model
        from src.models.gpt3_model import gpt3_embed_text
        print("✅ GPT-3 model imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Model imports failed: {e}")
        return False

def main():
    """Main test function."""
    print("🔍 Minimal Functionality Test")
    print("=" * 40)
    
    tests = [
        test_basic_functionality,
        test_dataset_loading,
        test_geometric_tools,
        test_model_imports
    ]
    
    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✅ All tests passed!")
        print("\n📋 The environment is working correctly.")
        print("   You can now run experiments with confidence.")
    else:
        print("❌ Some tests failed.")
        print("   Check the error messages above for details.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
