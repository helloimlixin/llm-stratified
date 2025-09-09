"""
Simple test script to verify the conda environment setup.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test basic imports to verify environment setup."""
    print("Testing basic imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    try:
        import plotly
        print(f"✅ Plotly {plotly.__version__}")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        import umap
        print(f"✅ UMAP {umap.__version__}")
    except ImportError as e:
        print(f"❌ UMAP import failed: {e}")
        return False
    
    try:
        import ripser
        print(f"✅ Ripser {ripser.__version__}")
    except ImportError as e:
        print(f"❌ Ripser import failed: {e}")
        return False
    
    try:
        import openai
        print(f"✅ OpenAI {openai.__version__}")
    except ImportError as e:
        print(f"❌ OpenAI import failed: {e}")
        return False
    
    return True

def test_project_structure():
    """Test project structure and basic functionality."""
    print("\nTesting project structure...")
    
    # Check if src directory exists
    if not os.path.exists('src'):
        print("❌ src directory not found")
        return False
    print("✅ src directory exists")
    
    # Check if main modules exist
    modules = [
        'src/models/roberta_model.py',
        'src/models/bert_model.py', 
        'src/models/gpt3_model.py',
        'src/geometric_tools/moe_models.py',
        'src/geometric_tools/training.py',
        'src/utils/data_utils.py'
    ]
    
    for module in modules:
        if os.path.exists(module):
            print(f"✅ {module}")
        else:
            print(f"❌ {module} not found")
            return False
    
    return True

def main():
    """Main test function."""
    print("🧪 Testing Stratified Manifold Learning Environment")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        return False
    
    # Test project structure
    if not test_project_structure():
        print("\n❌ Project structure tests failed!")
        return False
    
    print("\n" + "=" * 50)
    print("✅ All tests passed! Environment is ready.")
    print("\n📋 Next steps:")
    print("   1. Run experiments: python main.py --model roberta --samples-per-domain 100")
    print("   2. Check configuration: cat config/default_config.json")
    print("   3. View documentation: cat README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
