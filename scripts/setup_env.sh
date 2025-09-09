#!/bin/bash

# Setup script for Stratified Manifold Learning project
# This script creates a conda environment and installs all dependencies

set -e  # Exit on any error

echo "🚀 Setting up Stratified Manifold Learning environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "stratified-manifold-learning"; then
    echo "⚠️  Environment 'stratified-manifold-learning' already exists."
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n stratified-manifold-learning -y
    else
        echo "ℹ️  Using existing environment. Activating..."
        conda activate stratified-manifold-learning
        echo "✅ Environment activated!"
        exit 0
    fi
fi

# Create conda environment from environment.yml
echo "📦 Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Activate the environment
echo "🔄 Activating environment..."
conda activate stratified-manifold-learning

# Verify installation
echo "🔍 Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; torch.cuda.is_available()"; then
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate the environment: conda activate stratified-manifold-learning"
echo "   2. Run a quick test: python example.py"
echo "   3. Run experiments: python main.py --model roberta --samples-per-domain 100"
echo ""
echo "🔧 Environment info:"
echo "   - Environment name: stratified-manifold-learning"
echo "   - Python version: $(python --version)"
echo "   - Conda environment location: $(conda info --envs | grep stratified-manifold-learning | awk '{print $2}')"
echo ""
echo "💡 Tips:"
echo "   - Always activate the environment before working: conda activate stratified-manifold-learning"
echo "   - To deactivate: conda deactivate"
echo "   - To remove environment: conda env remove -n stratified-manifold-learning"
