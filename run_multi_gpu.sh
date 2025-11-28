#!/bin/bash
# Multi-GPU training launch script for TinyViT
# Usage: ./run_multi_gpu.sh [dataset] [other_args...]
# Example: ./run_multi_gpu.sh CIFAR10 --epochs 50 --wandb --project tinyvit

# Default dataset if not provided
DATASET=${1:-CIFAR10}
shift  # Remove first argument, rest are passed to python script

# Number of GPUs (adjust if needed)
NUM_GPUS=${NUM_GPUS:-2}

# Set PyTorch environment variables
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "Launching TinyViT Multi-GPU Training"
echo "Dataset: $DATASET"
echo "GPUs: $NUM_GPUS"
echo "=========================================="
date
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

# Launch with torchrun (recommended method)
torchrun --nproc_per_node=$NUM_GPUS tinyvit.py \
    --dataset "$DATASET" \
    --wandb \
    --project tinyvit \
    "$@"

echo "Training completed!"
