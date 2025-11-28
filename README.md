# TinyViT - Vision Transformer Training Script

A PyTorch implementation of TinyViT (Vision Transformer) with multi-GPU support, comprehensive dataset support, and integrated wandb logging.

## Features

- ✅ **Multi-GPU Training**: DistributedDataParallel (DDP) support for efficient multi-GPU training
- ✅ **Auto GPU Detection**: Automatically detects and uses multiple GPUs
- ✅ **Wandb Integration**: Comprehensive experiment tracking with Weights & Biases
- ✅ **Multiple Datasets**: Support for CIFAR10, CIFAR100, STL10, Food101, Flowers102, SVHN, and CelebA
- ✅ **Advanced Training**: Cosine annealing, label smoothing, gradient clipping, warmup scheduling
- ✅ **Model Compilation**: Optional `torch.compile` for faster training
- ✅ **Flexible Configuration**: Extensive CLI arguments for easy experimentation

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+ (with CUDA support for GPU training)
- torchvision
- wandb (optional, for experiment tracking)

### Setup

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision

# Install wandb (optional)
pip install wandb

# Login to wandb (optional, for online logging)
wandb login
```

## Quick Start

### Single GPU Training

```bash
python tinyvit.py --dataset CIFAR10 --epochs 20 --wandb --project tinyvit
```

### Multi-GPU Training (Auto-detected)

The script automatically detects multiple GPUs and enables DDP:

```bash
python tinyvit.py --dataset CIFAR10 --epochs 20 --wandb --project tinyvit
```

### Multi-GPU Training with torchrun (Recommended)

```bash
torchrun --nproc_per_node=2 tinyvit.py --dataset CIFAR10 --epochs 20 --wandb --project tinyvit
```

### Using the Launch Script

```bash
./run_multi_gpu.sh CIFAR10 --epochs 50 --wandb --project tinyvit
```

## Supported Datasets

| Dataset | Default Image Size | Classes | Task Type |
|---------|-------------------|---------|-----------|
| CIFAR10 | 32×32 | 10 | Multiclass |
| CIFAR100 | 32×32 | 100 | Multiclass |
| STL10 | 96×96 | 10 | Multiclass |
| Food101 | 224×224 | 101 | Multiclass |
| Flowers102 | 224×224 | 102 | Multiclass |
| SVHN | 32×32 | 10 | Multiclass |
| CelebA | 64×64 | 40 | Multilabel |

## Command-Line Arguments

### Dataset Options

```bash
--dataset {CIFAR10,CIFAR100,STL10,FOOD101,FLOWERS102,SVHN,CELEBA}
                        Dataset to use (default: CIFAR10)
--root PATH             Root directory for datasets (default: ./data)
--img-size SIZE         Image size (default: dataset-specific)
--batch-size SIZE       Training batch size (default: 128)
--batch-size-test SIZE  Test batch size (default: 256)
--num-workers N          Number of data loader workers (default: 4)
```

### Model Architecture

```bash
--patch-size SIZE       Patch size for image embedding (default: 4)
--embed-dim DIM         Embedding dimension (default: 192)
--depth N               Number of transformer blocks (default: 8)
--num-heads N           Number of attention heads (default: 3)
--mlp-ratio RATIO       MLP expansion ratio (default: 2.0)
--dropout RATE          Dropout rate (default: 0.1)
```

### Training Options

```bash
--epochs N              Number of training epochs (default: 10)
--lr RATE               Learning rate (default: 3e-4)
--wd RATE               Weight decay (default: 0.05)
--grad-clip VALUE       Gradient clipping threshold (default: 1.0)
--label-smoothing VALUE Label smoothing coefficient (default: 0.0)
--warmup-epochs N       Warmup epochs (default: auto)
--no-cosine             Disable cosine annealing schedule
--compile               Use torch.compile for faster training
```

### Multi-GPU Options

```bash
--ddp                   Explicitly enable DistributedDataParallel
--local-rank N          Local rank for distributed training (auto-set by torchrun)
```

### Wandb Options

```bash
--wandb                 Enable wandb logging
--project NAME          Wandb project name (default: tinyvit)
--run-name NAME         Wandb run name (optional)
```

### Output Options

```bash
--outdir PATH           Output directory for checkpoints (default: ./tinyvit_runs)
--save-interval N       Checkpoint save interval (default: 2)
--runs N                Number of independent runs (default: 1)
--seed-base N           Base seed for runs (default: 1337)
```

## Usage Examples

### CIFAR10 Training

```bash
python tinyvit.py \
    --dataset CIFAR10 \
    --epochs 50 \
    --batch-size 128 \
    --lr 3e-4 \
    --wandb \
    --project tinyvit \
    --run-name cifar10_baseline
```

### Food101 with Custom Architecture

```bash
torchrun --nproc_per_node=2 tinyvit.py \
    --dataset FOOD101 \
    --img-size 224 \
    --patch-size 16 \
    --embed-dim 384 \
    --depth 10 \
    --num-heads 6 \
    --epochs 20 \
    --batch-size 64 \
    --wandb \
    --project tinyvit \
    --compile
```

### CelebA Multi-label Classification

```bash
python tinyvit.py \
    --dataset CELEBA \
    --img-size 64 \
    --patch-size 8 \
    --embed-dim 256 \
    --depth 8 \
    --num-heads 4 \
    --epochs 10 \
    --wandb \
    --project tinyvit_celeba
```

### Multi-GPU Training with Offline Wandb

```bash
WANDB_MODE=offline torchrun --nproc_per_node=2 tinyvit.py \
    --dataset CIFAR10 \
    --epochs 20 \
    --wandb \
    --project tinyvit
```

## Multi-GPU Training Details

### How It Works

1. **Auto-detection**: The script automatically detects multiple GPUs and enables DDP
2. **Batch Size**: Total batch size is divided across GPUs (e.g., `--batch-size 128` with 2 GPUs = 64 per GPU)
3. **Learning Rate**: Automatically scaled by number of GPUs (linear scaling rule)
4. **Data Distribution**: Uses `DistributedSampler` to distribute data across GPUs
5. **Metrics**: Metrics are synchronized and averaged across all GPUs
6. **Checkpoints**: Only saved on rank 0 (main process)

### Performance Tips

- Use `torchrun` for better process management
- Set `--num-workers` based on your CPU cores (typically 4-8)
- Use `--compile` for faster training (PyTorch 2.0+)
- Adjust batch size based on GPU memory

## Wandb Logging

### Online Logging (Recommended)

1. Login to wandb:
   ```bash
   wandb login
   ```

2. Run training:
   ```bash
   python tinyvit.py --wandb --project tinyvit
   ```

### Offline Logging

Logs are saved locally and can be synced later:

```bash
WANDB_MODE=offline python tinyvit.py --wandb --project tinyvit
wandb sync  # Sync later when online
```

### Logged Metrics

- `train/loss`: Training loss
- `train/acc`: Training accuracy
- `val/loss`: Validation loss
- `val/acc`: Validation accuracy
- `lr`: Learning rate
- `epoch`: Current epoch

### Configuration Tracking

All hyperparameters are automatically logged to wandb, including:
- Model architecture (embed_dim, depth, num_heads, etc.)
- Training parameters (lr, batch_size, epochs, etc.)
- Multi-GPU settings (num_gpus, use_ddp, effective_batch_size)
- Dataset information

## Output Structure

```
tinyvit_runs/
├── CIFAR10_run_000/
│   ├── epoch_000.pt
│   ├── epoch_002.pt
│   ├── epoch_004.pt
│   └── epoch_009.pt
└── CIFAR10_run_001/
    └── ...
```

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Training metrics
- Configuration (dataset, architecture, etc.)
- Timestamp and seed

## Model Architecture

TinyViT consists of:
- **Patch Embedding**: Convolutional patch embedding layer
- **Positional Embedding**: Learnable positional embeddings
- **Transformer Blocks**: Multi-head self-attention + MLP
- **Classification Head**: Linear layer for final predictions

## Troubleshooting

### Wandb Not Logging

- Check if logged in: `wandb status`
- Login: `wandb login`
- Or use offline mode: `WANDB_MODE=offline`

### Multi-GPU Issues

- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU count: `python -c "import torch; print(torch.cuda.device_count())"`
- Use `torchrun` instead of manual DDP setup

### Out of Memory

- Reduce batch size: `--batch-size 64`
- Reduce model size: `--embed-dim 128 --depth 6`
- Use gradient accumulation (modify code)

### Slow Training

- Enable compilation: `--compile`
- Increase workers: `--num-workers 8`
- Use mixed precision (already enabled for CUDA)

## Citation

If you use this code, please cite:

```bibtex
@misc{tinyvit2024,
  title={TinyViT: Vision Transformer Training Script},
  author={Your Name},
  year={2024}
}
```

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
