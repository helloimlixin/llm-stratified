# TinyViT — Fiber Bundle Tests

This repo focuses on running **fiber bundle / stratified diagnostics** on a TinyViT-style Vision Transformer, and producing an image-heavy experiment report (`RESULTS.md`) that renders cleanly on GitHub.

The main entry point is:

- `tinyvit_fiber_bundle.py`

## Installation

### Requirements

- Python 3.10+
- PyTorch + torchvision
- `wandb` (optional; logs are still written locally even without online sync)

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

### Fiber bundle test run (recommended)

```bash
python tinyvit_fiber_bundle.py \
  --dataset STL10 \
  --root ./data \
  --img-size 96 \
  --patch-size 32 \
  --epochs 200 \
  --embed-interval 10 \
  --subset-test 64 \
  --batch-size 64 \
  --max-tokens 8192 \
  --outdir ./runs/fiber_test_stl10_96 \
  --wandb \
  --project tinyvit_fiber \
  --run-name fiber_test_stl10_96
```

### View results

- **Report**: `RESULTS.md`
- **GitHub-safe images**: `imgs/` (all image references in `RESULTS.md` point here)

If you re-run experiments and regenerate plots, update/copy the new plots into `imgs/` and keep `RESULTS.md` referencing `imgs/...` paths.

## What gets produced

The run output directory (`--outdir`) contains:

- **Metrics**
  - `train_history.json`: per-epoch train/val loss + accuracy
  - `fiber_history.json`: per-checkpoint fiber summary statistics
- **Saved visualizations**
  - `fiber_analysis/epoch_XXX_*.png`: token radius curves, token patches, low/high-dim panels, token-slot counts, patch-count curves, etc.
  - `class_dims_epoch_XXX.png`: class-wise dimension summary at checkpoints

`RESULTS.md` embeds a curated subset of these plots across epochs (and copies the necessary assets into `imgs/` for GitHub rendering).

## Supported datasets

| Dataset | Default Image Size | Classes | Task Type |
|---------|-------------------|---------|-----------|
| CIFAR10 | 32×32 | 10 | Multiclass |
| CIFAR100 | 32×32 | 100 | Multiclass |
| STL10 | 96×96 | 10 | Multiclass |
| Food101 | 224×224 | 101 | Multiclass |
| Flowers102 | 224×224 | 102 | Multiclass |
| SVHN | 32×32 | 10 | Multiclass |
| CelebA | 64×64 | 40 | Multilabel |

## Key command-line arguments (fiber bundle script)

```bash
--dataset               Dataset name
--root                  Dataset root directory (default: ./data)
--img-size              Input image size (dataset-dependent default)
--patch-size            Patch size
--epochs                Number of training epochs
--embed-interval        Save/checkpoint cadence for embedding + fiber plots
--subset-train          Optional training subset size
--subset-test           Optional test subset size
--max-tokens            Cap number of token embeddings used for visuals
--outdir                Output directory (writes runs + plots)
--wandb                 Enable W&B logging (optional)
--project / --run-name  W&B project and run name (optional)
```

## Repo layout

```
.
├── tinyvit_fiber_bundle.py   # main: TinyViT + fiber bundle diagnostics + plot dumping
├── runs/                     # training outputs (metrics, checkpoints, plots)
├── imgs/                     # curated copies of plots referenced by RESULTS.md (GitHub-safe)
└── RESULTS.md                # main report
```

