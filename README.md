# TinyViT — Fiber Bundle Tests

This repo focuses on running **fiber bundle / stratified diagnostics** on a TinyViT-style Vision Transformer, and producing an image-heavy experiment report (`docs/RESULTS.md`) that renders cleanly on GitHub. The working position is not that natural images lie on a single smooth low-dimensional manifold. Instead, even in continuous pixel space, image data are better treated as a **stratified space with nonuniform local dimension**; learned features may smooth or reorganize that structure, but do not remove the need for stratified diagnostics.

Main entry points:

- `src/train.py` (Hydra-configured training + fiber diagnostics)
- `src/imagegpt.py` (discrete-token ImageGPT-style model + polysemy probes)

## Installation

### Requirements

- Python 3.10+
- PyTorch + torchvision
- `hydra-core` (`src/train.py` uses Hydra for all training / probe runs)
- `wandb` (optional; logs are still written locally even without online sync)
- `transformers` (optional; only needed for the frozen DINOv2 continuity probes)

### Setup

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision

# Install Hydra (required for src/train.py)
pip install hydra-core

# Install wandb (optional)
pip install wandb

# Install transformers (optional, for DINOv2 probing)
pip install transformers

# Login to wandb (optional, for online logging)
wandb login
```

## Quick Start

### Basic training (no fiber)

```bash
python src/train.py \
  data=cifar10 \
  data.root=./data \
  training.epochs=10 \
  hydra.run.dir=./runs/tinyvit/cifar10_baseline
```

### Fiber bundle test run (recommended)

```bash
python src/train.py \
  data=stl10 \
  data.root=./data \
  data.img_size=96 \
  data.batch_size=64 \
  data.subset_test=64 \
  model.patch_size=32 \
  training.epochs=200 \
  fiber=basic \
  fiber.embed_interval=10 \
  fiber.max_tokens=8192 \
  hydra.run.dir=./runs/tinyvit/fiber_test_stl10_96 \
  wandb.enabled=true \
  wandb.project=tinyvit_fiber \
  wandb.name=fiber_test_stl10_96
```

### No-training volume probe (sphere-growth / volume scaling)

This runs the **same volume/stratification estimator** in three spaces:
- **`tokens`**: ViT patch-token embeddings (post-transformer)
- **`patch_embeddings`**: patch-embed vectors (pre-transformer)
- **`patch_pixels`**: raw pixel patches (flattened RGB patches)

Conceptually: for each point, grow a kNN “sphere” and estimate local scaling / stratification.
This is a convenient baseline for testing the stronger claim that **continuous image representations are still not well-described by the classical manifold hypothesis**. Raw patch pixels, patch embeddings, and post-transformer tokens can all exhibit local dimension changes; the question is which representation is most stratified, which is smoothest, and where singular behavior persists.

```bash
python src/train.py \
  volume_probe=basic \
  data=stl10 \
  data.root=./data \
  data.img_size=96 \
  model.patch_size=32 \
  data.subset_test=64 \
  volume_probe.max_tokens=2048
```

Shortcut (same thing, via an experiment preset):

```bash
python src/train.py +experiment=volume_probe data.root=./data
```

Outputs go to `runs/hydra/.../volume_probe/volume_summary.json` (includes per-representation `summary` + `knn_curve`) plus per-representation `*_dims.npy`.

See `docs/STRATIFIED_PIXEL_SPACE.md` for the intended interpretation of these probes and a concrete experiment framing.

To log to W&B:

```bash
python src/train.py +experiment=volume_probe data.root=./data \
  wandb.enabled=true wandb.project=tinyvit_fiber wandb.name=stl10_volume_probe
```

### DINOv2 continuity probe

This preset compares overlapping raw pixel patches against frozen DINOv2 patch features.
It uses `Food101` by default and downloads DINOv2 weights on the first run.

```bash
python src/train.py +experiment=dinov2_continuity data.root=./data
```

Cluster / Singularity launch (mirrors the `../laser` workflow, keeps W&B online if `/scratch/$USER/.secrets/wandb_api_key` exists, and writes `sbatch` logs to `runs/slurm_logs/`):

```bash
scripts/launch_dinov2_continuity_slurm.sh
```

Common overrides:

```bash
EXPERIMENT=dinov2_continuity_flowers \
DATA_ROOT=/cache/home/xl598/Projects/data \
WANDB_NAME=dinov2_flowers_online \
scripts/launch_dinov2_continuity_slurm.sh volume_probe.max_tokens=4096
```

For larger cluster runs or Hydra multiruns, move outputs off the repo tree first:

```bash
export LLM_STRATIFIED_OUTPUT_ROOT=/scratch/$USER/runs/llm-stratified
```

When W&B is enabled, local run files default to `<hydra.run.dir>/wandb` unless `WANDB_DIR` is already set.

### COCO 2017 setup

COCO is supported by the loader but is **not** auto-downloaded. The repo expects:

```text
<data.root>/coco/
  train2017/
  val2017/
  annotations/instances_train2017.json
  annotations/instances_val2017.json
```

Use the helper script to download and extract it:

```bash
scripts/setup_coco2017.sh /scratch/$USER/data
```

Verify an existing install without downloading:

```bash
scripts/setup_coco2017.sh --verify-only /scratch/$USER/data
```

Then run COCO experiments with either dataset alias:

```bash
python src/train.py data=coco data.root=/scratch/$USER/data
python src/train.py data=coco2017 data.root=/scratch/$USER/data
```

Reusable COCO presets:

```bash
# Validated real-data smoke test
python src/train.py +experiment=coco_fiber_smoke data.root=/scratch/$USER/data

# Larger starter run on real COCO
python src/train.py +experiment=coco_fiber data.root=/scratch/$USER/data
```

For COCO specifically, `data.num_workers=0` is the safest default in this repo.
The dataset wrapper keeps a large in-memory annotation object, so worker startup can be slow or stall on small/debug runs.

### Hydra quick test (sanity check)

```bash
python src/train.py +experiment=quick_test
```

`quick_test` uses **`data=fakedata` by default**, so it runs without any dataset downloads.
If you want to sanity-check the full dataset pipeline instead, override the data group, e.g.:

```bash
python src/train.py +experiment=quick_test data=cifar10 data.root=./data
```

### Polysemy study (fiber=polysemy)

```bash
python src/train.py \
  data=stl10 \
  data.root=./data \
  data.img_size=96 \
  model.patch_size=32 \
  training.epochs=50 \
  fiber=polysemy \
  fiber.embed_interval=5 \
  hydra.run.dir=./runs/tinyvit/stl10_polysemy \
  wandb.enabled=true \
  wandb.project=tinyvit_fiber \
  wandb.name=stl10_polysemy
```

### ViT token-polysemy experiment (short command)

This runs the **polysemy clusters vs controls** ablation with robust defaults:

```bash
torchrun --standalone --nproc_per_node=2 src/train.py \
  compute=ddp \
  data=stl10 \
  data.root=./data \
  data.img_size=96 \
  model.patch_size=32 \
  training.epochs=50 \
  fiber=vit_polysemy \
  fiber.embed_interval=5 \
  wandb.enabled=true \
  wandb.project=tinyvit_fiber \
  wandb.name=vit_polysemy_controls
```

### ImageGPT-style discrete-token run (polysemy for image tokens)

This mirrors the “token embedding irregularity / instability” motivation in the fiber-bundle hypothesis work ([arXiv:2504.01002](https://arxiv.org/abs/2504.01002)), but for **discrete image tokens**:

1) **Fit a patch-tokenizer** (k-means over normalized patches)

```bash
python src/imagegpt.py --dataset STL10 --root ./data --img-size 96 --patch-size 8 --codebook 1024 \
  --tokenizer-path ./runs/imagegpt/tokenizer_stl10_96_p8_k1024.npy \
  fit-tokenizer --max-images 5000 --batch-size 64
```

2) **Train a GPT over token sequences** (+ W&B samples)

```bash
python src/imagegpt.py --dataset STL10 --root ./data --img-size 96 --patch-size 8 --codebook 1024 \
  --tokenizer-path ./runs/imagegpt/tokenizer_stl10_96_p8_k1024.npy \
  --outdir ./runs/imagegpt/gpt_stl10_96_p8_k1024 \
  train-gpt --epochs 20 --batch-size 64 --lr 3e-4 --n-embd 384 --n-head 6 --n-layer 6 \
  --wandb --project imagegpt_polysemy --run-name gpt_stl10_96_p8_k1024
```

3) **Probe “polysemy” for a single token**:
   - show real dataset patches assigned to that token
   - generate images while forcing that token at a chosen position

```bash
python src/imagegpt.py --dataset STL10 --root ./data --img-size 96 --patch-size 8 --codebook 1024 \
  --tokenizer-path ./runs/imagegpt/gpt_stl10_96_p8_k1024/tokenizer.npy \
  --outdir ./runs/imagegpt/probe_token_123 \
  probe-polysemy --ckpt ./runs/imagegpt/gpt_stl10_96_p8_k1024/gpt_epoch_019.pt \
  --token-id 123 --max-occ 64 --gen-samples 64 --temperature 1.0 --top-k 64 \
  --wandb --project imagegpt_polysemy --run-name probe_token_123
```

### View results

- **Report**: `docs/RESULTS.md`
- **GitHub-safe images**: `docs/imgs/` (all image references in `docs/RESULTS.md` point here)

If you re-run experiments and regenerate plots, update/copy the new plots into `docs/imgs/` and keep `docs/RESULTS.md` referencing `imgs/...` paths.

When `fiber.enabled=true`, W&B now also logs an `embeddings/progression` animation that shows the 2D PCA embedding layout evolving across fiber-analysis checkpoints.

## What gets produced

The run output directory (`hydra.run.dir`) contains:

- **Metrics**
  - `train_history.json`: per-epoch train/val loss + accuracy
  - `fiber_history.json`: per-checkpoint fiber summary statistics
- **Saved visualizations**
  - `fiber_analysis/epoch_XXX_*.png`: token radius curves, token patches, low/high-dim panels, token-slot counts, patch-count curves, etc.
  - `class_dims_epoch_XXX.png`: class-wise dimension summary at checkpoints
  - `embedding_progression.gif`: 2D PCA animation across fiber-analysis checkpoints when `fiber.embedding_animation=true`

`docs/RESULTS.md` embeds a curated subset of these plots across epochs (and copies the necessary assets into `docs/imgs/` for GitHub rendering).

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

## Key Hydra overrides (fiber diagnostics)

```bash
data=<name>             Dataset config group (e.g., data=stl10)
data.root=...           Dataset root directory (default: ./data)
data.img_size=...       Input image size (dataset-dependent default)
model.patch_size=...    Patch size
training.epochs=...     Number of training epochs
fiber=<name>            Fiber config group (basic, polysemy, vit_polysemy, disabled)
fiber.embed_interval=... Save/checkpoint cadence for embedding + fiber plots
fiber.max_tokens=...    Cap number of token embeddings used for visuals
fiber.embedding_animation=true   Save/log a GIF of embedding progression
fiber.embedding_animation_fps=4  Playback speed for the embedding GIF
data.subset_train=...   Optional training subset size
data.subset_test=...    Optional test subset size
hydra.run.dir=...       Output directory (writes runs + plots)
wandb.enabled=true      Enable W&B logging (optional)
wandb.project=...       W&B project name (optional)
wandb.name=...          W&B run name (optional)
compute=ddp             Enable DDP (use with torchrun)
```

## Tests

```bash
python -m unittest discover -s tests
```

Unit tests avoid dataset downloads and run on CPU.

## Repo layout

```
.
├── src/                       # entrypoints + core modules
├── configs/                   # Hydra configs
├── scripts/                   # helper shell scripts
├── runs/                      # training outputs (metrics, checkpoints, plots)
└── docs/                      # RESULTS.md + imgs/ (GitHub-safe report assets)
```
