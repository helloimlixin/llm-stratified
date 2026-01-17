#!/bin/bash
#SBATCH --partition=gpu                  # queue
#SBATCH --requeue                       # return to queue if preempted
#SBATCH --job-name=tinyvit_food101      # job name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8               # match DataLoader workers
#SBATCH --gres=gpu:1                    # single GPU (script is single-GPU)
#SBATCH --mem=128100                    # MB
#SBATCH --time=05:00:00
#SBATCH --output=tinyvit.%j.out         # STDOUT
#SBATCH --error=tinyvit.%j.err          # STDERR

echo "==== Environment ===="
date
hostname
nvidia-smi

# Repo root (so the script works from any cwd)
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# --- safer defaults for PyTorch on HPC ---
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- workspace ---
WORKDIR=/scratch/$USER/test_tinyvit
DATADIR=/scratch/$USER/data
OUTDIR=/scratch/$USER/test_tinyvit/runs
mkdir -p "$WORKDIR" "$DATADIR" "$OUTDIR"
cd "$WORKDIR"

# --- conda env (adjust if your env name differs) ---
# If your env is not "research", change it below.
source ~/.bashrc
conda activate tinyvit

# If you hit GLIBC issues with this env on your node,
# comment the conda lines above and use the Singularity section at the bottom.

# --- run (Food-101 example; change flags for other datasets) ---
srun python "$ROOT_DIR/src/train.py" \
  data=food101 \
  data.root="$DATADIR" \
  data.img_size=224 \
  data.num_workers=${SLURM_CPUS_PER_TASK} \
  model.patch_size=16 \
  model.embed_dim=384 \
  model.depth=10 \
  model.num_heads=6 \
  model.compile=true \
  training.epochs=20 \
  hydra.run.dir="$OUTDIR"

# ---------- Alternative runs (uncomment one) ----------
# CelebA (multi-label, 40 attrs)
# srun python "$ROOT_DIR/src/train.py" \
#   data=celeba \
#   data.root="$DATADIR" \
#   data.img_size=64 \
#   data.num_workers=${SLURM_CPUS_PER_TASK} \
#   model.patch_size=8 \
#   model.embed_dim=256 \
#   model.depth=8 \
#   model.num_heads=4 \
#   model.compile=true \
#   training.epochs=5 \
#   hydra.run.dir="$OUTDIR"

# STL10 (96x96)
# srun python "$ROOT_DIR/src/train.py" \
#   data=stl10 \
#   data.root="$DATADIR" \
#   data.img_size=96 \
#   data.num_workers=${SLURM_CPUS_PER_TASK} \
#   model.patch_size=8 \
#   model.embed_dim=256 \
#   model.depth=8 \
#   model.num_heads=4 \
#   model.compile=true \
#   training.epochs=20 \
#   hydra.run.dir="$OUTDIR"

# -------- Optional: Singularity fallback (avoids old GLIBC) --------
# module load singularity  # or apptainer, if available
# IMG=docker://pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
# srun singularity exec --nv $IMG \
#   bash -lc "python $ROOT_DIR/src/train.py \
#     data=food101 \
#     data.root='$DATADIR' \
#     data.img_size=224 \
#     data.num_workers=${SLURM_CPUS_PER_TASK} \
#     model.patch_size=16 \
#     model.embed_dim=384 \
#     model.depth=10 \
#     model.num_heads=6 \
#     model.compile=true \
#     training.epochs=20 \
#     hydra.run.dir='$OUTDIR'"
