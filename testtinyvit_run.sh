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
srun python tinyvit.py \
  --dataset FOOD101 \
  --root "$DATADIR" \
  --outdir "$OUTDIR" \
  --img-size 224 \
  --patch-size 16 \
  --embed-dim 384 \
  --depth 10 \
  --num-heads 6 \
  --epochs 20 \
  --num-workers ${SLURM_CPUS_PER_TASK} \
  --compile

# ---------- Alternative runs (uncomment one) ----------
# CelebA (multi-label, 40 attrs)
# srun python tinyvit_train.py \
#   --dataset CELEBA \
#   --root "$DATADIR" \
#   --outdir "$OUTDIR" \
#   --img-size 64 \
#   --patch-size 8 \
#   --embed-dim 256 \
#   --depth 8 \
#   --num-heads 4 \
#   --epochs 5 \
#   --num-workers ${SLURM_CPUS_PER_TASK} \
#   --compile

# STL10 (96x96)
# srun python tinyvit_train.py \
#   --dataset STL10 \
#   --root "$DATADIR" \
#   --outdir "$OUTDIR" \
#   --img-size 96 \
#   --patch-size 8 \
#   --embed-dim 256 \
#   --depth 8 \
#   --num-heads 4 \
#   --epochs 20 \
#   --num-workers ${SLURM_CPUS_PER_TASK} \
#   --compile

# -------- Optional: Singularity fallback (avoids old GLIBC) --------
# module load singularity  # or apptainer, if available
# IMG=docker://pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
# srun singularity exec --nv $IMG \
#   bash -lc "python tinyvit_train.py \
#     --dataset FOOD101 \
#     --root '$DATADIR' \
#     --outdir '$OUTDIR' \
#     --img-size 224 \
#     --patch-size 16 \
#     --embed-dim 384 \
#     --depth 10 \
#     --num-heads 6 \
#     --epochs 20 \
#     --num-workers ${SLURM_CPUS_PER_TASK} \
#     --compile"
