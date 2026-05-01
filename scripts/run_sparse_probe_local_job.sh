#!/bin/bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/data}"
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/llm-stratified/sparse_probe_local_sweep}"
CONDA_ENV="${CONDA_ENV:-tinyvit}"
SEED_VALUES="${SEED_VALUES:-1337}"
RESIDUAL_THRESHOLD="${RESIDUAL_THRESHOLD:-0.15}"
NEIGHBOR_K="${NEIGHBOR_K:-32}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-stratified-manifold-learning}"
DRY_RUN="${DRY_RUN:-false}"

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Missing root dir: $ROOT_DIR" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Missing data root: $DATA_ROOT" >&2
  exit 1
fi

SEED_VALUES="${SEED_VALUES// /}"

IFS=',' read -r -a SEED_LIST <<< "$SEED_VALUES"

TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

if (( ${#SEED_LIST[@]} == 0 )); then
  echo "Sweep axes must all be non-empty" >&2
  exit 1
fi

TOTAL_JOBS=${#SEED_LIST[@]}
if (( TASK_INDEX < 0 || TASK_INDEX >= TOTAL_JOBS )); then
  echo "Array task index $TASK_INDEX is out of range for total jobs=$TOTAL_JOBS" >&2
  exit 1
fi

remaining="$TASK_INDEX"
seed_idx=$(( remaining % ${#SEED_LIST[@]} ))
remaining=$(( remaining / ${#SEED_LIST[@]} ))
if (( remaining != 0 )); then
  echo "Internal sweep index decode failure for TASK_INDEX=$TASK_INDEX" >&2
  exit 1
fi

TRAINING_SEED="${SEED_LIST[$seed_idx]}"

JOB_BASE="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
RUN_DIR="$OUT_ROOT/seed_${TRAINING_SEED}/job_${JOB_BASE}_task_${TASK_INDEX}"
WANDB_RUN_NAME="sparse_probe_local_seed${TRAINING_SEED}_job${JOB_BASE}_task${TASK_INDEX}"

mkdir -p "$RUN_DIR"

CONDA_RUN="conda run --no-capture-output -n $CONDA_ENV"

CMD=(
  $CONDA_RUN python "$ROOT_DIR/src/train.py"
  "data=stl10"
  "fiber=sparse_probe"
  "hydra.run.dir=$RUN_DIR"
  "data.root=$DATA_ROOT"
  "seed=$TRAINING_SEED"
  "fiber.sparse_probe_residual_threshold=$RESIDUAL_THRESHOLD"
  "fiber.sparse_probe_auto_neighbor_k=$NEIGHBOR_K"
)

if [[ "$WANDB_ENABLED" == "true" ]]; then
  CMD+=(
    "wandb.enabled=true"
    "wandb.project=$WANDB_PROJECT"
    "wandb.name=$WANDB_RUN_NAME"
  )
else
  CMD+=("wandb.enabled=false")
fi

CMD+=("$@")

echo "Sweep task $TASK_INDEX / $((TOTAL_JOBS - 1))"
echo "  seed=$TRAINING_SEED"
echo "  neighbor_k=$NEIGHBOR_K"
echo "  residual_threshold=$RESIDUAL_THRESHOLD"
echo "  run_dir=$RUN_DIR"

if [[ "$DRY_RUN" == "true" ]]; then
  printf 'DRY_RUN command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

exec "${CMD[@]}"
