#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_NAME="${JOB_NAME:-sparse-probe-k-sweep}"
PARTITION="${PARTITION:-gpu-redhat}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-48000}"
TIME_LIMIT="${TIME_LIMIT:-6:00:00}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/data}"
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/llm-stratified/sparse_probe_k_sweep}"
SNAPSHOT_PARENT="${SNAPSHOT_PARENT:-/scratch/$USER/submission_snapshots}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-stratified-manifold-learning}"
K_VALUES="${K_VALUES:-8,16,32,64}"
SEED_VALUES="${SEED_VALUES:-1337}"
RESIDUAL_THRESHOLD="${RESIDUAL_THRESHOLD:-0.15}"
CONDA_ENV="${CONDA_ENV:-tinyvit}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-4}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Missing data root: $DATA_ROOT" >&2
  exit 1
fi
command -v rsync >/dev/null 2>&1 || { echo "rsync not found" >&2; exit 1; }
command -v sbatch >/dev/null 2>&1 || { echo "sbatch not found" >&2; exit 1; }

K_VALUES="${K_VALUES// /}"
SEED_VALUES="${SEED_VALUES// /}"

IFS=',' read -r -a K_LIST <<< "$K_VALUES"
IFS=',' read -r -a SEED_LIST <<< "$SEED_VALUES"

if (( ${#K_LIST[@]} == 0 )); then
  echo "No K values configured" >&2
  exit 1
fi
if (( ${#SEED_LIST[@]} == 0 )); then
  echo "No seeds configured" >&2
  exit 1
fi

TOTAL_JOBS=$(( ${#K_LIST[@]} * ${#SEED_LIST[@]} ))
DEFAULT_ARRAY_SPEC="0-$((TOTAL_JOBS - 1))"
if [[ "$ARRAY_CONCURRENCY" =~ ^[0-9]+$ ]] && (( ARRAY_CONCURRENCY > 0 )); then
  DEFAULT_ARRAY_SPEC="${DEFAULT_ARRAY_SPEC}%${ARRAY_CONCURRENCY}"
fi
ARRAY_SPEC="${ARRAY_SPEC:-$DEFAULT_ARRAY_SPEC}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SNAPSHOT_DIR="${SNAPSHOT_PARENT}/${JOB_NAME}_${TIMESTAMP}"
SNAPSHOT_REPO="${SNAPSHOT_DIR}/repo"
LOG_DIR="${SNAPSHOT_DIR}/logs"

mkdir -p "$SNAPSHOT_REPO" "$LOG_DIR" "$OUT_ROOT"

rsync -a \
  --exclude '/.git/' \
  --exclude '/runs/' \
  --exclude '/wandb/' \
  --exclude '/data/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.out' \
  --exclude '*.err' \
  "$ROOT_DIR/" "$SNAPSHOT_REPO/"

chmod +x "$SNAPSHOT_REPO/scripts/run_sparse_probe_k_job.sh"

export ROOT_DIR="$SNAPSHOT_REPO"
export DATA_ROOT
export OUT_ROOT
export K_VALUES
export SEED_VALUES
export RESIDUAL_THRESHOLD
export CONDA_ENV
export WANDB_ENABLED
export WANDB_MODE
export WANDB_PROJECT

SBATCH_ARGS=(
  --partition="$PARTITION"
  --requeue
  --job-name="$JOB_NAME"
  --array="$ARRAY_SPEC"
  --nodes=1
  --ntasks=1
  --cpus-per-task="$CPUS_PER_TASK"
  --gres="gpu:${GPUS}"
  --mem="$MEM_MB"
  --time="$TIME_LIMIT"
  --chdir="$SNAPSHOT_REPO"
  --output="$LOG_DIR/%x_%A_%a.out"
  --error="$LOG_DIR/%x_%A_%a.err"
)
if [[ -n "$EXCLUDE_NODES" ]]; then
  SBATCH_ARGS+=("--exclude=$EXCLUDE_NODES")
fi

JOB_ID="$(sbatch --parsable "${SBATCH_ARGS[@]}" "$SNAPSHOT_REPO/scripts/run_sparse_probe_k_job.sh" "$@")"

echo "Submitted batch job $JOB_ID"
echo "Snapshot: $SNAPSHOT_REPO"
echo "Logs: $LOG_DIR"
echo "Output root: $OUT_ROOT"
echo "K values: $K_VALUES"
echo "Seeds: $SEED_VALUES"
echo "Residual threshold: $RESIDUAL_THRESHOLD"
echo "Total jobs: $TOTAL_JOBS"
if [[ -n "$EXCLUDE_NODES" ]]; then
  echo "Excluded nodes: $EXCLUDE_NODES"
fi
