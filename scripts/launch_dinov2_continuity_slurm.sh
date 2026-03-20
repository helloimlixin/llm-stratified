#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT="${EXPERIMENT:-dinov2_continuity}"
JOB_NAME="${JOB_NAME:-dinov2-continuity}"
PARTITION="${PARTITION:-gpu}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-96000}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
LOG_PREFIX="${LOG_PREFIX:-dinov2_continuity}"
DATA_ROOT="${DATA_ROOT:-/cache/home/xl598/Projects/data}"
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/llm-stratified/${EXPERIMENT}}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/runs/slurm_logs}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/scratch/$USER/.secrets/wandb_api_key}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Missing data root: $DATA_ROOT" >&2
  exit 1
fi
if [[ "$WANDB_MODE" == "online" && ! -f "$WANDB_API_KEY_FILE" ]]; then
  echo "Missing W&B API key file for online mode: $WANDB_API_KEY_FILE" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT" "$LOG_DIR"

sbatch \
  --partition="$PARTITION" \
  --requeue \
  --job-name="$JOB_NAME" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$CPUS_PER_TASK" \
  --gres="gpu:${GPUS}" \
  --mem="$MEM_MB" \
  --time="$TIME_LIMIT" \
  --chdir="$ROOT_DIR" \
  --output="$LOG_DIR/${LOG_PREFIX}_%j.out" \
  --error="$LOG_DIR/${LOG_PREFIX}_%j.err" \
  "$ROOT_DIR/scripts/run_dinov2_continuity_job.sh" "$@"
