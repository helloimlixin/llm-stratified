#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT="${EXPERIMENT:-coco_sam_fiber}"
JOB_NAME="${JOB_NAME:-coco-sam-fiber}"
PARTITION="${PARTITION:-gpu-redhat}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-96000}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/data}"
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/llm-stratified/${EXPERIMENT}}"
SNAPSHOT_PARENT="${SNAPSHOT_PARENT:-/scratch/$USER/submission_snapshots}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/scratch/$USER/.secrets/wandb_api_key}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Missing data root: $DATA_ROOT" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT/coco/train2017" || ! -d "$DATA_ROOT/coco/val2017" ]]; then
  echo "COCO dataset not found under $DATA_ROOT/coco" >&2
  exit 1
fi
if [[ "$WANDB_MODE" == "online" && ! -f "$WANDB_API_KEY_FILE" ]]; then
  echo "Missing W&B API key file for online mode: $WANDB_API_KEY_FILE" >&2
  exit 1
fi
command -v rsync >/dev/null 2>&1 || { echo "rsync_not_found" >&2; exit 1; }
command -v sbatch >/dev/null 2>&1 || { echo "sbatch_not_found" >&2; exit 1; }

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

chmod +x "$SNAPSHOT_REPO/scripts/run_coco_sam_fiber_job.sh" "$SNAPSHOT_REPO/scripts/launch_coco_sam_fiber_slurm.sh"

export ROOT_DIR="$SNAPSHOT_REPO"
export EXPERIMENT
export DATA_ROOT
export OUT_ROOT

JOB_ID="$(sbatch --parsable \
  --partition="$PARTITION" \
  --requeue \
  --job-name="$JOB_NAME" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$CPUS_PER_TASK" \
  --gres="gpu:${GPUS}" \
  --mem="$MEM_MB" \
  --time="$TIME_LIMIT" \
  --chdir="$SNAPSHOT_REPO" \
  --output="$LOG_DIR/%x_%j.out" \
  --error="$LOG_DIR/%x_%j.err" \
  "$SNAPSHOT_REPO/scripts/run_coco_sam_fiber_job.sh" "$@")"

echo "Submitted batch job $JOB_ID"
echo "Snapshot: $SNAPSHOT_REPO"
echo "Logs: $LOG_DIR"
echo "Output root: $OUT_ROOT"
