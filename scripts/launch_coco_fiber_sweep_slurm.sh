#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT="${EXPERIMENT:-coco_fiber}"
JOB_NAME="${JOB_NAME:-coco-fiber-sweep}"
PARTITION="${PARTITION:-gpu-redhat}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-96000}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/data}"
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/llm-stratified/${EXPERIMENT}_sweep}"
SNAPSHOT_PARENT="${SNAPSHOT_PARENT:-/scratch/$USER/submission_snapshots}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-stratified-manifold-learning}"
LR_VALUES="${LR_VALUES:-3e-4,1e-4,5e-5}"
SEED_VALUES="${SEED_VALUES:-1337,2025,4242}"
PATCH_CONFIGS="${PATCH_CONFIGS:-16:16:16:1536;32:32:32:2048}"
WEIGHT_DECAY_VALUES="${WEIGHT_DECAY_VALUES:-0.05}"
LABEL_SMOOTHING_VALUES="${LABEL_SMOOTHING_VALUES:-0.0}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-2}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Missing data root: $DATA_ROOT" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT/coco/train2017" || ! -d "$DATA_ROOT/coco/val2017" ]]; then
  echo "COCO dataset not found under $DATA_ROOT/coco" >&2
  exit 1
fi
command -v rsync >/dev/null 2>&1 || { echo "rsync_not_found" >&2; exit 1; }
command -v sbatch >/dev/null 2>&1 || { echo "sbatch_not_found" >&2; exit 1; }

LR_VALUES="${LR_VALUES// /}"
SEED_VALUES="${SEED_VALUES// /}"
PATCH_CONFIGS="${PATCH_CONFIGS// /}"
WEIGHT_DECAY_VALUES="${WEIGHT_DECAY_VALUES// /}"
LABEL_SMOOTHING_VALUES="${LABEL_SMOOTHING_VALUES// /}"

IFS=',' read -r -a LR_LIST <<< "$LR_VALUES"
IFS=',' read -r -a SEED_LIST <<< "$SEED_VALUES"
IFS=';' read -r -a PATCH_CONFIG_LIST <<< "$PATCH_CONFIGS"
IFS=',' read -r -a WEIGHT_DECAY_LIST <<< "$WEIGHT_DECAY_VALUES"
IFS=',' read -r -a LABEL_SMOOTHING_LIST <<< "$LABEL_SMOOTHING_VALUES"

if (( ${#LR_LIST[@]} == 0 )); then
  echo "No learning rates configured in LR_VALUES" >&2
  exit 1
fi
if (( ${#SEED_LIST[@]} == 0 )); then
  echo "No seeds configured in SEED_VALUES" >&2
  exit 1
fi
if (( ${#PATCH_CONFIG_LIST[@]} == 0 )); then
  echo "No patch profiles configured in PATCH_CONFIGS" >&2
  exit 1
fi
if (( ${#WEIGHT_DECAY_LIST[@]} == 0 )); then
  echo "No values configured in WEIGHT_DECAY_VALUES" >&2
  exit 1
fi
if (( ${#LABEL_SMOOTHING_LIST[@]} == 0 )); then
  echo "No values configured in LABEL_SMOOTHING_VALUES" >&2
  exit 1
fi
for patch_cfg in "${PATCH_CONFIG_LIST[@]}"; do
  IFS=':' read -r patch_sz train_bs test_bs max_tokens <<< "$patch_cfg"
  if [[ -z "${patch_sz:-}" || -z "${train_bs:-}" || -z "${test_bs:-}" || -z "${max_tokens:-}" ]]; then
    echo "Invalid PATCH_CONFIGS entry '$patch_cfg'; expected patch:train_bs:test_bs:max_tokens" >&2
    exit 1
  fi
done

TOTAL_JOBS=$((${#LR_LIST[@]} * ${#SEED_LIST[@]} * ${#PATCH_CONFIG_LIST[@]} * ${#WEIGHT_DECAY_LIST[@]} * ${#LABEL_SMOOTHING_LIST[@]}))
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

chmod +x "$SNAPSHOT_REPO/scripts/run_coco_fiber_job.sh" "$SNAPSHOT_REPO/scripts/launch_coco_fiber_sweep_slurm.sh"

export ROOT_DIR="$SNAPSHOT_REPO"
export EXPERIMENT
export DATA_ROOT
export OUT_ROOT
export LR_VALUES
export SEED_VALUES
export PATCH_CONFIGS
export WEIGHT_DECAY_VALUES
export LABEL_SMOOTHING_VALUES
export GPUS
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

JOB_ID="$(sbatch --parsable "${SBATCH_ARGS[@]}" "$SNAPSHOT_REPO/scripts/run_coco_fiber_job.sh" "$@")"

echo "Submitted batch job $JOB_ID"
echo "Snapshot: $SNAPSHOT_REPO"
echo "Logs: $LOG_DIR"
echo "Output root: $OUT_ROOT"
echo "Learning rates: $LR_VALUES"
echo "Seeds: $SEED_VALUES"
echo "Patch configs: $PATCH_CONFIGS"
echo "Weight decays: $WEIGHT_DECAY_VALUES"
echo "Label smoothing values: $LABEL_SMOOTHING_VALUES"
echo "Total jobs: $TOTAL_JOBS"
if [[ -n "$EXCLUDE_NODES" ]]; then
  echo "Excluded nodes: $EXCLUDE_NODES"
fi
