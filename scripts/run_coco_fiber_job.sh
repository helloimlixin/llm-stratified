#!/bin/bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
EXPERIMENT="${EXPERIMENT:-coco_fiber}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/data}"
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/llm-stratified/${EXPERIMENT}_sweep}"
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYTHONUSERBASE_DIR="${PYTHONUSERBASE_DIR:-/scratch/$USER/.pydeps/llm_stratified_py311}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-stratified-manifold-learning}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/scratch/$USER/.secrets/wandb_api_key}"
WANDB_VERSION="${WANDB_VERSION:-0.19.11}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LR_VALUES="${LR_VALUES:-3e-4,1e-4,5e-5}"
SEED_VALUES="${SEED_VALUES:-1337,2025,4242}"
PATCH_CONFIGS="${PATCH_CONFIGS:-16:16:16:1536;32:32:32:2048}"
WEIGHT_DECAY_VALUES="${WEIGHT_DECAY_VALUES:-0.05}"
LABEL_SMOOTHING_VALUES="${LABEL_SMOOTHING_VALUES:-0.0}"
PROGRESS_LOG_INTERVAL="${PROGRESS_LOG_INTERVAL:-100}"
GPUS="${GPUS:-1}"
DRY_RUN="${DRY_RUN:-false}"

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Missing root dir: $ROOT_DIR" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Missing data root: $DATA_ROOT" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT/coco/train2017" || ! -d "$DATA_ROOT/coco/val2017" ]]; then
  echo "COCO dataset not found under $DATA_ROOT/coco" >&2
  exit 1
fi
if ! [[ "$GPUS" =~ ^[0-9]+$ ]] || (( GPUS < 1 )); then
  echo "GPUS must be a positive integer; got '$GPUS'" >&2
  exit 1
fi
if [[ "$WANDB_ENABLED" == "true" && "$WANDB_MODE" == "online" && ! -f "$WANDB_API_KEY_FILE" ]]; then
  echo "Missing W&B API key file for online mode: $WANDB_API_KEY_FILE" >&2
  exit 1
fi

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
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( ${#LR_LIST[@]} == 0 )); then
  echo "No learning rates configured in LR_VALUES" >&2
  exit 1
fi
if (( ${#SEED_LIST[@]} == 0 || ${#PATCH_CONFIG_LIST[@]} == 0 || ${#WEIGHT_DECAY_LIST[@]} == 0 || ${#LABEL_SMOOTHING_LIST[@]} == 0 )); then
  echo "Sweep axes must all be non-empty" >&2
  exit 1
fi
TOTAL_JOBS=$((${#LR_LIST[@]} * ${#SEED_LIST[@]} * ${#PATCH_CONFIG_LIST[@]} * ${#WEIGHT_DECAY_LIST[@]} * ${#LABEL_SMOOTHING_LIST[@]}))
if (( TASK_INDEX < 0 || TASK_INDEX >= TOTAL_JOBS )); then
  echo "Array task index $TASK_INDEX is out of range for total jobs=$TOTAL_JOBS" >&2
  exit 1
fi

remaining="$TASK_INDEX"
lr_idx=$(( remaining % ${#LR_LIST[@]} ))
remaining=$(( remaining / ${#LR_LIST[@]} ))
seed_idx=$(( remaining % ${#SEED_LIST[@]} ))
remaining=$(( remaining / ${#SEED_LIST[@]} ))
patch_idx=$(( remaining % ${#PATCH_CONFIG_LIST[@]} ))
remaining=$(( remaining / ${#PATCH_CONFIG_LIST[@]} ))
weight_decay_idx=$(( remaining % ${#WEIGHT_DECAY_LIST[@]} ))
remaining=$(( remaining / ${#WEIGHT_DECAY_LIST[@]} ))
label_smoothing_idx=$(( remaining % ${#LABEL_SMOOTHING_LIST[@]} ))
remaining=$(( remaining / ${#LABEL_SMOOTHING_LIST[@]} ))
if (( remaining != 0 )); then
  echo "Internal sweep index decode failure for TASK_INDEX=$TASK_INDEX" >&2
  exit 1
fi

TRAINING_LR="${LR_LIST[$lr_idx]}"
TRAINING_SEED="${SEED_LIST[$seed_idx]}"
PATCH_CONFIG="${PATCH_CONFIG_LIST[$patch_idx]}"
TRAINING_WEIGHT_DECAY="${WEIGHT_DECAY_LIST[$weight_decay_idx]}"
TRAINING_LABEL_SMOOTHING="${LABEL_SMOOTHING_LIST[$label_smoothing_idx]}"

IFS=':' read -r PATCH_SIZE TRAIN_BATCH_SIZE TEST_BATCH_SIZE FIBER_MAX_TOKENS <<< "$PATCH_CONFIG"
if [[ -z "${PATCH_SIZE:-}" || -z "${TRAIN_BATCH_SIZE:-}" || -z "${TEST_BATCH_SIZE:-}" || -z "${FIBER_MAX_TOKENS:-}" ]]; then
  echo "Invalid patch config '$PATCH_CONFIG'; expected patch:train_bs:test_bs:max_tokens" >&2
  exit 1
fi

LR_TAG="$(printf '%s' "$TRAINING_LR" | tr -c '[:alnum:]' '_')"
WD_TAG="$(printf '%s' "$TRAINING_WEIGHT_DECAY" | tr -c '[:alnum:]' '_')"
LS_TAG="$(printf '%s' "$TRAINING_LABEL_SMOOTHING" | tr -c '[:alnum:]' '_')"
PROFILE_TAG="ps_${PATCH_SIZE}_bs_${TRAIN_BATCH_SIZE}_bt_${TEST_BATCH_SIZE}_mt_${FIBER_MAX_TOKENS}"

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    set +u
    source /etc/profile.d/modules.sh
    set -u
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    set +u
    source /usr/share/Modules/init/bash
    set -u
  fi
fi
if ! command -v singularity >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load singularity 2>/dev/null || true
    module load singularityce 2>/dev/null || true
    module load singularity-ce 2>/dev/null || true
  fi
fi
command -v singularity >/dev/null 2>&1 || { echo "singularity_not_found" >&2; exit 1; }

USER_NAME="$(id -un)"
PYTHON_SITE="$PYTHONUSERBASE_DIR/lib/python3.11/site-packages"
JOB_BASE="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
RUN_DIR="$OUT_ROOT/${PROFILE_TAG}/seed_${TRAINING_SEED}/wd_${WD_TAG}/ls_${LS_TAG}/lr_${LR_TAG}/job_${JOB_BASE}_task_${TASK_INDEX}"
WANDB_DIR="${WANDB_DIR:-$RUN_DIR/wandb}"
WANDB_RUN_NAME="${EXPERIMENT}_${PROFILE_TAG}_seed_${TRAINING_SEED}_wd_${TRAINING_WEIGHT_DECAY}_ls_${TRAINING_LABEL_SMOOTHING}_lr_${TRAINING_LR}_job_${JOB_BASE}_task_${TASK_INDEX}"
HF_HOME="${HF_HOME:-/scratch/$USER_NAME/.cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

mkdir -p "$OUT_ROOT" "$RUN_DIR" "$PYTHONUSERBASE_DIR" "$HF_HOME" "$WANDB_DIR"
if [[ -f "$WANDB_API_KEY_FILE" ]]; then
  export WANDB_API_KEY="$(tr -d '\r\n' < "$WANDB_API_KEY_FILE")"
fi

export PYTHONUSERBASE="$PYTHONUSERBASE_DIR"
export PYTHONNOUSERSITE=0
export PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PATH="$PYTHONUSERBASE_DIR/bin${PATH:+:$PATH}"
export WANDB_MODE WANDB_PROJECT WANDB_DIR HF_HOME TRANSFORMERS_CACHE
TOTAL_CPUS="${SLURM_CPUS_PER_TASK:-8}"
PER_PROC_THREADS=$(( TOTAL_CPUS / GPUS ))
if (( PER_PROC_THREADS < 1 )); then
  PER_PROC_THREADS=1
fi
export OMP_NUM_THREADS="$PER_PROC_THREADS"
export MKL_NUM_THREADS="$PER_PROC_THREADS"

BIND_ARGS=(
  --bind "$ROOT_DIR"
  --bind "/scratch/$USER_NAME"
  --bind "$DATA_ROOT"
  --bind "$OUT_ROOT"
)

deps_ready() {
  PYTHONUSERBASE="$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}" \
    singularity exec "${BIND_ARGS[@]}" "$IMAGE" python3 - <<'PY_DEP' >/dev/null 2>&1
import hydra
import matplotlib
import PIL
import scipy
import sklearn
import timm
import torchvision
import wandb
PY_DEP
}

install_deps() {
  PYTHONUSERBASE="$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}" \
    singularity exec "${BIND_ARGS[@]}" "$IMAGE" python3 -m pip install --user --upgrade \
      hydra-core matplotlib pillow scipy scikit-learn timm "torchvision==${TORCHVISION_VERSION}" "wandb==${WANDB_VERSION}"
}

CMD=(
  python3 "$ROOT_DIR/src/train.py"
  "+experiment=$EXPERIMENT"
  "hydra.run.dir=$RUN_DIR"
  "data.root=$DATA_ROOT"
  "data.num_workers=$NUM_WORKERS"
  "data.batch_size=$TRAIN_BATCH_SIZE"
  "data.batch_size_test=$TEST_BATCH_SIZE"
  "model.patch_size=$PATCH_SIZE"
  "fiber.max_tokens=$FIBER_MAX_TOKENS"
  "seed=$TRAINING_SEED"
  "training.lr=$TRAINING_LR"
  "training.weight_decay=$TRAINING_WEIGHT_DECAY"
  "training.label_smoothing=$TRAINING_LABEL_SMOOTHING"
)

if [[ "$PROGRESS_LOG_INTERVAL" =~ ^[0-9]+$ ]] && (( PROGRESS_LOG_INTERVAL > 0 )); then
  CMD+=("training.progress_log_interval=$PROGRESS_LOG_INTERVAL")
fi

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
echo "  lr=$TRAINING_LR"
echo "  seed=$TRAINING_SEED"
echo "  patch_config=$PATCH_CONFIG"
echo "  weight_decay=$TRAINING_WEIGHT_DECAY"
echo "  label_smoothing=$TRAINING_LABEL_SMOOTHING"
echo "  run_dir=$RUN_DIR"

LAUNCH=( "${CMD[@]}" )
if (( GPUS > 1 )); then
  LAUNCH=(
    torchrun
    --standalone
    "--nproc_per_node=$GPUS"
    "$ROOT_DIR/src/train.py"
    compute=ddp
    "${CMD[@]:2}"
  )
fi

if [[ "$DRY_RUN" == "true" ]]; then
  printf 'DRY_RUN command:'
  printf ' %q' "${LAUNCH[@]}"
  printf '\n'
  exit 0
fi

if ! deps_ready; then
  LOCK_DIR="$PYTHONUSERBASE_DIR/.bootstrap_lock"
  until mkdir "$LOCK_DIR" 2>/dev/null; do
    sleep 2
  done
  cleanup_lock() {
    rmdir "$LOCK_DIR" 2>/dev/null || true
  }
  trap cleanup_lock EXIT
  if ! deps_ready; then
    install_deps
  fi
  cleanup_lock
  trap - EXIT
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

exec srun singularity exec --nv "${BIND_ARGS[@]}" "$IMAGE" "${LAUNCH[@]}"
