#!/bin/bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
EXPERIMENT="${EXPERIMENT:-coco_sam_fiber}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/data}"
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/llm-stratified/${EXPERIMENT}}"
IMAGE="${IMAGE:-docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
PYTHONUSERBASE_DIR="${PYTHONUSERBASE_DIR:-/scratch/$USER/.pydeps/llm_stratified_py311}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/scratch/$USER/.secrets/wandb_api_key}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-stratified-manifold-learning}"
WANDB_NAME="${WANDB_NAME:-${EXPERIMENT}}"
WANDB_VERSION="${WANDB_VERSION:-0.19.11}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE_TEST="${BATCH_SIZE_TEST:-4}"
SUBSET_TEST="${SUBSET_TEST:-128}"
EPOCHS="${EPOCHS:-100}"
MAX_TOKENS="${MAX_TOKENS:-1536}"
SAM_MODEL="${SAM_MODEL:-facebook/sam-vit-base}"
ANALYSIS_PATCH_SIZE="${ANALYSIS_PATCH_SIZE:-16}"
MAX_BOXES_PER_IMAGE="${MAX_BOXES_PER_IMAGE:-16}"
SPARSE_PROBE_ALGORITHM="${SPARSE_PROBE_ALGORITHM:-omp}"
SPARSE_PROBE_RESIDUAL_THRESHOLD="${SPARSE_PROBE_RESIDUAL_THRESHOLD:-0.15}"
SPARSE_PROBE_MAX_ANCHORS="${SPARSE_PROBE_MAX_ANCHORS:-null}"
SPARSE_PROBE_HEATMAP_IMAGES="${SPARSE_PROBE_HEATMAP_IMAGES:-8}"

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
command -v singularity >/dev/null 2>&1 || { echo singularity_not_found >&2; exit 1; }

USER_NAME="$(id -un)"
PYTHON_SITE="$PYTHONUSERBASE_DIR/lib/python3.11/site-packages"
RUN_DIR="$OUT_ROOT/job_${SLURM_JOB_ID:-manual}"
WANDB_RUN_NAME="${WANDB_NAME}_${SLURM_JOB_ID:-manual}"
HF_HOME="${HF_HOME:-/scratch/$USER_NAME/.cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
WANDB_DIR="${WANDB_DIR:-$RUN_DIR/wandb}"

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

BIND_ARGS=(
  --bind "$ROOT_DIR"
  --bind "/scratch/$USER_NAME"
  --bind "$DATA_ROOT"
  --bind "$OUT_ROOT"
)

if ! PYTHONUSERBASE="$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}" \
  singularity exec "${BIND_ARGS[@]}" "$IMAGE" python3 - <<'PY_DEP' >/dev/null 2>&1
import hydra
import matplotlib
import PIL
import scipy
import sklearn
import timm
import transformers
import wandb
import torchvision
PY_DEP
then
  PYTHONUSERBASE="$PYTHONUSERBASE_DIR" PYTHONNOUSERSITE=0 PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}" \
    singularity exec "${BIND_ARGS[@]}" "$IMAGE" python3 -m pip install --user --upgrade \
      hydra-core matplotlib pillow scipy scikit-learn timm transformers "torchvision==${TORCHVISION_VERSION}" "wandb==${WANDB_VERSION}"
fi

nvidia-smi

CMD=(
  python3 "$ROOT_DIR/src/train.py"
  "+experiment=$EXPERIMENT"
  "hydra.run.dir=$RUN_DIR"
  "data.root=$DATA_ROOT"
  "data.num_workers=$NUM_WORKERS"
  "data.batch_size_test=$BATCH_SIZE_TEST"
  "data.subset_test=$SUBSET_TEST"
  "sam_fiber.epochs=$EPOCHS"
  "sam_fiber.max_tokens=$MAX_TOKENS"
  "sam_fiber.model_name=$SAM_MODEL"
  "sam_fiber.analysis_patch_size=$ANALYSIS_PATCH_SIZE"
  "sam_fiber.max_boxes_per_image=$MAX_BOXES_PER_IMAGE"
  "sam_fiber.sparse_probe_algorithm=$SPARSE_PROBE_ALGORITHM"
  "sam_fiber.sparse_probe_residual_threshold=$SPARSE_PROBE_RESIDUAL_THRESHOLD"
  "sam_fiber.sparse_probe_max_anchors=$SPARSE_PROBE_MAX_ANCHORS"
  "sam_fiber.sparse_probe_heatmap_images=$SPARSE_PROBE_HEATMAP_IMAGES"
  "wandb.enabled=true"
  "wandb.project=$WANDB_PROJECT"
  "wandb.name=$WANDB_RUN_NAME"
)
CMD+=("$@")

exec srun singularity exec --nv "${BIND_ARGS[@]}" "$IMAGE" "${CMD[@]}"
