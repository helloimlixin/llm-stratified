#!/bin/bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/data}"
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/llm-stratified/sparse_probe_pretrained_sweep}"
CONDA_ENV="${CONDA_ENV:-tinyvit}"
MODELS="${MODELS:-timm_vit,timm_vit_small}"
DATASETS="${DATASETS:-food101,celebahq,coco}"
SEED_VALUES="${SEED_VALUES:-1337}"
RESIDUAL_THRESHOLD="${RESIDUAL_THRESHOLD:-0.15}"
NEIGHBOR_K="${NEIGHBOR_K:-32}"
EPOCHS="${EPOCHS:-3}"
EMBED_INTERVAL="${EMBED_INTERVAL:-1}"
SUBSET_TEST="${SUBSET_TEST:-512}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-stratified-manifold-learning}"
DRY_RUN="${DRY_RUN:-false}"

if [[ ! -d "$ROOT_DIR" ]]; then echo "Missing root dir: $ROOT_DIR" >&2; exit 1; fi
if [[ ! -d "$DATA_ROOT" ]]; then echo "Missing data root: $DATA_ROOT" >&2; exit 1; fi

MODELS="${MODELS// /}"
DATASETS="${DATASETS// /}"
SEED_VALUES="${SEED_VALUES// /}"

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
IFS=',' read -r -a SEED_LIST <<< "$SEED_VALUES"

TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

if (( ${#MODEL_LIST[@]} == 0 || ${#DATASET_LIST[@]} == 0 || ${#SEED_LIST[@]} == 0 )); then
  echo "Sweep axes must all be non-empty" >&2; exit 1
fi

TOTAL_JOBS=$(( ${#MODEL_LIST[@]} * ${#DATASET_LIST[@]} * ${#SEED_LIST[@]} ))
if (( TASK_INDEX < 0 || TASK_INDEX >= TOTAL_JOBS )); then
  echo "Array task index $TASK_INDEX out of range (total=$TOTAL_JOBS)" >&2; exit 1
fi

remaining="$TASK_INDEX"
model_idx=$(( remaining % ${#MODEL_LIST[@]} )); remaining=$(( remaining / ${#MODEL_LIST[@]} ))
dataset_idx=$(( remaining % ${#DATASET_LIST[@]} )); remaining=$(( remaining / ${#DATASET_LIST[@]} ))
seed_idx=$(( remaining % ${#SEED_LIST[@]} )); remaining=$(( remaining / ${#SEED_LIST[@]} ))
if (( remaining != 0 )); then echo "Internal decode failure" >&2; exit 1; fi

MODEL="${MODEL_LIST[$model_idx]}"
DATASET="${DATASET_LIST[$dataset_idx]}"
TRAINING_SEED="${SEED_LIST[$seed_idx]}"

JOB_BASE="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
RUN_DIR="$OUT_ROOT/${MODEL}/${DATASET}/seed_${TRAINING_SEED}/job_${JOB_BASE}_task_${TASK_INDEX}"
WANDB_RUN_NAME="sparse_probe_${MODEL}_${DATASET}_seed${TRAINING_SEED}_job${JOB_BASE}_task${TASK_INDEX}"
mkdir -p "$RUN_DIR"

CONDA_RUN="conda run --no-capture-output -n $CONDA_ENV"

CMD=(
  $CONDA_RUN python "$ROOT_DIR/src/train.py"
  "model=$MODEL"
  "hydra.run.dir=$RUN_DIR"
  "seed=$TRAINING_SEED"
  "fiber=sparse_probe"
  "fiber.sparse_probe_residual_threshold=$RESIDUAL_THRESHOLD"
  "fiber.sparse_probe_auto_neighbor_k=$NEIGHBOR_K"
  "fiber.embed_interval=$EMBED_INTERVAL"
  "fiber.embedding_animation=false"
  "training.epochs=$EPOCHS"
  "training.lr=1e-4"
  "training.warmup_epochs=0"
  "training.cosine_schedule=false"
)

if [[ "$MODEL" == "sam_base" ]]; then
  BS_TRAIN_DEFAULT=4; BS_TEST_DEFAULT=8
elif [[ "$MODEL" == "dinov2_base" ]]; then
  BS_TRAIN_DEFAULT=16; BS_TEST_DEFAULT=32
else
  BS_TRAIN_DEFAULT=32; BS_TEST_DEFAULT=64
fi

case "$DATASET" in
  food101)
    CMD+=(
      "data=food101"
      "data.root=$DATA_ROOT"
      "data.img_size=224"
      "data.batch_size=$BS_TRAIN_DEFAULT"
      "data.batch_size_test=$BS_TEST_DEFAULT"
      "data.subset_test=$SUBSET_TEST"
      "data.num_workers=4"
    )
    ;;
  celebahq)
    CMD+=(
      "data=celebahq"
      "data.img_size=224"
      "data.batch_size=$BS_TRAIN_DEFAULT"
      "data.batch_size_test=$BS_TEST_DEFAULT"
      "data.subset_test=$SUBSET_TEST"
      "data.num_workers=4"
    )
    ;;
  coco)
    coco_bs_train=$BS_TRAIN_DEFAULT
    coco_bs_test=$BS_TEST_DEFAULT
    if (( coco_bs_train > 16 )); then coco_bs_train=16; fi
    if (( coco_bs_test > 16 )); then coco_bs_test=16; fi
    CMD+=(
      "data=coco"
      "data.root=$DATA_ROOT"
      "data.img_size=224"
      "data.batch_size=$coco_bs_train"
      "data.batch_size_test=$coco_bs_test"
      "data.subset_test=$SUBSET_TEST"
      "data.num_workers=0"
    )
    ;;
  dtd)
    CMD+=(
      "data=dtd"
      "data.root=$DATA_ROOT"
      "data.img_size=224"
      "data.batch_size=$BS_TRAIN_DEFAULT"
      "data.batch_size_test=$BS_TEST_DEFAULT"
      "data.subset_test=$SUBSET_TEST"
      "data.num_workers=4"
    )
    ;;
  flowers102)
    CMD+=(
      "data=flowers102"
      "data.root=$DATA_ROOT"
      "data.img_size=224"
      "data.batch_size=$BS_TRAIN_DEFAULT"
      "data.batch_size_test=$BS_TEST_DEFAULT"
      "data.subset_test=$SUBSET_TEST"
      "data.num_workers=4"
    )
    ;;
  clevr)
    CMD+=(
      "data=clevr"
      "data.root=$DATA_ROOT"
      "data.img_size=224"
      "data.batch_size=$BS_TRAIN_DEFAULT"
      "data.batch_size_test=$BS_TEST_DEFAULT"
      "data.subset_test=$SUBSET_TEST"
      "data.num_workers=4"
    )
    ;;
  *)
    echo "Unknown DATASET: $DATASET" >&2; exit 1
    ;;
esac

if [[ "$WANDB_ENABLED" == "true" ]]; then
  CMD+=("wandb.enabled=true" "wandb.project=$WANDB_PROJECT" "wandb.name=$WANDB_RUN_NAME")
else
  CMD+=("wandb.enabled=false")
fi

CMD+=("$@")

echo "Sweep task $TASK_INDEX / $((TOTAL_JOBS - 1))"
echo "  model=$MODEL"
echo "  dataset=$DATASET"
echo "  seed=$TRAINING_SEED"
echo "  epochs=$EPOCHS, embed_interval=$EMBED_INTERVAL"
echo "  run_dir=$RUN_DIR"

if [[ "$DRY_RUN" == "true" ]]; then
  printf 'DRY_RUN command:'; printf ' %q' "${CMD[@]}"; printf '\n'
  exit 0
fi

if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi || true; fi

exec "${CMD[@]}"
