#!/bin/bash
set -euo pipefail
#
# Pretrained-vs-pixel stratification sweep.
#
# Compares frozen DINOv2-base representations against raw pixel patches.
# DINOv2 uses patch_size=14 natively.
#
# Sweep axes:
#   dataset        : CIFAR10 (32px), STL10 (96px)
#   backbone       : dinov2 (pretrained, frozen) vs untrained TinyViT
#   dinov2_layers  : single layer (-1 = last) vs multi-layer (3,6,-1)
#   pixel_stride   : full (=patch_size), half (=patch_size/2)

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/llm-stratified/pretrained_pixel_sweep}"
DATA_ROOT="${DATA_ROOT:-/cache/home/xl598/Projects/data}"
CONDA_ENV="${CONDA_ENV:-tinyvit}"
DRY_RUN="${DRY_RUN:-false}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
VOL_MAX="${VOL_MAX:-128}"
HF_HOME="${HF_HOME:-/scratch/$USER/.cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-stratified-manifold-learning}"
WANDB_TAGS="${WANDB_TAGS:-}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/scratch/$USER/.cache/wandb}"
WANDB_DATA_DIR="${WANDB_DATA_DIR:-$WANDB_CACHE_DIR/data}"
WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$WANDB_CACHE_DIR/artifacts}"
TMPDIR="${TMPDIR:-/scratch/$USER/.cache/wandb/tmp}"

mkdir -p "$OUT_ROOT" "$HF_HOME" "$TRANSFORMERS_CACHE" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR" "$TMPDIR"
export HF_HOME TRANSFORMERS_CACHE WANDB_CACHE_DIR WANDB_DATA_DIR WANDB_ARTIFACT_DIR TMPDIR

run_dinov2() {
  local ds="$1" layers_tag="$2" layers_val="$3" stride_tag="$4" stride_val="$5"
  local tag="${ds,,}_dinov2_${layers_tag}_${stride_tag}"
  local run_dir="$OUT_ROOT/$tag"

  echo "--- $tag ---"
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "  [dry-run] dataset=$ds layers=$layers_val pixel_stride=$stride_val"
    return 0
  fi
  mkdir -p "$run_dir"

  local -a cmd=(
    conda run -n "$CONDA_ENV"
    python "$ROOT_DIR/src/train.py"
    "+experiment=pretrained_pixel_sweep"
    "data=${ds,,}"
    "volume_probe.max_tokens=$MAX_TOKENS"
    "volume_probe.vol_max=$VOL_MAX"
    "data.root=$DATA_ROOT"
    "data.batch_size_test=16"
    "wandb.enabled=$WANDB_ENABLED"
    "wandb.project=$WANDB_PROJECT"
    "wandb.name=$tag"
    "hydra.run.dir=$run_dir"
  )
  if [[ -n "$WANDB_TAGS" ]]; then
    cmd+=("wandb.tags=$WANDB_TAGS")
  fi
  if [[ "$layers_val" != "null" ]]; then
    cmd+=("volume_probe.dinov2_layers=[$layers_val]")
  fi
  if [[ "$stride_val" != "null" ]]; then
    cmd+=("volume_probe.pixel_patch_stride=$stride_val")
  fi

  "${cmd[@]}" 2>&1 | tee "$run_dir/log.txt"

  echo "  -> $run_dir"
}

run_untrained() {
  local ds="$1" ps="$2"
  local tag="${ds,,}_untrained_ps${ps}"
  local run_dir="$OUT_ROOT/$tag"

  echo "--- $tag ---"
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "  [dry-run] dataset=$ds untrained patch_size=$ps"
    return 0
  fi
  mkdir -p "$run_dir"

  local -a cmd=(
    conda run -n "$CONDA_ENV"
    python "$ROOT_DIR/src/train.py"
    "+experiment=pixel_stratification_sweep"
    "data=${ds,,}"
    "model.patch_size=$ps"
    "volume_probe.max_tokens=$MAX_TOKENS"
    "volume_probe.vol_max=$VOL_MAX"
    "data.root=$DATA_ROOT"
    "wandb.enabled=$WANDB_ENABLED"
    "wandb.project=$WANDB_PROJECT"
    "wandb.name=$tag"
    "hydra.run.dir=$run_dir"
  )
  if [[ -n "$WANDB_TAGS" ]]; then
    cmd+=("wandb.tags=$WANDB_TAGS")
  fi

  "${cmd[@]}" 2>&1 | tee "$run_dir/log.txt"

  echo "  -> $run_dir"
}

DATASETS=("CIFAR10" "STL10")

total=0
for ds in "${DATASETS[@]}"; do
  total=$((total + 4))
done
echo "Pretrained sweep: $total configurations across ${#DATASETS[@]} datasets"
echo ""

for ds in "${DATASETS[@]}"; do
  # DINOv2 last layer only, full stride
  run_dinov2 "$ds" "last" "null" "full" "null"
  echo ""
  # DINOv2 multi-layer (3,6,last), full stride
  run_dinov2 "$ds" "multilayer" "3,6,-1" "full" "null"
  echo ""
  # DINOv2 last layer, half stride (stride=7)
  run_dinov2 "$ds" "last" "null" "halfstride" "7"
  echo ""
  # Untrained TinyViT baseline for comparison
  run_untrained "$ds" 8
  echo ""
done

echo ""
echo "Sweep complete. Results in: $OUT_ROOT"
echo "Analyze with:  conda run -n $CONDA_ENV python scripts/analyze_pixel_sweep.py --sweep-dir $OUT_ROOT"
