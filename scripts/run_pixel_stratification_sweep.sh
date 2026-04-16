#!/bin/bash
set -euo pipefail
#
# Pixel-space stratification sweep.
#
# Tests whether raw pixel patches at varying granularities (patch_size x stride)
# exhibit stratified manifold structure across datasets.
#
# Sweep axes:
#   dataset    : CIFAR10 (32px), STL10 (96px), FOOD101 (224px)
#   patch_size : 4, 8, 16
#   stride     : same-as-patch (non-overlapping), half-patch (overlapping)
#   max_tokens : 4096  (higher than default for stable dimension estimates)
#
# Usage:
#   bash scripts/run_pixel_stratification_sweep.sh           # full sweep
#   DRY_RUN=true bash scripts/run_pixel_stratification_sweep.sh  # preview

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
OUT_ROOT="${OUT_ROOT:-/scratch/$USER/runs/llm-stratified/pixel_stratification_sweep}"
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

# ── sweep grid ──────────────────────────────────────────────────────────
DATASETS=("CIFAR10" "STL10")
PATCH_SIZES=(4 8 16)
# stride_mode: "full" = stride == patch_size, "half" = stride == patch_size/2
STRIDE_MODES=("full" "half")

mkdir -p "$OUT_ROOT" "$HF_HOME" "$TRANSFORMERS_CACHE" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR" "$TMPDIR"
export HF_HOME TRANSFORMERS_CACHE WANDB_CACHE_DIR WANDB_DATA_DIR WANDB_ARTIFACT_DIR TMPDIR

run_one() {
  local ds="$1" ps="$2" stride_mode="$3"
  local stride
  if [[ "$stride_mode" == "half" ]]; then
    stride=$(( ps / 2 ))
    [[ $stride -lt 1 ]] && stride=1
  else
    stride="$ps"
  fi

  local tag="${ds,,}_ps${ps}_stride${stride}"
  local run_dir="$OUT_ROOT/$tag"

  echo "━━━ $tag ━━━"
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "  [dry-run] dataset=$ds patch_size=$ps pixel_patch_stride=$stride"
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
  if [[ "$stride" != "$ps" ]]; then
    cmd+=("volume_probe.pixel_patch_stride=$stride")
  fi

  "${cmd[@]}" 2>&1 | tee "$run_dir/log.txt"

  echo "  -> $run_dir"
}

total=0
for ds in "${DATASETS[@]}"; do
  for ps in "${PATCH_SIZES[@]}"; do
    for sm in "${STRIDE_MODES[@]}"; do
      # Skip if patch_size > image_size (e.g. ps=16 on CIFAR 32px is fine, but ps=32 would not be)
      # Also skip half-stride when ps=4 (stride=2 is very fine, slow, and noisy)
      if [[ "$sm" == "half" && "$ps" -le 4 ]]; then
        continue
      fi
      total=$((total + 1))
    done
  done
done
echo "Sweep: $total configurations across ${#DATASETS[@]} datasets"
echo ""

for ds in "${DATASETS[@]}"; do
  for ps in "${PATCH_SIZES[@]}"; do
    for sm in "${STRIDE_MODES[@]}"; do
      if [[ "$sm" == "half" && "$ps" -le 4 ]]; then
        continue
      fi
      run_one "$ds" "$ps" "$sm"
      echo ""
    done
  done
done

echo ""
echo "Sweep complete. Results in: $OUT_ROOT"
echo "Analyze with:  conda run -n $CONDA_ENV python scripts/analyze_pixel_sweep.py --sweep-dir $OUT_ROOT"
