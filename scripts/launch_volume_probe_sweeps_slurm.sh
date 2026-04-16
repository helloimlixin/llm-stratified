#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/cache/home/xl598/Projects/data}"
OUT_ROOT_BASE="${OUT_ROOT_BASE:-/scratch/$USER/runs/llm-stratified}"
SNAPSHOT_PARENT="${SNAPSHOT_PARENT:-/scratch/$USER/submission_snapshots}"
PARTITION="${PARTITION:-gpu-redhat}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_MB="${MEM_MB:-96000}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
CONDA_ENV="${CONDA_ENV:-tinyvit}"
SWEEP_MODES="${SWEEP_MODES:-pixel,pretrained}"
HF_HOME="${HF_HOME:-/scratch/$USER/.cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-stratified-manifold-learning}"
WANDB_TAGS="${WANDB_TAGS:-}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/scratch/$USER/.cache/wandb}"
WANDB_DATA_DIR="${WANDB_DATA_DIR:-$WANDB_CACHE_DIR/data}"
WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$WANDB_CACHE_DIR/artifacts}"
TMPDIR="${TMPDIR:-/scratch/$USER/.cache/wandb/tmp}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Missing data root: $DATA_ROOT" >&2
  exit 1
fi
command -v rsync >/dev/null 2>&1 || { echo "rsync_not_found" >&2; exit 1; }
command -v sbatch >/dev/null 2>&1 || { echo "sbatch_not_found" >&2; exit 1; }

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LABEL="${RUN_LABEL:-${TIMESTAMP}_volume_probe_visual_sweep}"
RUN_ROOT="${OUT_ROOT_BASE}/${RUN_LABEL}"
SNAPSHOT_DIR="${SNAPSHOT_PARENT}/volume_probe_sweeps_${TIMESTAMP}"
SNAPSHOT_REPO="${SNAPSHOT_DIR}/repo"
LOG_DIR="${SNAPSHOT_DIR}/logs"

mkdir -p "$SNAPSHOT_REPO" "$LOG_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$RUN_ROOT" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR" "$TMPDIR"

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

chmod +x \
  "$SNAPSHOT_REPO/scripts/run_pixel_stratification_sweep.sh" \
  "$SNAPSHOT_REPO/scripts/run_pretrained_pixel_sweep.sh" \
  "$SNAPSHOT_REPO/scripts/launch_volume_probe_sweeps_slurm.sh"

IFS=',' read -r -a MODE_LIST <<< "${SWEEP_MODES// /}"

submit_one() {
  local mode="$1"
  local job_name script_path out_root job_id
  case "$mode" in
    pixel)
      job_name="pixel-strat-sweep"
      script_path="$SNAPSHOT_REPO/scripts/run_pixel_stratification_sweep.sh"
      out_root="$RUN_ROOT/pixel_stratification_sweep"
      ;;
    pretrained)
      job_name="pretrained-pixel-sweep"
      script_path="$SNAPSHOT_REPO/scripts/run_pretrained_pixel_sweep.sh"
      out_root="$RUN_ROOT/pretrained_pixel_sweep"
      ;;
    *)
      echo "Unsupported sweep mode: $mode" >&2
      exit 1
      ;;
  esac

  mkdir -p "$out_root"

  local -a sbatch_args=(
    --parsable
    --partition="$PARTITION"
    --requeue
    --job-name="$job_name"
    --nodes=1
    --ntasks=1
    --cpus-per-task="$CPUS_PER_TASK"
    --gres="gpu:${GPUS}"
    --mem="$MEM_MB"
    --time="$TIME_LIMIT"
    --chdir="$SNAPSHOT_REPO"
    --output="$LOG_DIR/%x_%j.out"
    --error="$LOG_DIR/%x_%j.err"
    --export="ALL,ROOT_DIR=$SNAPSHOT_REPO,DATA_ROOT=$DATA_ROOT,OUT_ROOT=$out_root,CONDA_ENV=$CONDA_ENV,HF_HOME=$HF_HOME,TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE,WANDB_ENABLED=$WANDB_ENABLED,WANDB_PROJECT=$WANDB_PROJECT,WANDB_TAGS=$WANDB_TAGS,WANDB_CACHE_DIR=$WANDB_CACHE_DIR,WANDB_DATA_DIR=$WANDB_DATA_DIR,WANDB_ARTIFACT_DIR=$WANDB_ARTIFACT_DIR,TMPDIR=$TMPDIR"
  )
  if [[ "$GPUS" =~ ^[0-9]+$ ]] && (( GPUS > 0 )); then
    sbatch_args+=("--gres=gpu:${GPUS}")
  fi
  if [[ -n "$EXCLUDE_NODES" ]]; then
    sbatch_args+=("--exclude=$EXCLUDE_NODES")
  fi

  job_id="$(sbatch "${sbatch_args[@]}" "$script_path")"
  echo "$mode:$job_id:$out_root"
}

submitted=()
for mode in "${MODE_LIST[@]}"; do
  [[ -n "$mode" ]] || continue
  submitted+=("$(submit_one "$mode")")
done

echo "Snapshot: $SNAPSHOT_REPO"
echo "Logs: $LOG_DIR"
echo "Run root: $RUN_ROOT"
jobs_file="$RUN_ROOT/submitted_jobs.txt"
: > "$jobs_file"
for item in "${submitted[@]}"; do
  IFS=':' read -r mode job_id out_root <<< "$item"
  echo "Submitted $mode sweep job $job_id -> $out_root"
  echo "$mode $job_id $out_root" >> "$jobs_file"
done
echo "Combined report command:"
echo "  conda run -n $CONDA_ENV python $SNAPSHOT_REPO/scripts/analyze_pixel_sweep.py --sweep-dir $RUN_ROOT/pixel_stratification_sweep --sweep-dir $RUN_ROOT/pretrained_pixel_sweep --output $RUN_ROOT/volume_probe_sweep_report.json --report $RUN_ROOT/volume_probe_sweep_report.md"
