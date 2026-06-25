#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXPERIMENT="${EXPERIMENT:-quick_test}"
DATA_ROOT="${DATA_ROOT:-${LLM_STRATIFIED_DATA_ROOT:-$ROOT_DIR/../data}}"
OUT_ROOT="${OUT_ROOT:-${LLM_STRATIFIED_OUTPUT_ROOT:-runs}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT_ROOT}/local/${EXPERIMENT}/${RUN_STAMP}"

mkdir -p "$RUN_DIR"
export PYTHONUNBUFFERED=1
export LLM_STRATIFIED_DATA_ROOT="$DATA_ROOT"
export LLM_STRATIFIED_OUTPUT_ROOT="$OUT_ROOT"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$WANDB_ENABLED" != "true" ]]; then
  export WANDB_MODE=disabled
fi

echo "Running local experiment '${EXPERIMENT}'"
echo "Data root: ${DATA_ROOT}"
echo "Run dir:   ${RUN_DIR}"
echo "Python:    ${PYTHON_BIN}"

exec "$PYTHON_BIN" "$ROOT_DIR/src/train.py" \
  "+experiment=${EXPERIMENT}" \
  "data.root=${DATA_ROOT}" \
  "output_root=${OUT_ROOT}" \
  "hydra.run.dir=${RUN_DIR}" \
  "data.num_workers=0" \
  "wandb.enabled=${WANDB_ENABLED}" \
  "$@"
