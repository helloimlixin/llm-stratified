#!/bin/bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/setup_coco2017.sh [parent_data_root]

Download COCO 2017 into a layout this repo already understands:
  <parent_data_root>/coco/
    train2017/
    val2017/
    annotations/instances_train2017.json
    annotations/instances_val2017.json

After setup, point the repo at the parent root:
  python src/train.py data=coco data.root=<parent_data_root>

Options:
  --coco-dir PATH     Use an exact COCO dataset directory instead of <parent>/coco
  --verify-only       Only verify the expected files/directories exist
  --keep-archives     Keep downloaded zip files under <coco_dir>/.downloads
  -h, --help          Show this message

Environment:
  COCO_PARENT_ROOT    Default parent root if no positional path is provided
  COCO_DIR            Exact COCO dataset directory (same effect as --coco-dir)
EOF
}

die() {
  echo "[error] $*" >&2
  exit 1
}

log() {
  echo "[coco-setup] $*"
}

VERIFY_ONLY=0
KEEP_ARCHIVES=0
POSITIONAL_ROOT=""
COCO_ROOT="${COCO_DIR:-}"
PARENT_ROOT="${COCO_PARENT_ROOT:-/scratch/$USER/data}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --coco-dir)
      shift
      [[ $# -gt 0 ]] || die "--coco-dir requires a path"
      COCO_ROOT="$1"
      ;;
    --verify-only)
      VERIFY_ONLY=1
      ;;
    --keep-archives)
      KEEP_ARCHIVES=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      if [[ -n "$POSITIONAL_ROOT" ]]; then
        die "only one parent_data_root may be provided"
      fi
      POSITIONAL_ROOT="$1"
      ;;
  esac
  shift
done

if [[ -n "$POSITIONAL_ROOT" && -z "$COCO_ROOT" ]]; then
  PARENT_ROOT="$POSITIONAL_ROOT"
fi
if [[ -z "$COCO_ROOT" ]]; then
  COCO_ROOT="$PARENT_ROOT/coco"
fi

ANN_DIR="$COCO_ROOT/annotations"
TRAIN_DIR="$COCO_ROOT/train2017"
VAL_DIR="$COCO_ROOT/val2017"
TRAIN_JSON="$ANN_DIR/instances_train2017.json"
VAL_JSON="$ANN_DIR/instances_val2017.json"
DOWNLOAD_DIR="$COCO_ROOT/.downloads"

TRAIN_URL="http://images.cocodataset.org/zips/train2017.zip"
VAL_URL="http://images.cocodataset.org/zips/val2017.zip"
ANN_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

verify_layout() {
  local missing=0
  for path in "$TRAIN_DIR" "$VAL_DIR" "$TRAIN_JSON" "$VAL_JSON"; do
    if [[ ! -e "$path" ]]; then
      echo "[missing] $path" >&2
      missing=1
    fi
  done
  if [[ $missing -ne 0 ]]; then
    return 1
  fi
  return 0
}

pick_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo python3
    return 0
  fi
  return 1
}

download_file() {
  local url="$1"
  local dest="$2"
  if [[ -f "$dest" ]]; then
    log "archive already present: $dest"
    return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    log "downloading $(basename "$dest") with wget"
    wget -c -O "$dest" "$url"
    return 0
  fi
  if command -v curl >/dev/null 2>&1; then
    log "downloading $(basename "$dest") with curl"
    curl -fL -C - -o "$dest" "$url"
    return 0
  fi
  die "need either wget or curl to download COCO"
}

extract_zip() {
  local archive="$1"
  local dest_dir="$2"
  if command -v unzip >/dev/null 2>&1; then
    log "extracting $(basename "$archive") with unzip"
    unzip -q "$archive" -d "$dest_dir"
    return 0
  fi

  local py
  py="$(pick_python)" || die "need either unzip or python3 to extract zip archives"
  log "extracting $(basename "$archive") with python zipfile"
  "$py" - "$archive" "$dest_dir" <<'PY'
import sys
from pathlib import Path
from zipfile import ZipFile

archive = Path(sys.argv[1])
dest = Path(sys.argv[2])
with ZipFile(archive) as zf:
    zf.extractall(dest)
PY
}

maybe_install_split() {
  local name="$1"
  local target_path="$2"
  local url="$3"
  local archive="$DOWNLOAD_DIR/${name}.zip"

  if [[ -e "$target_path" ]]; then
    log "already present: $target_path"
    return 0
  fi

  mkdir -p "$DOWNLOAD_DIR"
  download_file "$url" "$archive"
  extract_zip "$archive" "$COCO_ROOT"

  if [[ ! -e "$target_path" ]]; then
    die "expected $target_path after extracting $(basename "$archive")"
  fi

  if [[ $KEEP_ARCHIVES -eq 0 ]]; then
    rm -f "$archive"
  fi
}

if [[ $VERIFY_ONLY -eq 1 ]]; then
  if verify_layout; then
    log "verified COCO layout under $COCO_ROOT"
    exit 0
  fi
  die "COCO layout is incomplete under $COCO_ROOT"
fi

mkdir -p "$COCO_ROOT" "$ANN_DIR"
log "install root: $COCO_ROOT"
log "expected extracted size is large; make sure this filesystem has enough free space"

maybe_install_split "train2017" "$TRAIN_DIR" "$TRAIN_URL"
maybe_install_split "val2017" "$VAL_DIR" "$VAL_URL"
maybe_install_split "annotations_trainval2017" "$TRAIN_JSON" "$ANN_URL"

verify_layout || die "COCO setup finished with missing files under $COCO_ROOT"

log "COCO 2017 is ready under $COCO_ROOT"
log "use it with: python src/train.py data=coco data.root=$(dirname "$COCO_ROOT")"
