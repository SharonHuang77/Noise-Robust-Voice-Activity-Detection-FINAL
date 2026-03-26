#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRAIN_DIR="$PROJECT_ROOT/data/generated/train"
DEV_DIR="$PROJECT_ROOT/data/generated/dev"
TEST_DIR="$PROJECT_ROOT/data/generated/test"

RUN_SCRIPT="$PROJECT_ROOT/src/05_build_pytorch_dataset/check_dataset.py"

MANIFEST_TYPE="${MANIFEST_TYPE:-noisy}"

echo "========================================"
echo "Dataset Check"
echo "Project root:  $PROJECT_ROOT"
echo "Manifest type: $MANIFEST_TYPE"
echo "========================================"

"$PYTHON_BIN" "$RUN_SCRIPT" \
  --train_dir "$TRAIN_DIR" \
  --dev_dir "$DEV_DIR" \
  --test_dir "$TEST_DIR" \
  --manifest_type "$MANIFEST_TYPE"