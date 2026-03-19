#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config
# -----------------------------
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FEATURE_SCRIPT="$PROJECT_ROOT/src/04_extract_features/extract_features.py"

TRAIN_DIR="$PROJECT_ROOT/data/generated/train"
DEV_DIR="$PROJECT_ROOT/data/generated/dev"
TEST_DIR="$PROJECT_ROOT/data/generated/test"

MANIFEST_TYPE="${MANIFEST_TYPE:-noisy}"   # noisy for noise-robust VAD
SAVE_STACKED="${SAVE_STACKED:-1}"         # 1 = save stacked 1331-dim features
N_MELS="${N_MELS:-40}"
CONTEXT_LEFT="${CONTEXT_LEFT:-5}"
CONTEXT_RIGHT="${CONTEXT_RIGHT:-5}"

TRAIN_STATS="$TRAIN_DIR/features/${MANIFEST_TYPE}_norm_stats.npz"

# -----------------------------
# Helpers
# -----------------------------
run_extract() {
  local split="$1"
  local generated_dir="$2"
  local extra_args=("${@:3}")

  echo "========================================"
  echo "Extracting features for split: $split"
  echo "Generated dir: $generated_dir"
  echo "Manifest type: $MANIFEST_TYPE"
  echo "========================================"

  "$PYTHON_BIN" "$FEATURE_SCRIPT" \
    --split "$split" \
    --generated_dir "$generated_dir" \
    --manifest_type "$MANIFEST_TYPE" \
    --n_mels "$N_MELS" \
    --context_left "$CONTEXT_LEFT" \
    --context_right "$CONTEXT_RIGHT" \
    "${extra_args[@]}"
}

# -----------------------------
# Train
# -----------------------------
TRAIN_ARGS=()
if [[ "$SAVE_STACKED" == "1" ]]; then
  TRAIN_ARGS+=(--save_stacked)
fi

run_extract "train" "$TRAIN_DIR" "${TRAIN_ARGS[@]}"

# -----------------------------
# Check stats exist
# -----------------------------
if [[ ! -f "$TRAIN_STATS" ]]; then
  echo "ERROR: Expected train stats file not found:"
  echo "  $TRAIN_STATS"
  exit 1
fi

echo "Using train stats:"
echo "  $TRAIN_STATS"

# -----------------------------
# Dev
# -----------------------------
DEV_ARGS=(--norm_stats_in "$TRAIN_STATS")
if [[ "$SAVE_STACKED" == "1" ]]; then
  DEV_ARGS+=(--save_stacked)
fi

run_extract "dev" "$DEV_DIR" "${DEV_ARGS[@]}"

# -----------------------------
# Test
# -----------------------------
TEST_ARGS=(--norm_stats_in "$TRAIN_STATS")
if [[ "$SAVE_STACKED" == "1" ]]; then
  TEST_ARGS+=(--save_stacked)
fi

run_extract "test" "$TEST_DIR" "${TEST_ARGS[@]}"

echo
echo "Done extracting features for train/dev/test."