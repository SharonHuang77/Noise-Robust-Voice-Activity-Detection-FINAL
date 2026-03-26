#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRAIN_DIR="$PROJECT_ROOT/data/generated/train"
DEV_DIR="$PROJECT_ROOT/data/generated/dev"
TEST_DIR="$PROJECT_ROOT/data/generated/test"

RUN_SCRIPT="$PROJECT_ROOT/src/06_train_baseline_mlp/run_training.py"
RESULTS_DIR="$PROJECT_ROOT/results"

MANIFEST_TYPE="${MANIFEST_TYPE:-noisy}"
INPUT_DIM="${INPUT_DIM:-1331}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"

BATCH_SIZE="${BATCH_SIZE:-512}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
EPOCHS="${EPOCHS:-10}"

NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"

RUN_NAME="${RUN_NAME:-baseline_mlp_noisy}"

echo "========================================"
echo "Baseline MLP Training"
echo "Project root:   $PROJECT_ROOT"
echo "Manifest type:  $MANIFEST_TYPE"
echo "Epochs:         $EPOCHS"
echo "Learning rate:  $LEARNING_RATE"
echo "Weight decay:   $WEIGHT_DECAY"
echo "Batch size:     $BATCH_SIZE"
echo "Hidden dim:     $HIDDEN_DIM"
echo "Run name:       $RUN_NAME"
echo "========================================"

"$PYTHON_BIN" "$RUN_SCRIPT" \
  --train_dir "$TRAIN_DIR" \
  --dev_dir "$DEV_DIR" \
  --test_dir "$TEST_DIR" \
  --manifest_type "$MANIFEST_TYPE" \
  --input_dim "$INPUT_DIM" \
  --hidden_dim "$HIDDEN_DIM" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --epochs "$EPOCHS" \
  --num_workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --results_dir "$RESULTS_DIR" \
  --run_name "$RUN_NAME"