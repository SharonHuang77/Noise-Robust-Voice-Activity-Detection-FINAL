#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 06 - Train Offline MLP on Noisy Data
# ============================================================
# This script trains the offline stacked-feature MLP using noisy features.
# Features: 1331-dim stacked (context: 5 left + 121 center + 5 right)
# Architecture: 1331 -> 512 -> 256 -> 1 (binary classification)

# ============================================================
# Config
# ============================================================
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRAINING_SCRIPT="$PROJECT_ROOT/src/05_baseline_training/train_baseline_mlp.py"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/generated}"
MANIFEST_TYPE="noisy"

# Training hyperparameters
BATCH_SIZE="${BATCH_SIZE:-2048}"
EPOCHS="${EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
DROPOUT="${DROPOUT:-0.1}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_PATH="${SAVE_PATH:-$PROJECT_ROOT/artifacts/noisy_mlp_best.pt}"
TRAIN_FRACTION="${TRAIN_FRACTION:-1.0}"
DEV_FRACTION="${DEV_FRACTION:-1.0}"

# Sweep mode (single-command tuning runs)
RUN_SWEEP="${RUN_SWEEP:-0}"              # 1 to enable sweep mode
SAVE_DIR="${SAVE_DIR:-$PROJECT_ROOT/artifacts/checkpoints}"
LR_LIST="${LR_LIST:-0.001,0.0003}"
WD_LIST="${WD_LIST:-0,1e-5,1e-4}"
DROPOUT_LIST="${DROPOUT_LIST:-0.0,0.1}"
SEED_LIST="${SEED_LIST:-42,1337}"

# ============================================================
# Validation
# ============================================================
if [[ ! -f "$TRAINING_SCRIPT" ]]; then
  echo "ERROR: Training script not found:"
  echo "  $TRAINING_SCRIPT"
  exit 1
fi

if [[ ! -f "$DATA_ROOT/train/features/train_noisy_features_manifest.jsonl" ]]; then
  echo "ERROR: Noisy train features manifest not found:"
  echo "  $DATA_ROOT/train/features/train_noisy_features_manifest.jsonl"
  echo "Please run step 04 first using noisy manifests:"
  echo "  MANIFEST_TYPE=noisy ./scripts/04_extract_features.sh"
  exit 1
fi

if [[ ! -f "$DATA_ROOT/dev/features/dev_noisy_features_manifest.jsonl" ]]; then
  echo "ERROR: Noisy dev features manifest not found:"
  echo "  $DATA_ROOT/dev/features/dev_noisy_features_manifest.jsonl"
  echo "Please run step 04 first using noisy manifests:"
  echo "  MANIFEST_TYPE=noisy ./scripts/04_extract_features.sh"
  exit 1
fi

run_one() {
  local lr="$1"
  local wd="$2"
  local dr="$3"
  local seed="$4"
  local save_path="$5"

  mkdir -p "$(dirname "$save_path")"
  echo "----------------------------------------"
  echo "Run config"
  echo "  manifest_type=$MANIFEST_TYPE"
  echo "  lr=$lr wd=$wd dropout=$dr seed=$seed"
  echo "  train_fraction=$TRAIN_FRACTION dev_fraction=$DEV_FRACTION"
  echo "  save_path=$save_path"
  echo "----------------------------------------"

  "$PYTHON_BIN" "$TRAINING_SCRIPT" \
    --data_root "$DATA_ROOT" \
    --manifest_type "$MANIFEST_TYPE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$lr" \
    --weight_decay "$wd" \
    --dropout "$dr" \
    --seed "$seed" \
    --num_workers "$NUM_WORKERS" \
    --train_fraction "$TRAIN_FRACTION" \
    --dev_fraction "$DEV_FRACTION" \
    --save_path "$save_path"
}

# ============================================================
# Summary
# ============================================================
echo "========================================"
echo "Offline MLP Training (Noisy Features)"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "Data root: $DATA_ROOT"
echo "Manifest type: $MANIFEST_TYPE"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Number of workers: $NUM_WORKERS"
echo "Train fraction: $TRAIN_FRACTION"
echo "Dev fraction: $DEV_FRACTION"
echo "RUN_SWEEP: $RUN_SWEEP"
echo "========================================"
echo

if [[ "$RUN_SWEEP" == "1" ]]; then
  IFS=',' read -r -a LR_VALUES <<< "$LR_LIST"
  IFS=',' read -r -a WD_VALUES <<< "$WD_LIST"
  IFS=',' read -r -a DROPOUT_VALUES <<< "$DROPOUT_LIST"
  IFS=',' read -r -a SEED_VALUES <<< "$SEED_LIST"

  mkdir -p "$SAVE_DIR"

  total_runs=$(( ${#LR_VALUES[@]} * ${#WD_VALUES[@]} * ${#DROPOUT_VALUES[@]} * ${#SEED_VALUES[@]} ))
  run_idx=0

  echo "Sweep mode enabled"
  echo "Save directory: $SAVE_DIR"
  echo "Total runs: $total_runs"
  echo

  for lr in "${LR_VALUES[@]}"; do
    for wd in "${WD_VALUES[@]}"; do
      for dr in "${DROPOUT_VALUES[@]}"; do
        for seed in "${SEED_VALUES[@]}"; do
          run_idx=$((run_idx + 1))
          run_name="noisy_mlp_lr${lr}_wd${wd}_dr${dr}_seed${seed}"
          run_save_path="$SAVE_DIR/${run_name}.pt"

          echo "[$run_idx/$total_runs] Training $run_name"
          run_one "$lr" "$wd" "$dr" "$seed" "$run_save_path"
        done
      done
    done
  done

  echo
  echo "Done sweep training. Checkpoints saved under: $SAVE_DIR"
else
  echo "Single-run mode"
  echo "Learning rate: $LEARNING_RATE"
  echo "Weight decay: $WEIGHT_DECAY"
  echo "Dropout: $DROPOUT"
  echo "Seed: $SEED"
  echo "Save path: $SAVE_PATH"
  echo

  run_one "$LEARNING_RATE" "$WEIGHT_DECAY" "$DROPOUT" "$SEED" "$SAVE_PATH"

  echo
  echo "Done training noisy offline MLP model."
fi
