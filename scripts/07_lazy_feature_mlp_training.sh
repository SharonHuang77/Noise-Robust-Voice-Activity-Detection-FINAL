#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 07 - Train Lazy Feature MLP
# ============================================================
# This script trains the lazy stacked-feature MLP.
# Features are computed on-the-fly from waveform + labels.
# Architecture: 1331 -> 512 -> 256 -> 1 (binary classification)

# ============================================================
# Config
# ============================================================
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRAINING_SCRIPT="$PROJECT_ROOT/src/07_lazy_feature_mlp/train_lazy_mlp.py"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/generated}"
MANIFEST_TYPE="${MANIFEST_TYPE:-noisy}"  # clean or noisy
NORM_STATS_PATH="${NORM_STATS_PATH:-}"

# Training hyperparameters
BATCH_SIZE="${BATCH_SIZE:-2048}"
EPOCHS="${EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
DROPOUT="${DROPOUT:-0.1}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_PATH="${SAVE_PATH:-$PROJECT_ROOT/artifacts/checkpoints/lazy_mlp_${MANIFEST_TYPE}.pt}"
TRAIN_FRACTION="${TRAIN_FRACTION:-1.0}"
DEV_FRACTION="${DEV_FRACTION:-1.0}"
SAVE_EVERY="${SAVE_EVERY:-1}"

# Sweep mode
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

if [[ "$MANIFEST_TYPE" != "clean" && "$MANIFEST_TYPE" != "noisy" ]]; then
  echo "ERROR: MANIFEST_TYPE must be clean or noisy, got: $MANIFEST_TYPE"
  exit 1
fi

if [[ "$MANIFEST_TYPE" == "clean" ]]; then
  TRAIN_MANIFEST="$DATA_ROOT/train/manifests/train_manifest.jsonl"
  DEV_MANIFEST="$DATA_ROOT/dev/manifests/dev_manifest.jsonl"
else
  TRAIN_MANIFEST="$DATA_ROOT/train/manifests/train_noisy_manifest.jsonl"
  DEV_MANIFEST="$DATA_ROOT/dev/manifests/dev_noisy_manifest.jsonl"
fi

if [[ ! -f "$TRAIN_MANIFEST" ]]; then
  echo "ERROR: Train manifest not found:"
  echo "  $TRAIN_MANIFEST"
  echo "Please run data generation/noise steps first."
  exit 1
fi

if [[ ! -f "$DEV_MANIFEST" ]]; then
  echo "ERROR: Dev manifest not found:"
  echo "  $DEV_MANIFEST"
  echo "Please run data generation/noise steps first."
  exit 1
fi

if [[ -n "$NORM_STATS_PATH" && ! -f "$NORM_STATS_PATH" ]]; then
  echo "ERROR: NORM_STATS_PATH provided but file not found:"
  echo "  $NORM_STATS_PATH"
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
  echo "  save_every=$SAVE_EVERY"
  echo "  save_path=$save_path"
  if [[ -n "$NORM_STATS_PATH" ]]; then
    echo "  norm_stats_path=$NORM_STATS_PATH"
  else
    echo "  norm_stats_path=<none>"
  fi
  echo "----------------------------------------"

  cmd=(
    "$PYTHON_BIN" "$TRAINING_SCRIPT"
    --data_root "$DATA_ROOT"
    --manifest_type "$MANIFEST_TYPE"
    --batch_size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --lr "$lr"
    --weight_decay "$wd"
    --dropout "$dr"
    --seed "$seed"
    --num_workers "$NUM_WORKERS"
    --train_fraction "$TRAIN_FRACTION"
    --dev_fraction "$DEV_FRACTION"
    --save_every "$SAVE_EVERY"
    --save_path "$save_path"
  )

  if [[ -n "$NORM_STATS_PATH" ]]; then
    cmd+=(--norm_stats_path "$NORM_STATS_PATH")
  fi

  "${cmd[@]}"
}

# ============================================================
# Summary
# ============================================================
echo "========================================"
echo "Lazy Feature MLP Training"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "Data root: $DATA_ROOT"
echo "Manifest type: $MANIFEST_TYPE"
echo "Norm stats path: ${NORM_STATS_PATH:-<none>}"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Number of workers: $NUM_WORKERS"
echo "Train fraction: $TRAIN_FRACTION"
echo "Dev fraction: $DEV_FRACTION"
echo "Save every: $SAVE_EVERY"
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
          run_name="lazy_mlp_${MANIFEST_TYPE}_lr${lr}_wd${wd}_dr${dr}_seed${seed}"
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
  echo "Done training lazy feature MLP model."
fi
