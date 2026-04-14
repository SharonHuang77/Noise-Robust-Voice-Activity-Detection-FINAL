#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 08 - Train Lazy Feature CRNN (no context stacking)
# ============================================================
# This script trains a CRNN directly on sequence features from
# lazy extraction (shape [T, 121]) without context stacking.

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRAINING_SCRIPT="$PROJECT_ROOT/src/08_crnn/train_lazy_crnn.py"
GENERATED_DIR="${GENERATED_DIR:-$PROJECT_ROOT/data/generated}"
MANIFEST_TYPE="${MANIFEST_TYPE:-noisy}"

if [[ "$MANIFEST_TYPE" == "clean" ]]; then
  DEFAULT_NORM_STATS_PATH="$GENERATED_DIR/train/features/clean_norm_stats.npz"
else
  DEFAULT_NORM_STATS_PATH="$GENERATED_DIR/train/features/noisy_norm_stats.npz"
fi
NORM_STATS_PATH="${NORM_STATS_PATH:-$DEFAULT_NORM_STATS_PATH}"

# Training hyperparameters
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
DROPOUT="${DROPOUT:-0.1}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-auto}"

# Model hyperparameters
CONV_CHANNELS="${CONV_CHANNELS:-64,128}"
CONV_KERNEL_SIZE="${CONV_KERNEL_SIZE:-5}"
RNN_HIDDEN_SIZE="${RNN_HIDDEN_SIZE:-128}"
RNN_LAYERS="${RNN_LAYERS:-1}"
RNN_BIDIRECTIONAL="${RNN_BIDIRECTIONAL:-1}"

# Optional data subsampling
TRAIN_FRACTION="${TRAIN_FRACTION:-1.0}"
DEV_FRACTION="${DEV_FRACTION:-1.0}"

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/stage2_lazy_crnn}"

RUN_SWEEP="${RUN_SWEEP:-0}"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/lazy_crnn_checkpoints}"
LR_LIST="${LR_LIST:-0.001,0.0003}"
WD_LIST="${WD_LIST:-1e-5,1e-4}"
DROPOUT_LIST="${DROPOUT_LIST:-0.1,0.2}"
SEED_LIST="${SEED_LIST:-42}"

if [[ ! -f "$TRAINING_SCRIPT" ]]; then
  echo "ERROR: Training script not found: $TRAINING_SCRIPT"
  exit 1
fi

if [[ "$MANIFEST_TYPE" == "clean" ]]; then
  TRAIN_MANIFEST="$GENERATED_DIR/train/manifests/train_manifest.jsonl"
  DEV_MANIFEST="$GENERATED_DIR/dev/manifests/dev_manifest.jsonl"
else
  TRAIN_MANIFEST="$GENERATED_DIR/train/manifests/train_noisy_manifest.jsonl"
  DEV_MANIFEST="$GENERATED_DIR/dev/manifests/dev_noisy_manifest.jsonl"
fi

if [[ ! -f "$TRAIN_MANIFEST" ]]; then
  echo "ERROR: Train manifest not found: $TRAIN_MANIFEST"
  exit 1
fi
if [[ ! -f "$DEV_MANIFEST" ]]; then
  echo "ERROR: Dev manifest not found: $DEV_MANIFEST"
  exit 1
fi

run_one() {
  local lr="$1"
  local wd="$2"
  local dr="$3"
  local seed="$4"
  local out_dir="$5"

  mkdir -p "$out_dir"
  echo "----------------------------------------"
  echo "Run config"
  echo "  manifest_type=$MANIFEST_TYPE"
  echo "  lr=$lr wd=$wd dropout=$dr seed=$seed"
  echo "  train_fraction=$TRAIN_FRACTION dev_fraction=$DEV_FRACTION"
  echo "  conv_channels=$CONV_CHANNELS conv_kernel_size=$CONV_KERNEL_SIZE"
  echo "  rnn_hidden_size=$RNN_HIDDEN_SIZE rnn_layers=$RNN_LAYERS bidirectional=$RNN_BIDIRECTIONAL"
  echo "  output_dir=$out_dir"
  echo "----------------------------------------"

  local norm_args=()
  if [[ -f "$NORM_STATS_PATH" ]]; then
    norm_args=(--norm_stats_path "$NORM_STATS_PATH")
  else
    echo "WARNING: norm stats not found at $NORM_STATS_PATH"
    echo "WARNING: continuing without normalization stats"
  fi

  "$PYTHON_BIN" "$TRAINING_SCRIPT" \
    --generated_dir "$GENERATED_DIR" \
    --manifest_type "$MANIFEST_TYPE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$lr" \
    --weight_decay "$wd" \
    --dropout "$dr" \
    --seed "$seed" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --train_subset_fraction "$TRAIN_FRACTION" \
    --dev_subset_fraction "$DEV_FRACTION" \
    --conv_channels "$CONV_CHANNELS" \
    --conv_kernel_size "$CONV_KERNEL_SIZE" \
    --rnn_hidden_size "$RNN_HIDDEN_SIZE" \
    --rnn_layers "$RNN_LAYERS" \
    --rnn_bidirectional "$RNN_BIDIRECTIONAL" \
    --output_dir "$out_dir" \
    "${norm_args[@]}"
}

echo "========================================"
echo "Lazy Feature CRNN Training"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "Generated dir: $GENERATED_DIR"
echo "Manifest type: $MANIFEST_TYPE"
echo "Norm stats path: $NORM_STATS_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Number of workers: $NUM_WORKERS"
echo "Device: $DEVICE"
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

  mkdir -p "$SWEEP_OUTPUT_ROOT"

  total_runs=$(( ${#LR_VALUES[@]} * ${#WD_VALUES[@]} * ${#DROPOUT_VALUES[@]} * ${#SEED_VALUES[@]} ))
  run_idx=0

  echo "Sweep mode enabled"
  echo "Sweep output root: $SWEEP_OUTPUT_ROOT"
  echo "Total runs: $total_runs"
  echo

  for lr in "${LR_VALUES[@]}"; do
    for wd in "${WD_VALUES[@]}"; do
      for dr in "${DROPOUT_VALUES[@]}"; do
        for seed in "${SEED_VALUES[@]}"; do
          run_idx=$((run_idx + 1))
          run_name="lazy_crnn_lr${lr}_wd${wd}_dr${dr}_seed${seed}"
          run_out_dir="$SWEEP_OUTPUT_ROOT/$run_name"

          echo "[$run_idx/$total_runs] Training $run_name"
          run_one "$lr" "$wd" "$dr" "$seed" "$run_out_dir"
        done
      done
    done
  done

  echo
  echo "Done sweep training. Outputs saved under: $SWEEP_OUTPUT_ROOT"
else
  echo "Single-run mode"
  echo "Learning rate: $LEARNING_RATE"
  echo "Weight decay: $WEIGHT_DECAY"
  echo "Dropout: $DROPOUT"
  echo "Seed: $SEED"
  echo "Output dir: $OUTPUT_DIR"
  echo

  run_one "$LEARNING_RATE" "$WEIGHT_DECAY" "$DROPOUT" "$SEED" "$OUTPUT_DIR"

  echo
  echo "Done training lazy feature CRNN model."
fi
