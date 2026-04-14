#!/usr/bin/env bash
set -euo pipefail

# scripts/02_generation.sh
# Run from repo root:
#   bash scripts/02_generation.sh
# or:
#   chmod +x scripts/02_generation.sh && ./scripts/02_generation.sh

# -------- paths (edit if your folders differ) --------
LIBRISPEECH_ROOT="data/raw/LibriSpeech"
INDEX_DIR="data/indexes"
GEN_SCRIPT="src/02_generation/generate_sequences.py"
OUT_BASE="data/generated"

# -------- dataset size --------
TRAIN_N=3000
DEV_N=500
TEST_N=500

# -------- seed + params (your improved Step 2 geometry) --------
SEED=1337

COMMON_ARGS=(
  --sr 16000
  --frame_ms 25.0
  --hop_ms 10.0
  --overlap_thr 0.5
  --n_min 3
  --n_max 6
  --max_utt_s 3.5
  --gap_min_s 0.5
  --gap_max_s 2.0
  --lead_prob 0.8
  --trail_prob 0.8
  --leadtrail_min_s 0.3
  --leadtrail_max_s 2.5
  --seed "$SEED"
)

run_gen () {
  local split="$1"
  local index_file="$2"
  local num_examples="$3"

  local out_dir="${OUT_BASE}/${split}"

  echo "===================================================="
  echo "Generating split=${split}  num_examples=${num_examples}"
  echo "Index: ${index_file}"
  echo "Out:   ${out_dir}"
  echo "===================================================="

  # Replace existing output (clean regenerate)
  rm -rf "${out_dir}"

  python3 "${GEN_SCRIPT}" \
    --split "${split}" \
    --librispeech_root "${LIBRISPEECH_ROOT}" \
    --librispeech_index "${index_file}" \
    --out_dir "${out_dir}" \
    --num_examples "${num_examples}" \
    "${COMMON_ARGS[@]}"

  # quick check
  local manifest="${out_dir}/manifests/${split}_manifest.jsonl"
  if [[ ! -f "${manifest}" ]]; then
    echo "ERROR: Manifest not found after generation: ${manifest}" >&2
    exit 1
  else
    echo "[OK] Manifest: ${manifest}"
  fi
}

# -------- run all splits --------
run_gen "train" "${INDEX_DIR}/librispeech_train_clean_100.jsonl" "${TRAIN_N}"
run_gen "dev"   "${INDEX_DIR}/librispeech_dev_clean.jsonl"       "${DEV_N}"
run_gen "test"  "${INDEX_DIR}/librispeech_test_clean.jsonl"      "${TEST_N}"

echo
echo "All done."