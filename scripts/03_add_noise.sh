#!/usr/bin/env bash
# scripts/03_add_noise.sh
# Run from repo root:
#   ./scripts/03_add_noise.sh
#
# This runs Step 3 (add MUSAN noise) for train/dev/test using the Step 2 outputs.

set -euo pipefail

# -------- paths (edit if your folders differ) --------
MUSAN_ROOT="data/raw/musan"
NOISE_SCRIPT="src/03_add_noise/add_musan_noise.py"
OUT_BASE="data/generated"

# -------- seed + params --------
SEED=1337

# Noise-type weights
P_NOISE=0.5
P_MUSIC=0.3
P_BABBLE=0.2

# SNR buckets
SNR_BUCKETS=(-5 0 5 10 20)

# Babble controls
BABBLE_DIVIDE_BY_K=true
BABBLE_K_MIN=3
BABBLE_K_MAX=8
BABBLE_CHUNK_MIN_S=0.5
BABBLE_CHUNK_MAX_S=2.0

# Clipping guard
MAX_PEAK=0.99

run_noise() {
  local split="$1"

  local gen_dir="${OUT_BASE}/${split}"
  local manifest_in="${gen_dir}/manifests/${split}_manifest.jsonl"
  local manifest_out="${gen_dir}/manifests/${split}_noisy_manifest.jsonl"
  local noisy_dir="${gen_dir}/noisy_audio"

  echo "===================================================="
  echo "Adding MUSAN noise split=${split}"
  echo "GenDir:       ${gen_dir}"
  echo "Manifest In:  ${manifest_in}"
  echo "Manifest Out: ${manifest_out}"
  echo "===================================================="

  if [[ ! -f "${manifest_in}" ]]; then
    echo "Step 2 manifest not found. Run Step 2 first: ${manifest_in}"
    exit 1
  fi

  # Optional: clean regenerate noisy outputs only
  if [[ -d "${noisy_dir}" ]]; then
    rm -rf "${noisy_dir}"
  fi

  cmd=(
    python "${NOISE_SCRIPT}"
    --split "${split}"
    --generated_dir "${gen_dir}"
    --musan_root "${MUSAN_ROOT}"
    --seed "${SEED}"
    --p_noise "${P_NOISE}"
    --p_music "${P_MUSIC}"
    --p_babble "${P_BABBLE}"
    --max_peak "${MAX_PEAK}"
    --babble_k_min "${BABBLE_K_MIN}"
    --babble_k_max "${BABBLE_K_MAX}"
    --babble_chunk_min_s "${BABBLE_CHUNK_MIN_S}"
    --babble_chunk_max_s "${BABBLE_CHUNK_MAX_S}"
    --snr_buckets "${SNR_BUCKETS[@]}"
  )

  if [[ "${BABBLE_DIVIDE_BY_K}" == "true" ]]; then
    cmd+=(--babble_divide_by_k)
  fi

  "${cmd[@]}"

  # quick check
  if [[ ! -f "${manifest_out}" ]]; then
    echo "Noisy manifest not found after noise generation: ${manifest_out}"
    exit 1
  else
    echo "[OK] Noisy manifest: ${manifest_out}"
  fi

  if [[ ! -d "${noisy_dir}" ]]; then
    echo "Noisy audio dir not found: ${noisy_dir}"
    exit 1
  else
    count=$(find "${noisy_dir}" -type f | wc -l | tr -d ' ')
    echo "[OK] Noisy audio files: ${count} in ${noisy_dir}"
  fi
}

# -------- run all splits --------
run_noise "train"
run_noise "dev"
run_noise "test"

echo
echo "All done."