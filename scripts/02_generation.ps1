# scripts/02_generation.ps1
# Run from repo root:
#   .\scripts\generate_step2_all.ps1

$ErrorActionPreference = "Stop"

# -------- paths (edit if your folders differ) --------
$LIBRISPEECH_ROOT = "data\raw\LibriSpeech"
$INDEX_DIR = "data\indexes"
$GEN_SCRIPT = "src\02_generation\generate_sequences.py"
$OUT_BASE = "data\generated"

# -------- dataset size --------
$TRAIN_N = 2000
$DEV_N   = 200
$TEST_N  = 200

# -------- seed + params (your improved Step 2 geometry) --------
$SEED = 1337

$COMMON_ARGS = @(
  "--sr", "16000",
  "--frame_ms", "25.0",
  "--hop_ms", "10.0",
  "--overlap_thr", "0.5",
  "--n_min", "2",
  "--n_max", "6",
  "--max_utt_s", "4",
  "--gap_min_s", "0.5",
  "--gap_max_s", "2.0",
  "--lead_prob", "1.0",
  "--trail_prob", "1.0",
  "--leadtrail_min_s", "0.3",
  "--leadtrail_max_s", "1.0",
  "--seed", "$SEED"
)

function Run-Gen {
  param(
    [string]$Split,
    [string]$IndexFile,
    [int]$NumExamples
  )

  $OutDir = Join-Path $OUT_BASE $Split

  Write-Host "===================================================="
  Write-Host "Generating split=$Split  num_examples=$NumExamples"
  Write-Host "Index: $IndexFile"
  Write-Host "Out:   $OutDir"
  Write-Host "===================================================="

  # Replace existing output (clean regenerate)
  if (Test-Path $OutDir) {
    Remove-Item -Recurse -Force $OutDir
  }

  python $GEN_SCRIPT `
    --split $Split `
    --librispeech_root $LIBRISPEECH_ROOT `
    --librispeech_index $IndexFile `
    --out_dir $OutDir `
    --num_examples $NumExamples `
    @COMMON_ARGS

  # quick check
  $Manifest = Join-Path $OutDir "manifests\$Split`_manifest.jsonl"
  if (!(Test-Path $Manifest)) {
    throw "Manifest not found after generation: $Manifest"
  } else {
    Write-Host "[OK] Manifest: $Manifest"
  }
}

# -------- run all splits --------
Run-Gen -Split "train" -IndexFile (Join-Path $INDEX_DIR "librispeech_train_clean_100.jsonl") -NumExamples $TRAIN_N
Run-Gen -Split "dev"   -IndexFile (Join-Path $INDEX_DIR "librispeech_dev_clean.jsonl")       -NumExamples $DEV_N
Run-Gen -Split "test"  -IndexFile (Join-Path $INDEX_DIR "librispeech_test_clean.jsonl")      -NumExamples $TEST_N

Write-Host "`nAll done."