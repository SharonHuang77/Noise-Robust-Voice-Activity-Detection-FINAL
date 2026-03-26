## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd Noise-Robust-Voice-Activity-Detection
```

### 2. Data Acquisition
Note: The datasets are not stored in this repository due to size constraints. Run the provided setup script to download and extract the necessary subsets:

This step requires a Unix-like environment (macOS Terminal or Ubuntu/WSL on Windows).

1. Ensure your script has Linux line endings: If you are on Windows, ensure setup_data.sh is saved with LF line endings (check the bottom right of VS Code) to avoid "file not found" errors.

Troubleshooting: If you get a `-bash: ./setup_data.sh: cannot execute: required file not found error` on Windows/WSL, run `sed -i 's/\r$//' setup_data.sh` to fix Windows line endings.

2. Run the setup script:
```Bash
# Navigate to the project root
chmod +x setup_data.sh
./setup_data.sh
```

This script will download:

- LibriSpeech: dev-clean, test-clean, test-other, train-clean-100

- MUSAN: Full noise, music, and speech corpus

### 3. Build the Docker Image
```bash
docker build -t vad-project .
```
### 4 Run the Container
```bash
# Mac/Linux (Terminal)
docker run -it --rm -v $(pwd):/app vad-project
```

### 5. Run the Data Pipeline

Before running the shell scripts, make sure they are executable:

```bash
chmod +x scripts/*.sh
```

Step 5.1: Create Index Files
Scans LibriSpeech and MUSAN directories to create index files that list all audio files and their metadata.

```bash
python3 src/01_indexing/make_indexes.py \
  --librispeech_root data/raw/LibriSpeech \
  --musan_root data/raw/musan \
  --out_dir data/indexes \
  --ls_splits train-clean-100 dev-clean test-clean test-other
```

Step 5.2: Generate Clean Speech Segments
Extracts clean speech segments and prepares clean training examples.
```bash
./scripts/02_generation.sh
```

Step 5.3: Add Noise 
Mixes clean speech segments with noise from MUSAN at various SNR levels to create noisy training examples.
```bash
./scripts/03_add_noise.sh
```

### 6. Extract Features
Extracts log-mel + delta + delta-delta + log-energy features from the generated audio files for model training and evaluation.

By default, this extracts **noisy** features:
```bash
./scripts/04_extract_features.sh
```

For baseline training, you also need **clean** features:
```bash
MANIFEST_TYPE=clean ./scripts/04_extract_features.sh
```

This will create `train_clean_features_manifest.jsonl` (plus dev/test versions) needed for Step 7.

### 7. Train Baseline MLP
Trains a baseline MLP model for frame-level voice activity detection on clean LibriSpeech speech.
- Architecture: 1331 → 512 → 256 → 1 (binary classification)
- Features: 1331-dim stacked (5 left + 121 center + 5 right frames)
- Default: 5 epochs, batch size 2048, learning rate 0.001

```bash
RUN_SWEEP=1 ./scripts/05_baseline_training.sh
```

Optional custom sweep in one command:
```bash
RUN_SWEEP=1 LR_LIST=0.001,0.0003 WD_LIST=0,1e-5,1e-4 DROPOUT_LIST=0.0,0.1 SEED_LIST=42,1337 EPOCHS=5 BATCH_SIZE=2048 ./scripts/05_baseline_training.sh
```

Faster tuning (use only a fraction of train/dev frames):
```bash
RUN_SWEEP=1 TRAIN_FRACTION=0.20 DEV_FRACTION=0.30 EPOCHS=2 SEED_LIST=42 ./scripts/05_baseline_training.sh
```

Single-run with reduced data:
```bash
TRAIN_FRACTION=0.25 DEV_FRACTION=0.25 ./scripts/05_baseline_training.sh
```

Notes:
- `TRAIN_FRACTION` and `DEV_FRACTION` must be in `(0, 1]`.
- Use reduced fractions for quick search, then retrain top settings with `TRAIN_FRACTION=1.0 DEV_FRACTION=1.0`.