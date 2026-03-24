## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/SharonHuang77/Noise-Robust-Voice-Activity-Detection-FINAL.git
cd Noise-Robust-Voice-Activity-Detection-FINAL
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
python src/01_indexing/make_indexes.py \
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
```bash
./scripts/04_extract_features.sh
```