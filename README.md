## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd Noise-Robust-Voice-Activity-Detection
```

### 2. Environment Setup

With Linux/Mac:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt 
```

With PC:
```Powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

### 3. Data Acquisition
Note: The datasets are not stored in this repository due to size constraints. Run the provided setup script to download and extract the necessary subsets:

```Bash
chmod +x setup_data.sh
./setup_data.sh
```

This script will download:

LibriSpeech: dev-clean, test-clean, test-other, train-clean-100

MUSAN: Full noise, music, and speech corpus

### 4. Index the Datasets
After downloading the data, you must generate the index files. These files allow the data loaders to quickly access specific audio samples during training.

Run the indexing script inside your virtual environment:

```Bash
# Ensure the output directory exists
mkdir -p data/indexes

# Generate the indexes
python src/01_indexing/make_indexes.py \
  --librispeech_root data/raw/LibriSpeech \
  --musan_root data/raw/musan \
  --out_dir data/indexes \
  --ls_splits train-clean-100 dev-clean test-clean test-other
```

### 5. Construct labeled VAD sequences

Run the generation script to create the manifest with labeled VAD sequences:

From Linux/Mac:
```bash
./scripts/02_generation.sh
```

From PC:
```Powershell
.\scripts\02_generation.ps1
```

