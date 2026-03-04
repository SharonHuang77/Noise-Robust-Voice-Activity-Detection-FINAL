#!/bin/bash

# Create data directory structure
mkdir -p data/raw
cd data/raw

# List of LibriSpeech subsets from SLR12
LIBRISPEECH_FILES=("dev-clean.tar.gz" "test-clean.tar.gz" "test-other.tar.gz" "train-clean-100.tar.gz")

echo "--- Downloading LibriSpeech Subsets ---"
for file in "${LIBRISPEECH_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        curl -O "https://www.openslr.org/resources/12/$file"
        tar -xvzf "$file"
    else
        echo "$file already exists, skipping."
    fi
done

echo "--- Downloading MUSAN (SLR17) ---"
if [ ! -f "musan.tar.gz" ]; then
    curl -O "https://www.openslr.org/resources/17/musan.tar.gz"
    tar -xvzf "musan.tar.gz"
else
    echo "musan.tar.gz already exists, skipping."
fi

echo "--- Cleanup ---"
# Optional: Remove the .tar.gz files to save space
rm *.tar.gz

echo "Data setup complete."