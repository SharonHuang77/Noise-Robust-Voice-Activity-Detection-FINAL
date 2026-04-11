from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .lazy_features import (
        align_features_and_labels,
        extract_lazy_frame_features,
        load_label_array,
        load_norm_stats,
        load_waveform_array,
        normalize_frame_features,
    )
except ImportError:
    # Supports running as a direct script: python src/07_stage2_lazy_pipeline/lazy_dataset.py
    from lazy_features import (
        align_features_and_labels,
        extract_lazy_frame_features,
        load_label_array,
        load_norm_stats,
        load_waveform_array,
        normalize_frame_features,
    )


class VADLazySequenceDataset(Dataset):
    """
    Sequence-level PyTorch Dataset for Stage 2 lazy feature extraction.

    Each item corresponds to one full audio example / one manifest row.
    Features are computed on-the-fly without context stacking.
    """

    def __init__(
        self,
        generated_dir: str | Path,
        split: str,
        manifest_type: str = "noisy",
        norm_stats_path: str | Path | None = None,
        subset_fraction: float | None = None,
        subset_seed: int = 1337,
    ) -> None:
        super().__init__()

        # Validate inputs
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"split must be one of 'train', 'dev', 'test', got: {split}")
        if manifest_type not in {"clean", "noisy"}:
            raise ValueError(f"manifest_type must be 'clean' or 'noisy', got: {manifest_type}")

        self.generated_dir = Path(generated_dir).expanduser().resolve()
        self.split = split
        self.manifest_type = manifest_type
        self.norm_stats_path = Path(norm_stats_path) if norm_stats_path else None
        self.subset_fraction = subset_fraction
        self.subset_seed = subset_seed

        # Load normalization stats if provided
        if self.norm_stats_path:
            self.mean_frame, self.std_frame = load_norm_stats(str(self.norm_stats_path))
        else:
            self.mean_frame = None
            self.std_frame = None

        # Load manifest
        self.manifest_rows = self._load_manifest()
        print(f"Loaded {len(self.manifest_rows)} examples from {self.split} {self.manifest_type} manifest")

        # Apply subset sampling if requested
        if self.subset_fraction is not None:
            self._apply_subset_sampling()

    def _load_manifest(self) -> List[Dict]:
        """Load the manifest JSONL file for the given split and manifest_type."""
        manifest_path = self.generated_dir / self.split / "manifests" / f"{self.split}_manifest.jsonl"
        if self.manifest_type == "noisy":
            manifest_path = self.generated_dir / self.split / "manifests" / f"{self.split}_noisy_manifest.jsonl"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. "
                f"Verify the manifest filename convention: "
                f"for clean use '{self.split}_manifest.jsonl', "
                f"for noisy use '{self.split}_noisy_manifest.jsonl'."
            )

        rows = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        if not rows:
            raise ValueError(f"Manifest is empty: {manifest_path}")

        return rows

    def _apply_subset_sampling(self) -> None:
        """Sample a fraction of examples at the sequence level."""
        if not (0.0 < self.subset_fraction <= 1.0):
            raise ValueError(f"subset_fraction must be in (0, 1], got {self.subset_fraction}")

        total = len(self.manifest_rows)
        keep = max(1, int(total * self.subset_fraction))

        # Use deterministic sampling
        rng = np.random.RandomState(self.subset_seed)
        indices = rng.choice(total, size=keep, replace=False)

        self.manifest_rows = [self.manifest_rows[i] for i in sorted(indices)]
        print(f"Applied subset sampling: {keep}/{total} examples ({self.subset_fraction:.2%})")

    def __len__(self) -> int:
        return len(self.manifest_rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.manifest_rows[idx]

        # Validate required fields
        if "labels_path" not in row:
            raise ValueError(f"Row {row.get('ex_id', idx)} missing 'labels_path'")
        if "frame_params" not in row:
            raise ValueError(f"Row {row.get('ex_id', idx)} missing 'frame_params'")
        frame_params = row["frame_params"]
        if "frame_ms" not in frame_params:
            raise ValueError(f"Row {row.get('ex_id', idx)} missing 'frame_ms' in frame_params")
        if "hop_ms" not in frame_params:
            raise ValueError(f"Row {row.get('ex_id', idx)} missing 'hop_ms' in frame_params")
        if "sr" not in row:
            raise ValueError(f"Row {row.get('ex_id', idx)} missing 'sr'")

        # Determine audio path based on manifest_type
        if self.manifest_type == "clean":
            audio_path = row.get("clean_audio_path")
            if not audio_path:
                raise ValueError(f"Row {row.get('ex_id', idx)} missing clean_audio_path for clean manifest_type")
        else:  # noisy
            audio_path = row.get("noisy_audio_path")
            if not audio_path:
                raise ValueError(f"Row {row.get('ex_id', idx)} missing noisy_audio_path for noisy manifest_type")

        # Full paths
        audio_full_path = self.generated_dir / self.split / audio_path
        labels_full_path = self.generated_dir / self.split / row["labels_path"]

        # Load data
        waveform = load_waveform_array(str(audio_full_path))
        labels = load_label_array(str(labels_full_path))

        # Extract parameters
        frame_ms = frame_params["frame_ms"]
        hop_ms = frame_params["hop_ms"]
        sample_rate = row["sr"]

        # Compute features on the fly
        features = extract_lazy_frame_features(
            waveform=waveform,
            sample_rate=sample_rate,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
        )

        # Normalize if stats provided
        if self.mean_frame is not None and self.std_frame is not None:
            features = normalize_frame_features(features, self.mean_frame, self.std_frame)

        # Align features and labels
        features, labels = align_features_and_labels(features, labels)

        # Convert to tensors
        x = torch.from_numpy(features).float()
        y = torch.from_numpy(labels).float()

        return {
            "x": x,  # shape [T, 121]
            "y": y,  # shape [T]
            "ex_id": row["ex_id"],
            "num_frames": int(x.shape[0]),
        }


if __name__ == "__main__":
    # Optional manual check
    import argparse

    parser = argparse.ArgumentParser(description="Manual check for lazy_dataset.py")
    parser.add_argument("--generated_dir", type=str, required=True, help="Path to data/generated")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--manifest_type", type=str, default="noisy", choices=["clean", "noisy"])
    parser.add_argument("--norm_stats_path", type=str, help="Path to norm stats .npz file")
    parser.add_argument("--subset_fraction", type=float, help="Fraction of examples to use")
    args = parser.parse_args()

    print("Creating dataset...")
    dataset = VADLazySequenceDataset(
        generated_dir=args.generated_dir,
        split=args.split,
        manifest_type=args.manifest_type,
        norm_stats_path=args.norm_stats_path,
        subset_fraction=args.subset_fraction,
    )

    print(f"Dataset length: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample ex_id: {sample['ex_id']}")
        print(f"x shape: {sample['x'].shape}")
        print(f"y shape: {sample['y'].shape}")
        print(f"num_frames: {sample['num_frames']}")

    print("Manual check completed successfully!")
