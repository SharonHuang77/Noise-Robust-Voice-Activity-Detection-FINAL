from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

try:
    from .lazy_dataset import VADLazySequenceDataset
    from .lazy_context import stack_sequence_context
    from .lazy_features import load_label_array
except ImportError:
    # Supports running as a direct script: python src/07_stage2_lazy_pipeline/lazy_frame_dataset.py
    from lazy_dataset import VADLazySequenceDataset
    from lazy_context import stack_sequence_context
    from lazy_features import load_label_array


class VADLazyFrameDataset(Dataset):
    """
    Frame-level dataset wrapper for lazy features with context stacking.

    Wraps VADLazySequenceDataset to provide frame-level indexing with context stacking.
    Each item is one frame after stacking context from its sequence.
    """

    def __init__(
        self,
        generated_dir: str | Path,
        split: str,
        manifest_type: str = "noisy",
        norm_stats_path: str | Path | None = None,
        subset_fraction: float | None = None,
        subset_seed: int = 1337,
        context_left: int = 5,
        context_right: int = 5,
    ) -> None:
        super().__init__()

        # Assign attributes early
        self.generated_dir = Path(generated_dir).expanduser().resolve()
        self.split = split
        self.manifest_type = manifest_type
        self.context_left = context_left
        self.context_right = context_right

        # Build the underlying sequence dataset
        self.sequence_dataset = VADLazySequenceDataset(
            generated_dir=generated_dir,
            split=split,
            manifest_type=manifest_type,
            norm_stats_path=norm_stats_path,
            subset_fraction=subset_fraction,
            subset_seed=subset_seed,
        )

        # Cache for loaded and stacked sequences
        self._sequence_cache: Dict[int, Dict] = {}

        # Precompute frame index mapping from manifest rows
        # Avoid loading features during initialization
        self.frame_mapping: List[Tuple[int, int, str]] = []
        self.total_frames = 0

        for seq_idx, row in enumerate(self.sequence_dataset.manifest_rows):
            # Determine num_frames: prefer direct value, fallback to label length
            if "num_frames" in row:
                num_frames = row["num_frames"]
            elif "labels_path" in row:
                # Fallback: load labels and use their length as frame count
                # This ensures exact match with actual feature/label alignment
                labels_full_path = self.generated_dir / self.split / row["labels_path"]
                labels = load_label_array(str(labels_full_path))
                num_frames = len(labels)
            else:
                raise ValueError(f"Manifest row {row.get('ex_id', seq_idx)} missing both 'num_frames' and 'labels_path'")
            
            ex_id = row["ex_id"]

            for frame_idx in range(num_frames):
                self.frame_mapping.append((seq_idx, frame_idx, ex_id))

            self.total_frames += num_frames

        print(f"Frame dataset: {self.total_frames} total frames from {len(self.sequence_dataset)} sequences")

    def _get_cached_sequence(self, seq_idx: int) -> Dict:
        """Load and cache the context-stacked sequence if not already cached."""
        if seq_idx not in self._sequence_cache:
            # Load the sequence
            seq_sample = self.sequence_dataset[seq_idx]
            x_seq = seq_sample["x"]  # [T, 121]
            y_seq = seq_sample["y"]  # [T]
            ex_id = seq_sample["ex_id"]

            # Stack context once
            x_stacked_seq = stack_sequence_context(
                x_seq.numpy(),
                context_left=self.context_left,
                context_right=self.context_right,
            )  # [T, 1331]

            # Cache the result
            self._sequence_cache[seq_idx] = {
                "x_stacked_seq": x_stacked_seq,
                "y_seq": y_seq,
                "ex_id": ex_id,
            }

        return self._sequence_cache[seq_idx]

    def __len__(self) -> int:
        return self.total_frames

    def __getitem__(self, idx: int) -> Dict:
        seq_idx, frame_idx, ex_id = self.frame_mapping[idx]

        # Get cached sequence data
        cached = self._get_cached_sequence(seq_idx)
        x_stacked_seq = cached["x_stacked_seq"]
        y_seq = cached["y_seq"]

        # Safety check against manifest/frame-mapping mismatch
        if frame_idx >= len(y_seq):
            raise IndexError(f"frame_idx {frame_idx} >= len(y_seq) {len(y_seq)} for ex_id {ex_id}")

        # Extract the specific frame
        x_frame = x_stacked_seq[frame_idx]  # [1331]
        y_frame = y_seq[frame_idx]  # scalar

        return {
            "x": torch.from_numpy(x_frame).float(),
            "y": y_frame,
            "ex_id": cached["ex_id"],  # Use from cached for consistency
            "frame_idx": frame_idx,
        }


if __name__ == "__main__":
    # Tiny sanity check
    import argparse

    parser = argparse.ArgumentParser(description="Manual check for lazy_frame_dataset.py")
    parser.add_argument("--generated_dir", type=str, required=True, help="Path to data/generated")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--manifest_type", type=str, default="noisy", choices=["clean", "noisy"])
    parser.add_argument("--norm_stats_path", type=str, help="Path to norm stats .npz file")
    parser.add_argument("--subset_fraction", type=float, help="Fraction of examples to use")
    args = parser.parse_args()

    print("Creating frame dataset...")
    dataset = VADLazyFrameDataset(
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
        print(f"frame_idx: {sample['frame_idx']}")

        # Check shapes
        assert sample['x'].shape == (1331,), f"Expected x shape (1331,), got {sample['x'].shape}"
        assert sample['y'].dim() == 0, f"Expected scalar y, got shape {sample['y'].shape}"
        print("✓ Shapes correct")

    print("Manual check completed successfully!")