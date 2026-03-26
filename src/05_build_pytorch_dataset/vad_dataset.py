from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class VADStackedFrameDataset(Dataset):
    """
    PyTorch Dataset for offline stacked VAD features.

    Each item corresponds to one frame after context stacking:
        X: shape [1331]
        y: scalar label (0 or 1)

    This dataset reads a feature manifest and builds a global frame index
    over all examples in the split.
    """

    def __init__(
        self,
        generated_dir: str,
        split: str,
        manifest_type: str = "noisy",
    ) -> None:
        """
        Parameters
        ----------
        generated_dir : str
            Path to the generated split directory, for example:
            data/generated/train
        split : str
            Dataset split name: train / dev / test
        manifest_type : str
            Feature type: clean / noisy
        """
        self.generated_dir = Path(generated_dir)
        self.split = split
        self.manifest_type = manifest_type

        self.feature_manifest_path = (
            self.generated_dir
            / "features"
            / f"{split}_{manifest_type}_features_manifest.jsonl"
        )

        if not self.feature_manifest_path.exists():
            raise FileNotFoundError(
                f"Feature manifest not found: {self.feature_manifest_path}"
            )

        self.rows = self._load_manifest(self.feature_manifest_path)
        if len(self.rows) == 0:
            raise ValueError(f"Empty feature manifest: {self.feature_manifest_path}")

        # index_map[k] = (row_idx, frame_idx)
        # This lets us treat all frames across all examples as one dataset.
        self.index_map: List[Tuple[int, int]] = []
        self._build_index_map()

        print(f"[{self.split}] Loaded {len(self.rows)} examples")
        print(f"[{self.split}] Total frames: {len(self.index_map)}")

    @staticmethod
    def _load_manifest(path: Path) -> List[dict]:
        """Load JSONL feature manifest."""
        rows: List[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _build_index_map(self) -> None:
        """
        Build a global frame index across all examples.

        For each example, read only the array shape using mmap_mode='r',
        then append one index entry per frame.
        """
        for row_idx, row in enumerate(self.rows):
            stacked_path = self.generated_dir / row["stacked_features_path"]
            label_path = self.generated_dir / row["stacked_labels_path"]

            if not stacked_path.exists():
                raise FileNotFoundError(
                    f"Missing stacked feature file: {stacked_path}"
                )
            if not label_path.exists():
                raise FileNotFoundError(
                    f"Missing stacked label file: {label_path}"
                )

            X = np.load(stacked_path, mmap_mode="r")
            y = np.load(label_path, mmap_mode="r")

            if X.ndim != 2 or X.shape[1] != 1331:
                raise ValueError(
                    f"Expected stacked features of shape (T, 1331), "
                    f"got {X.shape} from {stacked_path}"
                )

            if y.ndim != 1:
                raise ValueError(
                    f"Expected labels of shape (T,), got {y.shape} from {label_path}"
                )

            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"Feature/label length mismatch in {stacked_path}: "
                    f"{X.shape[0]} vs {y.shape[0]}"
                )

            for frame_idx in range(X.shape[0]):
                self.index_map.append((row_idx, frame_idx))

    def __len__(self) -> int:
        """Return total number of frames in the split."""
        return len(self.index_map)

    def __getitem__(self, idx: int):
        """
        Return one stacked frame and its label.

        Returns
        -------
        x_frame : torch.FloatTensor
            Shape [1331]
        y_frame : torch.FloatTensor
            Scalar tensor, value 0.0 or 1.0
        """
        row_idx, frame_idx = self.index_map[idx]
        row = self.rows[row_idx]

        stacked_path = self.generated_dir / row["stacked_features_path"]
        label_path = self.generated_dir / row["stacked_labels_path"]

        X = np.load(stacked_path, mmap_mode="r")
        y = np.load(label_path, mmap_mode="r")

        x_frame = np.array(X[frame_idx], dtype=np.float32, copy=True)
        y_frame = np.float32(y[frame_idx])

        return torch.from_numpy(x_frame), torch.tensor(y_frame, dtype=torch.float32)