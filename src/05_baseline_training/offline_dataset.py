from __future__ import annotations

import argparse
import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class FrameFileRow:
    ex_id: str
    split: str
    x_path: Path
    y_path: Path
    num_frames: int


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class OfflineFrameDataset(Dataset):
    """
    Clean-only frame-level dataset from offline stacked features.

    Each item is one frame after context stacking:
      - X: shape (1331,)
      - y: scalar in {0, 1}
    """

    def __init__(
        self,
        generated_dir: str | Path,
        split: str,
        expected_dim: int = 1331,
    ) -> None:
        super().__init__()

        if split not in {"train", "dev", "test"}:
            raise ValueError(f"split must be train/dev/test, got: {split}")

        self.generated_dir = Path(generated_dir).expanduser().resolve()
        self.split = split
        self.expected_dim = expected_dim
        self.manifest_type = "clean"

        self.feature_manifest = (
            self.generated_dir / "features" / f"{self.split}_clean_features_manifest.jsonl"
        )
        if not self.feature_manifest.exists():
            raise FileNotFoundError(
                f"Missing clean feature manifest: {self.feature_manifest}. "
                "Extract clean stacked features first."
            )

        raw_rows = read_jsonl(self.feature_manifest)
        if not raw_rows:
            raise ValueError(f"Feature manifest is empty: {self.feature_manifest}")

        self.rows: List[FrameFileRow] = []
        self.cumulative_frames: List[int] = []
        total = 0

        for row in raw_rows:
            ex_id = str(row["ex_id"])
            x_rel = row.get("stacked_features_path")
            y_rel = row.get("stacked_labels_path")
            if x_rel is None or y_rel is None:
                raise ValueError(
                    f"Manifest row {ex_id} missing stacked paths. "
                    "Run extraction with save_stacked enabled."
                )

            x_path = self.generated_dir / x_rel
            y_path = self.generated_dir / y_rel
            if not x_path.exists():
                raise FileNotFoundError(f"Missing feature file for {ex_id}: {x_path}")
            if not y_path.exists():
                raise FileNotFoundError(f"Missing label file for {ex_id}: {y_path}")

            X = np.load(x_path, mmap_mode="r")
            y = np.load(y_path, mmap_mode="r")
            if X.ndim != 2:
                raise ValueError(f"{ex_id}: X must be 2D, got shape {X.shape}")
            if y.ndim != 1:
                raise ValueError(f"{ex_id}: y must be 1D, got shape {y.shape}")
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"{ex_id}: frame mismatch X.shape[0]={X.shape[0]} y.shape[0]={y.shape[0]}"
                )
            if X.shape[1] != self.expected_dim:
                raise ValueError(
                    f"{ex_id}: expected dim {self.expected_dim}, got {X.shape[1]}"
                )

            uniq = np.unique(y)
            if not np.all(np.isin(uniq, [0, 1])):
                raise ValueError(f"{ex_id}: labels must be binary, got {uniq.tolist()}")

            num_frames = int(X.shape[0])
            self.rows.append(
                FrameFileRow(
                    ex_id=ex_id,
                    split=self.split,
                    x_path=x_path,
                    y_path=y_path,
                    num_frames=num_frames,
                )
            )

            total += num_frames
            self.cumulative_frames.append(total)

        self.total_frames = total

        self._cache_file_idx: Optional[int] = None
        self._cache_X: Optional[np.ndarray] = None
        self._cache_y: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return self.total_frames

    def _resolve_index(self, idx: int) -> Tuple[int, int]:
        if idx < 0:
            idx = self.total_frames + idx
        if idx < 0 or idx >= self.total_frames:
            raise IndexError(f"Index out of range: {idx} for dataset size {self.total_frames}")

        file_idx = bisect.bisect_right(self.cumulative_frames, idx)
        prev = 0 if file_idx == 0 else self.cumulative_frames[file_idx - 1]
        local_idx = idx - prev
        return file_idx, local_idx

    def _load_file(self, file_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._cache_file_idx == file_idx and self._cache_X is not None and self._cache_y is not None:
            return self._cache_X, self._cache_y

        row = self.rows[file_idx]
        X = np.load(row.x_path, mmap_mode="r")
        y = np.load(row.y_path, mmap_mode="r")

        self._cache_file_idx = file_idx
        self._cache_X = X
        self._cache_y = y
        return X, y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx, local_idx = self._resolve_index(idx)
        X_file, y_file = self._load_file(file_idx)

        # Memmap slices can be read-only; copy to ensure writable backing memory for torch.
        x = np.asarray(X_file[local_idx], dtype=np.float32).copy()
        y = np.asarray(y_file[local_idx], dtype=np.float32)

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def build_dataloader(
    generated_dir: str | Path,
    split: str,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    dataset = OfflineFrameDataset(
        generated_dir=generated_dir,
        split=split,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick sanity check for clean OfflineFrameDataset")
    parser.add_argument("--generated_dir", required=True, type=str)
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--batch_size", default=64, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = build_dataloader(
        generated_dir=args.generated_dir,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
    )

    X, y = next(iter(loader))
    print(f"dataset size (frames): {len(loader.dataset)}")
    print(f"batch X shape: {tuple(X.shape)}")
    print(f"batch y shape: {tuple(y.shape)}")
    print(f"X dtype: {X.dtype} | y dtype: {y.dtype}")


if __name__ == "__main__":
    main()
