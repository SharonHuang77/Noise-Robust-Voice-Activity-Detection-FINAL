from __future__ import annotations

from typing import List, Tuple

import numpy as np


def compute_mean_std_from_files(feature_paths: Iterable[Path]) -> Tuple[np.ndarray, np.ndarray]:
    sum_x = None
    sum_x2 = None
    total_count = 0

    for path in feature_paths:
        feats = np.load(path).astype(np.float64, copy=False)  # use float64 for stable accumulation
        if feats.ndim != 2:
            raise ValueError(f"Expected 2D features, got {feats.shape} from {path}")

        if sum_x is None:
            dim = feats.shape[1]
            sum_x = np.zeros(dim, dtype=np.float64)
            sum_x2 = np.zeros(dim, dtype=np.float64)

        sum_x += feats.sum(axis=0)
        sum_x2 += np.square(feats).sum(axis=0)
        total_count += feats.shape[0]

    if total_count == 0:
        raise ValueError("No feature frames found while computing stats")

    mean = sum_x / total_count
    var = (sum_x2 / total_count) - np.square(mean)
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var)

    return mean.astype(np.float32), std.astype(np.float32)


def apply_normalization(
    feats: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return ((feats - mean) / std).astype(np.float32, copy=False)