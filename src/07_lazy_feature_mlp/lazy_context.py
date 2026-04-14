from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

# Robust import for lazy_features if needed, but not used here
# This file is standalone for context stacking


def stack_sequence_context(
    features: np.ndarray,
    context_left: int = 5,
    context_right: int = 5
) -> np.ndarray:
    """
    Stack context frames for sequence-level features.

    Input: frame-level features of shape [T, D] where D=121
    Output: context-stacked features of shape [T, (context_left + context_right + 1) * D]

    Uses replicate padding for edge frames (repeat nearest valid frame).

    Parameters
    ----------
    features : np.ndarray
        Frame features, shape [T, D]
    context_left : int, optional
        Number of left context frames, by default 5
    context_right : int, optional
        Number of right context frames, by default 5

    Returns
    -------
    np.ndarray
        Context-stacked features, shape [T, (11) * D] for default context
    """
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features [T, D], got shape {features.shape}")

    T, D = features.shape
    total_context = context_left + context_right + 1
    output_dim = total_context * D

    # Initialize output array
    stacked = np.zeros((T, output_dim), dtype=features.dtype)

    for t in range(T):
        # Collect context frames with padding
        context_frames = []
        for offset in range(-context_left, context_right + 1):
            idx = max(0, min(T - 1, t + offset))  # Clamp to valid indices
            context_frames.append(features[idx])

        # Concatenate and store
        stacked[t] = np.concatenate(context_frames)

    return stacked


if __name__ == "__main__":
    # Tiny shape check
    # Create dummy features: 10 frames, 121 dims
    dummy_features = np.random.randn(10, 121).astype(np.float32)

    stacked = stack_sequence_context(dummy_features, context_left=5, context_right=5)

    print(f"Input shape: {dummy_features.shape}")
    print(f"Output shape: {stacked.shape}")
    print(f"Expected output shape: (10, 1331)")
    assert stacked.shape == (10, 1331), f"Shape mismatch: {stacked.shape}"
    print("✓ Shape check passed!")