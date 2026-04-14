from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

# Robust import for voice_features.py
import sys
from pathlib import Path
current_dir = Path(__file__).parent
extract_features_dir = current_dir.parent / "04_extract_features"
if str(extract_features_dir) not in sys.path:
    sys.path.insert(0, str(extract_features_dir))
from voice_features import extract_frame_features


def load_waveform_array(path: str) -> np.ndarray:
    """
    Load waveform from a .npy file and return float32 numpy array.

    Parameters
    ----------
    path : str
        Path to the .npy file containing the waveform.

    Returns
    -------
    np.ndarray
        Float32 numpy array of shape (num_samples,).
    """
    waveform = np.load(path)
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32, copy=False)
    return waveform


def load_label_array(path: str) -> np.ndarray:
    """
    Load frame labels from a .npy file and return float32 or int64 numpy array.

    Parameters
    ----------
    path : str
        Path to the .npy file containing the labels.

    Returns
    -------
    np.ndarray
        Numpy array of shape (num_frames,) with labels, converted to float32.
    """
    labels = np.load(path)
    return labels.astype(np.float32, copy=False)


def load_norm_stats(stats_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mean_frame and std_frame from the saved .npz normalization stats file.
    These are frame-level normalization statistics from Stage 1.

    Parameters
    ----------
    stats_path : str
        Path to the .npz file containing 'mean_frame' and 'std_frame'.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        mean_frame: shape (121,), std_frame: shape (121,)
    """
    data = np.load(stats_path)
    mean_frame = data['mean_frame']
    std_frame = data['std_frame']
    return mean_frame, std_frame


def extract_lazy_frame_features(
    waveform: np.ndarray,
    sample_rate: int,
    frame_ms: float,
    hop_ms: float,
    n_mels: int = 40
) -> np.ndarray:
    """
    Compute frame-level features on the fly using the same Stage 1 logic as much as possible.
    The output shape must be [T, 121].
    Use Stage 1 voice feature functions if possible.

    Parameters
    ----------
    waveform : np.ndarray
        Audio waveform, shape (num_samples,).
    sample_rate : int
        Sample rate in Hz.
    frame_ms : float
        Frame length in milliseconds.
    hop_ms : float
        Hop length in milliseconds.
    n_mels : int, optional
        Number of mel bins, by default 40.

    Returns
    -------
    np.ndarray
        Frame features, shape (T, 121).
    """
    return extract_frame_features(
        y=waveform,
        sr=sample_rate,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        n_mels=n_mels,
    )


def normalize_frame_features(
    features: np.ndarray,
    mean_frame: np.ndarray,
    std_frame: np.ndarray
) -> np.ndarray:
    """
    Apply frame-level normalization safely.

    Parameters
    ----------
    features : np.ndarray
        Frame features, shape (T, 121).
    mean_frame : np.ndarray
        Mean per feature dimension, shape (121,).
    std_frame : np.ndarray
        Std per feature dimension, shape (121,).

    Returns
    -------
    np.ndarray
        Normalized features, shape (T, 121).
    """
    safe_std = np.where(std_frame < 1e-8, 1.0, std_frame)
    return (features - mean_frame) / safe_std


def align_features_and_labels(
    features: np.ndarray,
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Safely align feature frames and labels by truncating both to the minimum length.
    This is important because slight length mismatches may happen.
    Do not raise an error for small mismatches.

    Parameters
    ----------
    features : np.ndarray
        Frame features, shape (T, 121).
    labels : np.ndarray
        Frame labels, shape (T',).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Aligned features and labels, both shape (min(T, T'), ...).
    """
    min_len = min(len(features), len(labels))
    return features[:min_len], labels[:min_len]


if __name__ == "__main__":
    # Optional manual check
    import argparse

    parser = argparse.ArgumentParser(description="Manual check for lazy_features.py")
    parser.add_argument("--waveform_path", type=str, help="Path to waveform .npy file")
    parser.add_argument("--label_path", type=str, help="Path to label .npy file")
    parser.add_argument("--stats_path", type=str, help="Path to norm stats .npz file")
    args = parser.parse_args()

    if args.waveform_path and args.label_path and args.stats_path:
        print("Loading waveform...")
        waveform = load_waveform_array(args.waveform_path)
        print(f"Waveform shape: {waveform.shape}, dtype: {waveform.dtype}")

        print("Loading labels...")
        labels = load_label_array(args.label_path)
        print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")

        print("Loading norm stats...")
        mean_frame, std_frame = load_norm_stats(args.stats_path)
        print(f"Mean shape: {mean_frame.shape}, Std shape: {std_frame.shape}")

        print("Extracting features...")
        features = extract_lazy_frame_features(
            waveform=waveform,
            sample_rate=16000,
            frame_ms=25.0,
            hop_ms=10.0,
        )
        print(f"Features shape: {features.shape}")

        print("Normalizing features...")
        norm_features = normalize_frame_features(features, mean_frame, std_frame)
        print(f"Normalized features shape: {norm_features.shape}")

        print("Aligning features and labels...")
        aligned_features, aligned_labels = align_features_and_labels(norm_features, labels)
        print(f"Aligned features shape: {aligned_features.shape}, labels shape: {aligned_labels.shape}")

        print("Manual check completed successfully!")
    else:
        print("Provide --waveform_path, --label_path, and --stats_path for manual check.")