from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import librosa
except Exception as e:
    raise ImportError("voice_features.py requires librosa. Install with: pip install librosa") from e


EPS = 1e-6


def compute_logmel(
    y: np.ndarray,
    sr: int,
    frame_ms: float,
    hop_ms: float,
    n_mels: int = 40,
) -> np.ndarray:
    """
    Compute log-mel filterbank features.

    Returns
    -------
    np.ndarray
        Shape (T, n_mels)
    """
    n_fft = int(round(sr * frame_ms / 1000.0))
    hop_length = int(round(sr * hop_ms / 1000.0))
    win_length = n_fft

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=2.0,
        center=False,   # important for alignment with frame labels
    )  # (n_mels, T)

    logmel = np.log(mel + EPS).T  # (T, n_mels)
    return logmel.astype(np.float32, copy=False)


def compute_log_energy(
    y: np.ndarray,
    sr: int,
    frame_ms: float,
    hop_ms: float,
) -> np.ndarray:
    """
    Compute per-frame log energy.

    Returns
    -------
    np.ndarray
        Shape (T, 1)
    """
    frame_length = int(round(sr * frame_ms / 1000.0))
    hop_length = int(round(sr * hop_ms / 1000.0))

    if y.size < frame_length:
        return np.zeros((0, 1), dtype=np.float32)

    frames = librosa.util.frame(
        y,
        frame_length=frame_length,
        hop_length=hop_length,
    ).T  # (T, frame_length)

    energy = np.sum(frames ** 2, axis=1, keepdims=True)
    log_energy = np.log(energy + EPS)

    return log_energy.astype(np.float32, copy=False)


def compute_deltas(
    logmel: np.ndarray,
    width: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute delta and delta-delta from log-mel features.

    width=5 corresponds to N=2 in the standard delta formula.

    Parameters
    ----------
    logmel : np.ndarray
        Shape (T, n_mels)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        delta: shape (T, n_mels)
        delta-delta: shape (T, n_mels)
    """
    if logmel.ndim != 2:
        raise ValueError(f"logmel must be 2D, got shape {logmel.shape}")

    if logmel.shape[0] == 0:
        zeros = np.zeros_like(logmel, dtype=np.float32)
        return zeros, zeros

    delta = librosa.feature.delta(
        logmel.T,
        order=1,
        width=width,
        mode="nearest",
    ).T

    delta2 = librosa.feature.delta(
        logmel.T,
        order=2,
        width=width,
        mode="nearest",
    ).T

    return (
        delta.astype(np.float32, copy=False),
        delta2.astype(np.float32, copy=False),
    )


def extract_frame_features(
    y: np.ndarray,
    sr: int,
    frame_ms: float,
    hop_ms: float,
    n_mels: int = 40,
) -> np.ndarray:
    """
    Extract per-frame features:
      - 40 log-mel
      - 40 delta
      - 40 delta-delta
      - 1 log-energy

    Total = 121 dims

    Returns
    -------
    np.ndarray
        Shape (T, 121)
    """
    logmel = compute_logmel(
        y=y,
        sr=sr,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        n_mels=n_mels,
    )
    delta, delta2 = compute_deltas(logmel, width=5)
    log_energy = compute_log_energy(
        y=y,
        sr=sr,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )

    T = min(len(logmel), len(delta), len(delta2), len(log_energy))
    logmel = logmel[:T]
    delta = delta[:T]
    delta2 = delta2[:T]
    log_energy = log_energy[:T]

    feats = np.concatenate([logmel, delta, delta2, log_energy], axis=1)

    if feats.shape[1] != 121:
        raise ValueError(f"Expected 121 feature dims, got {feats.shape[1]}")

    return feats.astype(np.float32, copy=False)


def stack_context(
    feats: np.ndarray,
    left: int = 5,
    right: int = 5,
) -> np.ndarray:
    """
    Context stack framewise features.

    For default left=right=5:
      input  shape: (T, 121)
      output shape: (T, 1331)

    Edge handling uses replicate padding.

    Returns
    -------
    np.ndarray
        Shape (T, (left + right + 1) * D)
    """
    if feats.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {feats.shape}")

    T, D = feats.shape
    if T == 0:
        return np.zeros((0, (left + right + 1) * D), dtype=np.float32)

    padded = np.pad(feats, ((left, right), (0, 0)), mode="edge")
    out = np.empty((T, (left + right + 1) * D), dtype=np.float32)

    window_size = left + right + 1
    for t in range(T):
        out[t] = padded[t:t + window_size].reshape(-1)

    return out