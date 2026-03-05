from __future__ import annotations
from pathlib import Path
import numpy as np
import librosa

def pre_emphasis(x: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    # y[t] = x[t] - alpha*x[t-1]
    if x.size == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - alpha * x[:-1]
    return y

def load_audio_standardized(
    path: str | Path,
    target_sr: int = 16000,
    do_preemph: bool = False,
    preemph_alpha: float = 0.97,
    peak: float = 0.9,
    ) -> tuple[np.ndarray, int]:
    """
    Loads audio, converts to mono, resamples to target_sr, and applies safe peak normalization.
    Returns (waveform float32, sr).
    """
    x, sr = librosa.load(str(path), sr=target_sr, mono=True)  # float32 in [-1,1] typically
    x = x.astype(np.float32, copy=False)

    # Optional pre-emphasis
    if do_preemph:
        x = pre_emphasis(x, alpha=preemph_alpha)

    # Safe peak normalization: only scale down if needed
    mx = float(np.max(np.abs(x))) if x.size else 0.0
    if mx > peak and mx > 0:
        x = x * (peak / mx)

    return x
