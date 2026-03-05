# src/vad_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from utils_audio import load_audio_standardized


@dataclass
class SeqParams:
    sr: int = 16000
    n_min: int = 2
    n_max: int = 6

    gap_min_s: float = 0.5
    gap_max_s: float = 2.0

    leadtrail_min_s: float = 0.3
    leadtrail_max_s: float = 1.0
    lead_prob: float = 1.0
    trail_prob: float = 1.0

    max_utt_s: float = 4.0  #cap per utterance length


def _zeros_s(duration_s: float, sr: int) -> np.ndarray:
    n = int(round(duration_s * sr))
    return np.zeros((max(0, n),), dtype=np.float32)


def build_clean_sequence(
    utt_paths: List[str],
    utt_ids: List[str],
    params: SeqParams,
    *,
    rng: np.random.Generator,
    # standardization knobs forwarded to load_audio_standardized
    do_preemph: bool = False,
    preemph_alpha: float = 0.97,
    peak: float = 0.9,
) -> Tuple[np.ndarray, List[Dict], List[Tuple[int, int]], List[Dict]]:
    """
    Build a clean stitched sequence with inserted silences.

    Returns:
      x_clean: float32 waveform
      utter_meta: list of {"utt_id", "start", "end"}
      speech_intervals: list of (start_sample, end_sample)
      silence_meta: list of {"start","end","type"} for debugging/analysis
    """
    if len(utt_paths) != len(utt_ids):
        raise ValueError("utt_paths and utt_ids must have the same length")

    sr = params.sr
    pieces: List[np.ndarray] = []
    utter_meta: List[Dict] = []
    speech_intervals: List[Tuple[int, int]] = []
    silence_meta: List[Dict] = []

    cursor = 0

    # Optional leading silence
    if rng.random() < params.lead_prob:
        d = float(rng.uniform(params.leadtrail_min_s, params.leadtrail_max_s))
        z = _zeros_s(d, sr)
        if z.size:
            pieces.append(z)
            silence_meta.append({"start": cursor, "end": cursor + z.size, "type": "lead"})
            cursor += z.size

    for i, (p, uid) in enumerate(zip(utt_paths, utt_ids)):
        w = load_audio_standardized(
            p,
            target_sr=sr,
            do_preemph=do_preemph,
            preemph_alpha=preemph_alpha,
            peak=peak,
        )

        # Optional cap per utterance length
        if params.max_utt_s is not None:
            max_len = int(round(params.max_utt_s * sr))
            if w.size > max_len:
                w = w[:max_len]

        start = cursor
        pieces.append(w)
        cursor += w.size
        end = cursor

        speech_intervals.append((start, end))
        utter_meta.append({"utt_id": uid, "start": start, "end": end})

        # Inter-utterance silence
        if i < len(utt_paths) - 1:
            d = float(rng.uniform(params.gap_min_s, params.gap_max_s))
            z = _zeros_s(d, sr)
            if z.size:
                pieces.append(z)
                silence_meta.append({"start": cursor, "end": cursor + z.size, "type": "gap"})
                cursor += z.size

    # Optional trailing silence
    if rng.random() < params.trail_prob:
        d = float(rng.uniform(params.leadtrail_min_s, params.leadtrail_max_s))
        z = _zeros_s(d, sr)
        if z.size:
            pieces.append(z)
            silence_meta.append({"start": cursor, "end": cursor + z.size, "type": "trail"})
            cursor += z.size

    x_clean = np.concatenate(pieces, axis=0) if pieces else np.zeros((0,), dtype=np.float32)
    return x_clean.astype(np.float32, copy=False), utter_meta, speech_intervals, silence_meta


def make_sample_mask(num_samples: int, speech_intervals: List[Tuple[int, int]]) -> np.ndarray:
    """
    Boolean sample-level mask where True indicates speech (within any interval).
    """
    m = np.zeros((num_samples,), dtype=np.bool_)
    for s, e in speech_intervals:
        s2 = max(0, min(num_samples, int(s)))
        e2 = max(0, min(num_samples, int(e)))
        if e2 > s2:
            m[s2:e2] = True
    return m


def frame_labels_from_intervals(
    num_samples: int,
    speech_intervals: List[Tuple[int, int]],
    *,
    sr: int = 16000,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    overlap_thr: float = 0.5,
) -> np.ndarray:
    """
    Frame label = 1 if >= overlap_thr fraction of frame samples overlap any speech interval.
    """
    L = int(round(frame_ms / 1000.0 * sr))
    H = int(round(hop_ms / 1000.0 * sr))

    if num_samples < L:
        return np.zeros((0,), dtype=np.uint8)

    m = make_sample_mask(num_samples, speech_intervals)
    T = (num_samples - L) // H + 1

    thr = int(np.ceil(overlap_thr * L))
    y = np.zeros((T,), dtype=np.uint8)

    for k in range(T):
        fs = k * H
        fe = fs + L
        overlap = int(np.count_nonzero(m[fs:fe]))
        y[k] = 1 if overlap >= thr else 0

    return y