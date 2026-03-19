from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


def read_jsonl(path: Path) -> List[Dict]:
    """
    Read a JSONL file into a list of dicts.
    """
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[Dict], path: Path) -> None:
    """
    Write a list of dicts to a JSONL file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_manifest_path(
    generated_dir: Path,
    split: str,
    manifest_type: str,
) -> Tuple[Path, str]:
    """
    Resolve manifest path and audio field key.

    Parameters
    ----------
    generated_dir : Path
        e.g. data/generated/train
    split : str
        train / dev / test
    manifest_type : str
        clean / noisy

    Returns
    -------
    Tuple[Path, str]
        manifest_path, audio_key
    """
    if manifest_type == "noisy":
        manifest_path = generated_dir / "manifests" / f"{split}_noisy_manifest.jsonl"
        audio_key = "noisy_audio_path"
    elif manifest_type == "clean":
        manifest_path = generated_dir / "manifests" / f"{split}_manifest.jsonl"
        audio_key = "clean_audio_path"
    else:
        raise ValueError(f"manifest_type must be 'clean' or 'noisy', got {manifest_type}")

    return manifest_path, audio_key


def ensure_feature_dirs(
    generated_dir: Path,
    manifest_type: str,
    save_stacked: bool,
) -> Tuple[Path, Path]:
    """
    Create output feature directories.

    Returns
    -------
    Tuple[Path, Path]
        frame_dir, stacked_dir
    """
    feat_root = generated_dir / "features"
    frame_dir = feat_root / f"{manifest_type}_frame_121"
    stacked_dir = feat_root / f"{manifest_type}_stacked_1331"

    frame_dir.mkdir(parents=True, exist_ok=True)
    if save_stacked:
        stacked_dir.mkdir(parents=True, exist_ok=True)

    return frame_dir, stacked_dir


def get_stats_path(generated_dir: Path, manifest_type: str) -> Path:
    """
    Return normalization stats path.
    """
    return generated_dir / "features" / f"{manifest_type}_norm_stats.npz"