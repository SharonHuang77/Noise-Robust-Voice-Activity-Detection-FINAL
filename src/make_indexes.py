#!/usr/bin/env python3
"""
Step 1: Data pipeline indexing (LibriSpeech + MUSAN)

What this script does:
- Scans LibriSpeech split folders (train-clean-100, dev-clean, test-clean, etc.)
- Scans MUSAN categories (noise/, music/, speech/)
- Produces JSONL index files with (id, split/category, path, duration)

Usage examples:
python3 src/make_indexes.py \
  --librispeech_root data/raw/LibriSpeech \
  --musan_root data/raw/musan \
  --out_dir data/indexes \
  --ls_splits train_clean_100 dev_clean test_clean test_other \
  --strict_exist

Notes:
- LibriSpeech audio files are usually .flac. MUSAN is usually .wav.
- This script tries to compute durations using soundfile. If reading fails,
  it falls back to "dur_s": null (but still indexes the file).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Dict, Tuple

try:
    import soundfile as sf
except Exception:
    sf = None

SUPPORTED_AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"}


def read_duration_seconds(path: Path) -> Optional[float]:
    """Return duration in seconds, or None if unavailable/unreadable."""
    if sf is None:
        return None
    try:
        info = sf.info(str(path))
        if not info.frames or not info.samplerate:
            return None
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def iter_audio_files(root: Path, exts: Optional[set[str]] = None) -> Iterable[Path]:
    exts = exts or SUPPORTED_AUDIO_EXTS
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def write_jsonl(records: Iterable[Dict], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def infer_librispeech_utt_id(audio_path: Path) -> str:
    """Use filename stem like '84-121123-0001'."""
    return audio_path.stem


def infer_musan_file_id(rel_path_under_musan_root_no_ext: Path) -> str:
    """
    Create a stable ID from the relative path (no extension),
    with path separators replaced to keep it portable.
    Example: noise/free-sound/xyz -> noise__free-sound__xyz
    """
    return "__".join(rel_path_under_musan_root_no_ext.parts)


@dataclass(frozen=True)
class IndexConfig:
    librispeech_root: Path
    musan_root: Path
    out_dir: Path
    ls_splits: Tuple[str, ...]
    include_durations: bool
    strict_exist: bool


def index_librispeech(cfg: IndexConfig) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    ls_root = cfg.librispeech_root

    if cfg.strict_exist and not ls_root.exists():
        raise FileNotFoundError(f"LibriSpeech root not found: {ls_root}")

    for split in cfg.ls_splits:
        split_dir = ls_root / split
        if not split_dir.exists():
            if cfg.strict_exist:
                raise FileNotFoundError(f"LibriSpeech split not found: {split_dir}")
            print(f"[WARN] LibriSpeech split not found, skipping: {split_dir}")
            continue

        out_path = cfg.out_dir / f"librispeech_{split.replace('-', '_')}.jsonl"

        def records():
            for audio in iter_audio_files(split_dir, exts={".flac", ".wav"}):
                rel = audio.relative_to(ls_root)
                utt_id = infer_librispeech_utt_id(audio)
                dur_s = read_duration_seconds(audio) if cfg.include_durations else None
                yield {
                    "utt_id": utt_id,
                    "split": split,
                    "relpath": str(rel),
                    "dur_s": dur_s,
                }

        n = write_jsonl(records(), out_path)
        counts[split] = n
        print(f"[OK] LibriSpeech {split}: wrote {n} lines -> {out_path}")

    return counts


def index_musan(cfg: IndexConfig) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    musan_root = cfg.musan_root

    if cfg.strict_exist and not musan_root.exists():
        raise FileNotFoundError(f"MUSAN root not found: {musan_root}")

    for cat in ["noise", "music", "speech"]:
        cat_dir = musan_root / cat
        if not cat_dir.exists():
            if cfg.strict_exist:
                raise FileNotFoundError(f"MUSAN category not found: {cat_dir}")
            print(f"[WARN] MUSAN category not found, skipping: {cat_dir}")
            continue

        out_path = cfg.out_dir / f"musan_{cat}.jsonl"

        def records():
            for audio in iter_audio_files(cat_dir, exts={".wav", ".flac"}):
                rel = audio.relative_to(musan_root)
                rel_no_ext = rel.with_suffix("")
                file_id = infer_musan_file_id(rel_no_ext)
                dur_s = read_duration_seconds(audio) if cfg.include_durations else None
                yield {
                    "file_id": file_id,
                    "category": cat,
                    "relpath": str(rel),
                    "dur_s": dur_s,
                }

        n = write_jsonl(records(), out_path)
        counts[cat] = n
        print(f"[OK] MUSAN {cat}: wrote {n} lines -> {out_path}")

    return counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create portable JSONL indexes for LibriSpeech + MUSAN (no absolute paths).")
    p.add_argument("--librispeech_root", type=str, required=True, help="Path to LibriSpeech root folder")
    p.add_argument("--musan_root", type=str, required=True, help="Path to MUSAN root folder")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for index JSONL files")
    p.add_argument(
        "--ls_splits",
        nargs="+",
        default=["train-clean-100", "dev-clean", "test-clean"],
        help="LibriSpeech splits to index (space-separated)",
    )
    p.add_argument("--no_durations", action="store_true", help="Skip reading audio headers for durations (faster).")
    p.add_argument("--strict_exist", action="store_true", help="Fail if expected folders are missing (otherwise warn+skip).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = IndexConfig(
        librispeech_root=Path(args.librispeech_root).expanduser().resolve(),
        musan_root=Path(args.musan_root).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        ls_splits=tuple(args.ls_splits),
        include_durations=not args.no_durations,
        strict_exist=bool(args.strict_exist),
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Indexing LibriSpeech ===")
    ls_counts = index_librispeech(cfg)

    print("\n=== Indexing MUSAN ===")
    musan_counts = index_musan(cfg)

    print("\n=== Summary ===")
    print("LibriSpeech splits:")
    for k, v in ls_counts.items():
        print(f"  {k}: {v}")

    print("MUSAN categories:")
    for k, v in musan_counts.items():
        print(f"  {k}: {v}")

    print(f"\nIndexes written to: {cfg.out_dir}")
    if sf is None and cfg.include_durations:
        print("\n[NOTE] soundfile not available; durations will be null. Install with: pip install soundfile")


if __name__ == "__main__":
    main()