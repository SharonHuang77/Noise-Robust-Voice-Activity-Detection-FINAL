#!/usr/bin/env python3
# src/2. Construct labeled VAD/generate_sequences.py
#!/usr/bin/env python3
"""
Step 2: Construct labeled clean VAD sequences from LibriSpeech.

Inputs:
- LibriSpeech index JSONL created in Step 1 (must include 'relpath').

Outputs:
- data/generated/<split>/clean_audio/<ex_id>.npy    (float32 waveform @ 16kHz)
- data/generated/<split>/labels/<ex_id>_y.npy       (uint8 frame labels 0/1)
- data/generated/<split>/manifests/<split>_manifest.jsonl

Key idea:
- We create speech/non-speech transitions by inserting silences between utterances.
- Labels are derived from known boundaries (NOT energy-based).

Example usage with Mac/Linux:
  python3 src/02_generation/generate_sequences.py \
    --split train \
    --librispeech_root data/raw/LibriSpeech \
    --librispeech_index data/indexes/librispeech_train_clean_100.jsonl \
    --out_dir data/generated/train \
    --num_examples 2000 \
    --seed 1337

Exmaple usage with Windows PowerShell:
    python src\02_generation\generate_sequences.py `
    --split train `
    --librispeech_root data/raw/LibriSpeech `
    --librispeech_index data/indexes/librispeech_train_clean_100.jsonl `
    --out_dir data/generated/train `
    --num_examples 2000 `
    --seed 1337

For dev:
Example usage with Mac/Linux:
  python3 src/02_generation_generate_sequences.py \
    --split dev \
    --librispeech_root data/raw/LibriSpeech \
    --librispeech_index data/indexes/librispeech_dev_clean.jsonl \
    --out_dir data/generated/dev \
    --num_examples 200 \
    --seed 1337

Exmaple usage with Windows PowerShell:
    python src\02_generation\generate_sequences.py `
    --split dev `
    --librispeech_root data/raw/LibriSpeech `
    --librispeech_index data/indexes/librispeech_dev_clean.jsonl `
    --out_dir data/generated/dev `
    --num_examples 200 `
    --seed 1337

For test:
Example usage with Mac/Linux:
  python3 src/02_generation_generate_sequences.py \
    --split dev \
    --librispeech_root data/raw/LibriSpeech \
    --librispeech_index data/indexes/librispeech_test_clean.jsonl \
    --out_dir data/generated/test
    --num_examples 200 \
    --seed 1337

Exmaple usage with Windows PowerShell:
    python src\02_generation\generate_sequences.py `
    --split dev `
    --librispeech_root data/raw/LibriSpeech `
    --librispeech_index data/indexes/librispeech_test_clean.jsonl `
    --out_dir data/generated/test `
    --num_examples 200 `
    --seed 1337

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from vad_engine import SeqParams, build_clean_sequence, frame_labels_from_intervals


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 2: Generate clean stitched VAD sequences + frame labels.")
    p.add_argument("--split", choices=["train", "dev", "test"], required=True)
    p.add_argument("--librispeech_root", required=True, type=str, help="Local LibriSpeech root.")
    p.add_argument("--librispeech_index", required=True, type=str, help="Index JSONL (uses relpath).")
    p.add_argument("--out_dir", required=True, type=str, help="Output dir, e.g. data/generated/train")
    p.add_argument("--num_examples", required=True, type=int)
    p.add_argument("--seed", type=int, default=1337)

    # sequence parameters
    p.add_argument("--n_min", type=int, default=2)
    p.add_argument("--n_max", type=int, default=6)
    p.add_argument("--gap_min_s", type=float, default=0.2)
    p.add_argument("--gap_max_s", type=float, default=1.0)
    p.add_argument("--leadtrail_min_s", type=float, default=0.2)
    p.add_argument("--leadtrail_max_s", type=float, default=0.6)
    p.add_argument("--lead_prob", type=float, default=0.5)
    p.add_argument("--trail_prob", type=float, default=0.5)
    p.add_argument("--max_utt_s", type=float, default=None)

    # labeling parameters
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--frame_ms", type=float, default=25.0)
    p.add_argument("--hop_ms", type=float, default=10.0)
    p.add_argument("--overlap_thr", type=float, default=0.5)

    # standardization parameters
    p.add_argument("--preemph", action="store_true")
    p.add_argument("--preemph_alpha", type=float, default=0.97)
    p.add_argument("--peak", type=float, default=0.9)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    ls_root = Path(args.librispeech_root).expanduser().resolve()
    index_path = Path(args.librispeech_index).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    rows = read_jsonl(index_path)
    if not rows:
        raise ValueError(f"Index is empty: {index_path}")

    out_audio = out_dir / "clean_audio"
    out_labels = out_dir / "labels"
    out_manifest_dir = out_dir / "manifests"
    out_audio.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    out_manifest_dir.mkdir(parents=True, exist_ok=True)

    params = SeqParams(
        sr=args.sr,
        n_min=args.n_min,
        n_max=args.n_max,
        gap_min_s=args.gap_min_s,
        gap_max_s=args.gap_max_s,
        leadtrail_min_s=args.leadtrail_min_s,
        leadtrail_max_s=args.leadtrail_max_s,
        lead_prob=args.lead_prob,
        trail_prob=args.trail_prob,
        max_utt_s=args.max_utt_s,
    )

    rng = np.random.default_rng(args.seed)

    manifest_rows: List[Dict] = []
    n_records = len(rows)

    for i in range(args.num_examples):
        ex_id = f"{args.split}_{i:07d}"
        ex_seed = int(rng.integers(0, 2**31 - 1))  # per-example seed for reproducibility
        ex_rng = np.random.default_rng(ex_seed)

        N = int(ex_rng.integers(params.n_min, params.n_max + 1))

        chosen = [rows[int(ex_rng.integers(0, n_records))] for _ in range(N)]
        utt_relpaths = [c["relpath"] for c in chosen]
        utt_ids = [c["utt_id"] for c in chosen]

        utt_paths = [str(ls_root / rp) for rp in utt_relpaths]
        for pth in utt_paths:
            if not Path(pth).exists():
                raise FileNotFoundError(
                    f"Missing audio file: {pth}\n"
                    f"Check --librispeech_root and index relpath."
                )

        x_clean, utter_meta, speech_intervals, silence_meta = build_clean_sequence(
            utt_paths=utt_paths,
            utt_ids=utt_ids,
            params=params,
            rng=ex_rng,
            do_preemph=bool(args.preemph),
            preemph_alpha=float(args.preemph_alpha),
            peak=float(args.peak),
        )

        y = frame_labels_from_intervals(
            int(x_clean.size),
            speech_intervals,
            sr=args.sr,
            frame_ms=args.frame_ms,
            hop_ms=args.hop_ms,
            overlap_thr=args.overlap_thr,
        )

        audio_path = out_audio / f"{ex_id}.npy"
        labels_path = out_labels / f"{ex_id}_y.npy"
        np.save(audio_path, x_clean.astype(np.float32, copy=False))
        np.save(labels_path, y.astype(np.uint8, copy=False))

        manifest_rows.append({
            "ex_id": ex_id,
            "split": args.split,
            "sr": args.sr,

            "clean_audio_path": str(audio_path.relative_to(out_dir)),
            "num_samples": int(x_clean.size),

            "utterances": utter_meta,
            "speech_intervals": [{"start": s, "end": e} for (s, e) in speech_intervals],
            "silences": silence_meta,

            "labels_path": str(labels_path.relative_to(out_dir)),
            "frame_params": {"frame_ms": args.frame_ms, "hop_ms": args.hop_ms, "overlap_thr": args.overlap_thr},

            "standardize": {"preemph": bool(args.preemph), "preemph_alpha": float(args.preemph_alpha), "peak": float(args.peak)},

            "source": {"librispeech_index": index_path.name, "utt_relpaths": utt_relpaths},
            "seed": args.seed,
            "ex_seed": ex_seed,
        })

        if (i + 1) % 100 == 0:
            print(f"[OK] Generated {i+1}/{args.num_examples}")

    manifest_path = out_manifest_dir / f"{args.split}_manifest.jsonl"
    write_jsonl(manifest_rows, manifest_path)
    print(f"\n[Done] Wrote manifest: {manifest_path}")
    print(f"[Done] Clean audio: {out_audio}")
    print(f"[Done] Labels: {out_labels}")


if __name__ == "__main__":
    main()