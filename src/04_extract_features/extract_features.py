from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from voice_features import extract_frame_features, stack_context
from feature_normalization import compute_mean_std_from_files, apply_normalization
from manifest_tools import (
    read_jsonl,
    write_jsonl,
    get_manifest_path,
    ensure_feature_dirs,
    get_stats_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract framewise and optional context-stacked features for VAD."
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "dev", "test"],
        help="Dataset split to process",
    )
    parser.add_argument(
        "--generated_dir",
        required=True,
        type=str,
        help="Path to generated split directory, e.g. data/generated/train",
    )
    parser.add_argument(
        "--manifest_type",
        default="noisy",
        choices=["clean", "noisy"],
        help="Use noisy for noise-robust VAD training",
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=40,
        help="Number of mel bins",
    )
    parser.add_argument(
        "--context_left",
        type=int,
        default=5,
        help="Left context frames for stacking",
    )
    parser.add_argument(
        "--context_right",
        type=int,
        default=5,
        help="Right context frames for stacking",
    )
    parser.add_argument(
        "--save_stacked",
        action="store_true",
        help="Also save stacked features for LogReg/MLP",
    )
    parser.add_argument(
        "--norm_stats_in",
        type=str,
        default=None,
        help="Path to train normalization stats .npz. Required for dev/test.",
    )
    return parser.parse_args()


def extract_raw_features_for_split(
    rows: List[Dict],
    generated_dir: Path,
    audio_key: str,
    frame_dir: Path,
    split: str,
    manifest_type: str,
    n_mels: int,
) -> List[Dict]:
    """
    Extract raw framewise features for one split and save them to disk.

    Returns
    -------
    List[Dict]
        Metadata rows describing the saved feature files.
    """
    extracted_rows: List[Dict] = []

    # Read frame parameters from the manifest.
    frame_params = rows[0]["frame_params"]
    frame_ms = float(frame_params["frame_ms"])
    hop_ms = float(frame_params["hop_ms"])

    for idx, row in enumerate(rows):
        ex_id = row["ex_id"]
        sr = int(row["sr"])

        audio_path = generated_dir / row[audio_key]
        labels_path = generated_dir / row["labels_path"]

        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio file: {audio_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels file: {labels_path}")

        x = np.load(audio_path).astype(np.float32, copy=False)
        y = np.load(labels_path).astype(np.int64, copy=False)

        # Extract frame-level acoustic features with shape (T, 121)
        feats = extract_frame_features(
            y=x,
            sr=sr,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            n_mels=n_mels,
        )

        # Safe alignment in case of tiny off-by-one differences.
        T = min(len(feats), len(y))
        feats = feats[:T]
        y = y[:T]

        frame_out = frame_dir / f"{ex_id}_X.npy"
        label_out = frame_dir / f"{ex_id}_y.npy"

        np.save(frame_out, feats.astype(np.float32, copy=False))
        np.save(label_out, y.astype(np.int64, copy=False))

        extracted_rows.append(
            {
                "ex_id": ex_id,
                "split": row.get("split", split),
                "sr": sr,
                "manifest_type": manifest_type,
                "audio_path": row[audio_key],
                "labels_path": str(label_out.relative_to(generated_dir)),
                "frame_features_path": str(frame_out.relative_to(generated_dir)),
                "num_frames": int(T),
                "frame_dim": int(feats.shape[1]),
                "frame_params": row["frame_params"],
            }
        )

        if (idx + 1) % 200 == 0:
            print(f"[OK] Extracted {idx + 1}/{len(rows)} examples")

    return extracted_rows


def save_normalized_features(
    extracted_rows: List[Dict],
    generated_dir: Path,
    mean_frame: np.ndarray,
    std_frame: np.ndarray,
    save_stacked: bool,
    stacked_dir: Path,
    context_left: int,
    context_right: int,
) -> List[Dict]:
    """
    Normalize frame-level features and optionally save context-stacked features.

    Important:
    ----------
    We normalize frame features first using train-set frame stats.
    Then, if requested, we context-stack the normalized frame features.

    We do NOT compute separate stacked mean/std, which avoids a large
    memory-heavy second pass over the training data.
    """
    final_rows: List[Dict] = []

    for idx, row in enumerate(extracted_rows):
        frame_path = generated_dir / row["frame_features_path"]
        label_path = generated_dir / row["labels_path"]

        feats = np.load(frame_path).astype(np.float32, copy=False)
        y = np.load(label_path).astype(np.int64, copy=False)

        # Normalize frame-level features in place.
        feats_norm = apply_normalization(feats, mean_frame, std_frame)
        np.save(frame_path, feats_norm)

        out_row = dict(row)
        out_row["frame_features_normalized"] = True

        # Optional stacked features for LogReg / MLP
        if save_stacked:
            stacked = stack_context(
                feats_norm,
                left=context_left,
                right=context_right,
            )

            stacked_path = stacked_dir / f"{row['ex_id']}_X.npy"
            stacked_label_path = stacked_dir / f"{row['ex_id']}_y.npy"

            np.save(stacked_path, stacked.astype(np.float32, copy=False))
            np.save(stacked_label_path, y.astype(np.int64, copy=False))

            out_row["stacked_features_path"] = str(stacked_path.relative_to(generated_dir))
            out_row["stacked_labels_path"] = str(stacked_label_path.relative_to(generated_dir))
            out_row["stacked_dim"] = int(stacked.shape[1])
            out_row["context_left"] = int(context_left)
            out_row["context_right"] = int(context_right)

        final_rows.append(out_row)

        if (idx + 1) % 200 == 0:
            print(f"[OK] Normalized {idx + 1}/{len(extracted_rows)} examples")

    return final_rows


def main() -> None:
    args = parse_args()

    generated_dir = Path(args.generated_dir).expanduser().resolve()

    manifest_path, audio_key = get_manifest_path(
        generated_dir=generated_dir,
        split=args.split,
        manifest_type=args.manifest_type,
    )

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    rows = read_jsonl(manifest_path)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    frame_dir, stacked_dir = ensure_feature_dirs(
        generated_dir=generated_dir,
        manifest_type=args.manifest_type,
        save_stacked=args.save_stacked,
    )

    # --------------------------------------------------
    # Step 1: Extract raw frame-level features and save
    # --------------------------------------------------
    extracted_rows = extract_raw_features_for_split(
        rows=rows,
        generated_dir=generated_dir,
        audio_key=audio_key,
        frame_dir=frame_dir,
        split=args.split,
        manifest_type=args.manifest_type,
        n_mels=args.n_mels,
    )

    # --------------------------------------------------
    # Step 2: Compute or load train normalization stats
    # --------------------------------------------------
    stats_path = get_stats_path(generated_dir, args.manifest_type)

    if args.split == "train" and args.norm_stats_in is None:
        # Compute stats from saved frame feature files one at a time.
        # This is memory-safe and avoids concatenating the full train set.
        frame_feature_paths = [
            generated_dir / row["frame_features_path"]
            for row in extracted_rows
        ]

        mean_frame, std_frame = compute_mean_std_from_files(frame_feature_paths)

        np.savez(
            stats_path,
            mean_frame=mean_frame,
            std_frame=std_frame,
        )
        print(f"[Done] Saved normalization stats to: {stats_path}")

    else:
        if args.norm_stats_in is None:
            raise ValueError("For dev/test, you must provide --norm_stats_in")

        stats_path = Path(args.norm_stats_in).expanduser().resolve()
        if not stats_path.exists():
            raise FileNotFoundError(f"Normalization stats file not found: {stats_path}")

    # Load stats for both train and dev/test branches.
    stats = np.load(stats_path)
    mean_frame = stats["mean_frame"]
    std_frame = stats["std_frame"]

    # --------------------------------------------------
    # Step 3: Normalize and optionally save stacked feats
    # --------------------------------------------------
    final_rows = save_normalized_features(
        extracted_rows=extracted_rows,
        generated_dir=generated_dir,
        mean_frame=mean_frame,
        std_frame=std_frame,
        save_stacked=args.save_stacked,
        stacked_dir=stacked_dir,
        context_left=args.context_left,
        context_right=args.context_right,
    )

    # --------------------------------------------------
    # Step 4: Write feature manifest
    # --------------------------------------------------
    feature_manifest_path = (
        generated_dir
        / "features"
        / f"{args.split}_{args.manifest_type}_features_manifest.jsonl"
    )
    write_jsonl(final_rows, feature_manifest_path)

    print(f"[Done] Wrote feature manifest: {feature_manifest_path}")
    print(f"[Done] Frame features dir: {frame_dir}")
    if args.save_stacked:
        print(f"[Done] Stacked features dir: {stacked_dir}")


if __name__ == "__main__":
    main()