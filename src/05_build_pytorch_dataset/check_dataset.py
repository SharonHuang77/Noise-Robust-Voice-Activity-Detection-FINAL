from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "05_build_pytorch_dataset"))

from vad_dataset import VADStackedFrameDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check offline stacked VAD dataset.")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--dev_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--manifest_type", type=str, default="noisy", choices=["clean", "noisy"])
    return parser.parse_args()


def inspect_one_dataset(name: str, dataset: VADStackedFrameDataset) -> None:
    print(f"\n[{name}]")
    print(f"Total frames: {len(dataset)}")

    x0, y0 = dataset[0]
    print("First sample:")
    print(f"  x shape: {tuple(x0.shape)}")
    print(f"  x dtype: {x0.dtype}")
    print(f"  y value: {y0.item()}")
    print(f"  y dtype: {y0.dtype}")

    check_indices = [0, 10, 100, 1000]
    print("Sample labels:")
    for idx in check_indices:
        if idx < len(dataset):
            _, y = dataset[idx]
            print(f"  idx={idx}: y={y.item()}")


def main() -> None:
    args = parse_args()

    print("=" * 50)
    print("Checking VAD stacked datasets")
    print("=" * 50)
    print(f"Manifest type: {args.manifest_type}")

    train_dataset = VADStackedFrameDataset(
        generated_dir=args.train_dir,
        split="train",
        manifest_type=args.manifest_type,
    )
    dev_dataset = VADStackedFrameDataset(
        generated_dir=args.dev_dir,
        split="dev",
        manifest_type=args.manifest_type,
    )
    test_dataset = VADStackedFrameDataset(
        generated_dir=args.test_dir,
        split="test",
        manifest_type=args.manifest_type,
    )

    inspect_one_dataset("train", train_dataset)
    inspect_one_dataset("dev", dev_dataset)
    inspect_one_dataset("test", test_dataset)

    print("\nDataset check finished successfully.")


if __name__ == "__main__":
    main()