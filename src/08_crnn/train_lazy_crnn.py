from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

current_dir = Path(__file__).parent
lazy_dir = current_dir.parent / "07_lazy_feature_mlp"
if str(lazy_dir) not in sys.path:
    sys.path.insert(0, str(lazy_dir))

try:
    from .crnn import CRNNVAD
except ImportError:
    from crnn import CRNNVAD

from lazy_dataset import VADLazySequenceDataset


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Lazy CRNN (sequence input, no context stacking)")
    parser.add_argument("--generated_dir", type=str, required=True, help="Path to data/generated")
    parser.add_argument("--norm_stats_path", type=str, help="Path to norm stats .npz file")
    parser.add_argument("--manifest_type", type=str, default="noisy", choices=["clean", "noisy"])
    parser.add_argument("--train_subset_fraction", type=float, default=1.0)
    parser.add_argument("--dev_subset_fraction", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--conv_channels", type=str, default="64,128", help="Comma-separated conv channels, e.g. 64,128")
    parser.add_argument("--conv_kernel_size", type=int, default=5)
    parser.add_argument("--rnn_hidden_size", type=int, default=128)
    parser.add_argument("--rnn_layers", type=int, default=1)
    parser.add_argument("--rnn_bidirectional", type=int, default=1, choices=[0, 1])

    parser.add_argument("--output_dir", type=str, default="outputs/stage2_lazy_crnn")
    return parser.parse_args()


def parse_conv_channels(raw: str) -> tuple[int, int]:
    values = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if len(values) != 2:
        raise ValueError("--conv_channels must contain exactly two comma-separated integers")
    return (values[0], values[1])


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_pad_collate_fn() -> callable:
    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor | List[str]]:
        lengths = [int(item["num_frames"]) for item in batch]
        max_len = max(lengths)
        feat_dim = int(batch[0]["x"].shape[-1])

        batch_size = len(batch)
        x_pad = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)
        y_pad = torch.zeros((batch_size, max_len), dtype=torch.float32)
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        ex_ids: List[str] = []

        for i, item in enumerate(batch):
            t = int(item["num_frames"])
            x_pad[i, :t] = item["x"]
            y_pad[i, :t] = item["y"]
            mask[i, :t] = True
            ex_ids.append(item["ex_id"])

        return {
            "x": x_pad,
            "y": y_pad,
            "mask": mask,
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "ex_ids": ex_ids,
        }

    return collate


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    train_dataset = VADLazySequenceDataset(
        generated_dir=args.generated_dir,
        split="train",
        manifest_type=args.manifest_type,
        norm_stats_path=args.norm_stats_path,
        subset_fraction=args.train_subset_fraction,
        subset_seed=args.seed,
    )
    dev_dataset = VADLazySequenceDataset(
        generated_dir=args.generated_dir,
        split="dev",
        manifest_type=args.manifest_type,
        norm_stats_path=args.norm_stats_path,
        subset_fraction=args.dev_subset_fraction,
        subset_seed=args.seed + 1,
    )

    collate_fn = make_pad_collate_fn()
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return train_loader, dev_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None,
) -> EpochMetrics:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0.0
    total_frames = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            logits_valid = logits[mask]
            y_valid = y[mask]

            loss = criterion(logits_valid, y_valid)

            if is_train:
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            preds = (logits_valid >= 0.0).to(torch.float32)
            correct = (preds == y_valid).to(torch.float32).sum().item()
            n = int(y_valid.numel())

        total_loss += float(loss.item()) * n
        total_correct += float(correct)
        total_frames += n

    if total_frames == 0:
        return EpochMetrics(loss=0.0, accuracy=0.0)

    return EpochMetrics(
        loss=total_loss / total_frames,
        accuracy=total_correct / total_frames,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    conv_channels = parse_conv_channels(args.conv_channels)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    train_loader, dev_loader = build_dataloaders(args)

    model = CRNNVAD(
        input_dim=121,
        conv_channels=conv_channels,
        conv_kernel_size=args.conv_kernel_size,
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_layers=args.rnn_layers,
        rnn_bidirectional=bool(args.rnn_bidirectional),
        dropout=args.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 70)
    print("Lazy CRNN Training (sequence input, no context stacking)")
    print(f"Device: {device}")
    print(f"Train sequences: {len(train_loader.dataset)} | Dev sequences: {len(dev_loader.dataset)}")
    print(f"Train batches: {len(train_loader)} | Dev batches: {len(dev_loader)}")
    print("=" * 70)

    history: Dict[str, List[Dict[str, float]]] = {"train": [], "dev": []}
    best_dev_acc = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        dev_metrics = run_epoch(model, dev_loader, criterion, device, optimizer=None)

        history["train"].append({"loss": train_metrics.loss, "accuracy": train_metrics.accuracy})
        history["dev"].append({"loss": dev_metrics.loss, "accuracy": dev_metrics.accuracy})

        print(
            f"[Epoch {epoch:02d}/{args.epochs:02d}] "
            f"train_loss={train_metrics.loss:.6f} train_acc={train_metrics.accuracy:.4f} | "
            f"dev_loss={dev_metrics.loss:.6f} dev_acc={dev_metrics.accuracy:.4f}"
        )

        if dev_metrics.accuracy > best_dev_acc:
            best_dev_acc = dev_metrics.accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  Saved best model at epoch {epoch} (dev_acc={best_dev_acc:.4f})")

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Best dev_acc={best_dev_acc:.4f} at epoch {best_epoch}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()