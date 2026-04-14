from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

try:
    from .baseline_mlp import BaselineMLP
    from .offline_dataset import build_dataloader
except ImportError:
    # Supports running as a direct script: python src/05_baseline_training/train_baseline_mlp.py
    from baseline_mlp import BaselineMLP
    from offline_dataset import build_dataloader


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


def binary_accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (logits >= 0.0).to(torch.float32)
    return float((preds == labels).to(torch.float32).mean().item())


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None,
) -> EpochMetrics:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0.0
    total_frames = 0

    print(f"Entering {'train' if is_train else 'dev'} epoch with {len(loader)} batches")

    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx % 100 == 0:
            print(f"Loaded batch {batch_idx}/{len(loader)}")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = int(y.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_correct += binary_accuracy_from_logits(logits, y) * batch_size
        total_frames += batch_size

    if total_frames == 0:
        return EpochMetrics(loss=0.0, accuracy=0.0)

    return EpochMetrics(
        loss=total_loss / total_frames,
        accuracy=total_correct / total_frames,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Step 05 offline MLP")
    parser.add_argument("--data_root", type=str, required=True, help="Path to data/generated")
    parser.add_argument("--manifest_type", type=str, default="clean", choices=["clean", "noisy"])
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--dev_fraction", type=float, default=1.0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(
    data_root: str | Path,
    manifest_type: str,
    batch_size: int,
    num_workers: int,
    train_fraction: float,
    dev_fraction: float,
    seed: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_dir = Path(data_root) / "train"
    dev_dir = Path(data_root) / "dev"

    train_loader = build_dataloader(
        generated_dir=train_dir,
        split="train",
        manifest_type=manifest_type,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    dev_loader = build_dataloader(
        generated_dir=dev_dir,
        split="dev",
        manifest_type=manifest_type,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    train_loader = maybe_subsample_loader(
        loader=train_loader,
        fraction=train_fraction,
        seed=seed,
        shuffle=True,
        name="train",
    )
    dev_loader = maybe_subsample_loader(
        loader=dev_loader,
        fraction=dev_fraction,
        seed=seed + 1,
        shuffle=False,
        name="dev",
    )

    return train_loader, dev_loader


def maybe_subsample_loader(
    loader: DataLoader,
    fraction: float,
    seed: int,
    shuffle: bool,
    name: str,
) -> DataLoader:
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"{name}_fraction must be in (0, 1], got {fraction}")

    if fraction >= 1.0:
        return loader

    total = len(loader.dataset)
    keep = max(1, int(total * fraction))

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator)[:keep].tolist()
    subset = Subset(loader.dataset, indices)

    print(f"Using {name} subset: {keep}/{total} frames ({fraction:.2%})")

    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
    )


def maybe_save_model(model: nn.Module, save_path: str) -> None:
    if not save_path:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved model checkpoint: {path}")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, dev_loader = build_loaders(
        data_root=args.data_root,
        manifest_type=args.manifest_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_fraction=args.train_fraction,
        dev_fraction=args.dev_fraction,
        seed=args.seed,
    )

    model = BaselineMLP(input_dim=1331, hidden_dims=(512, 256), dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 70)
    print(f"Step 05 Offline MLP Training ({args.manifest_type.capitalize()} Features)")
    print(f"Device: {device}")
    print(f"Train frames: {len(train_loader.dataset)} | Dev frames: {len(dev_loader.dataset)}")
    print("=" * 70)

    best_dev_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        dev_metrics = run_epoch(model, dev_loader, criterion, device, optimizer=None)

        print(
            f"[Epoch {epoch:02d}/{args.epochs:02d}] "
            f"train_loss={train_metrics.loss:.6f} train_acc={train_metrics.accuracy:.4f} | "
            f"dev_loss={dev_metrics.loss:.6f} dev_acc={dev_metrics.accuracy:.4f}"
        )

        if dev_metrics.loss < best_dev_loss:
            best_dev_loss = dev_metrics.loss
            maybe_save_model(model, args.save_path)

    print("Training finished.")


if __name__ == "__main__":
    main()
