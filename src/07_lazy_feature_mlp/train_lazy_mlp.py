from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

# Robust imports
import sys
from pathlib import Path
current_dir = Path(__file__).parent
baseline_dir = current_dir.parent / "05_baseline_training"
if str(baseline_dir) not in sys.path:
    sys.path.insert(0, str(baseline_dir))
from baseline_mlp import BaselineMLP

# Import lazy frame dataset
try:
    from .lazy_frame_dataset import VADLazyFrameDataset
except ImportError:
    # Direct script mode
    lazy_dir = current_dir
    if str(lazy_dir) not in sys.path:
        sys.path.insert(0, str(lazy_dir))
    from lazy_frame_dataset import VADLazyFrameDataset


def binary_accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (logits >= 0.0).to(torch.float32)
    return float((preds == labels).to(torch.float32).mean().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0.0
    total_frames = 0

    for batch_idx, batch in enumerate(loader):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)

            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = int(y.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_correct += binary_accuracy_from_logits(logits, y) * batch_size
        total_frames += batch_size

    if total_frames == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    return {
        "loss": total_loss / total_frames,
        "accuracy": total_correct / total_frames,
    }


def build_dataloaders(
    generated_dir: str | Path,
    norm_stats_path: str | Path | None,
    manifest_type: str,
    train_subset_fraction: float | None,
    dev_subset_fraction: float | None,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = VADLazyFrameDataset(
        generated_dir=generated_dir,
        split="train",
        manifest_type=manifest_type,
        norm_stats_path=norm_stats_path,
        subset_fraction=train_subset_fraction,
        subset_seed=seed,
    )
    dev_dataset = VADLazyFrameDataset(
        generated_dir=generated_dir,
        split="dev",
        manifest_type=manifest_type,
        norm_stats_path=norm_stats_path,
        subset_fraction=dev_subset_fraction,
        subset_seed=seed + 1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, dev_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 2 Lazy MLP")
    parser.add_argument("--generated_dir", type=str, required=True, help="Path to data/generated")
    parser.add_argument("--norm_stats_path", type=str, help="Path to norm stats .npz file")
    parser.add_argument("--manifest_type", type=str, default="noisy", choices=["clean", "noisy"])
    parser.add_argument("--train_subset_fraction", type=float, help="Fraction of train examples")
    parser.add_argument("--dev_subset_fraction", type=float, help="Fraction of dev examples")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/stage2_lazy_mlp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training arguments
    args_path = output_dir / "train_args.json"
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # Build dataloaders
    train_loader, dev_loader = build_dataloaders(
        generated_dir=args.generated_dir,
        norm_stats_path=args.norm_stats_path,
        manifest_type=args.manifest_type,
        train_subset_fraction=args.train_subset_fraction,
        dev_subset_fraction=args.dev_subset_fraction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Build model
    model = BaselineMLP(
        input_dim=1331,
        hidden_dims=(512, 256),
        dropout=args.dropout,
    ).to(device)

    # Optimizer and criterion
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # Training history
    history = {"train": [], "dev": []}

    best_dev_accuracy = 0.0
    best_epoch = 0
    best_checkpoint_path = output_dir / "best_model.pt"

    print(f"Starting training for {args.epochs} epochs on {device}")
    print(f"Train batches: {len(train_loader)}, Dev batches: {len(dev_loader)}")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        history["train"].append(train_metrics)

        # Evaluate
        dev_metrics = run_epoch(model, dev_loader, criterion, device, None)
        history["dev"].append(dev_metrics)

        print(f"Epoch {epoch:2d}: Train Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f} | "
              f"Dev Loss={dev_metrics['loss']:.4f}, Acc={dev_metrics['accuracy']:.4f}")

        # Save best model
        if dev_metrics["accuracy"] > best_dev_accuracy:
            best_dev_accuracy = dev_metrics["accuracy"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  Saved best model with dev acc {best_dev_accuracy:.4f}")

    # Save history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Best dev accuracy: {best_dev_accuracy:.4f} at epoch {best_epoch}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()