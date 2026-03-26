from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "05_build_pytorch_dataset"))
sys.path.append(str(PROJECT_ROOT / "src" / "06_train_baseline_mlp"))

from vad_dataset import VADStackedFrameDataset
from baseline_mlp import BaselineMLP
from train_utils import train_one_epoch, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline MLP training for VAD.")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--dev_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--manifest_type", type=str, default="noisy", choices=["clean", "noisy"])

    parser.add_argument("--input_dim", type=int, default=1331)
    parser.add_argument("--hidden_dim", type=int, default=256)

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="baseline_mlp_noisy")

    return parser.parse_args()


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Baseline MLP Training")
    print("=" * 60)
    print(f"Device:         {device}")
    print(f"Manifest type:  {args.manifest_type}")
    print(f"Epochs:         {args.epochs}")
    print(f"Learning rate:  {args.learning_rate}")
    print(f"Weight decay:   {args.weight_decay}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Hidden dim:     {args.hidden_dim}")
    print("=" * 60)

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

    print(f"Train frames: {len(train_dataset)} | Dev frames: {len(dev_dataset)} | Test frames: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    model = BaselineMLP(input_dim=args.input_dim, hidden_dim=args.hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = {
        "train_loss": [],
        "train_f1": [],
        "train_far": [],
        "train_miss_rate": [],
        "dev_loss": [],
        "dev_f1": [],
        "dev_far": [],
        "dev_miss_rate": [],
    }

    best_dev_f1 = -1.0
    best_epoch = -1
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        dev_metrics = evaluate(
            model=model,
            dataloader=dev_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_f1"].append(train_metrics["f1"])
        history["train_far"].append(train_metrics["far"])
        history["train_miss_rate"].append(train_metrics["miss_rate"])

        history["dev_loss"].append(dev_metrics["loss"])
        history["dev_f1"].append(dev_metrics["f1"])
        history["dev_far"].append(dev_metrics["far"])
        history["dev_miss_rate"].append(dev_metrics["miss_rate"])

        print("Train:")
        for k, v in train_metrics.items():
            print(f"  {k}: {v:.6f}")

        print("Dev:")
        for k, v in dev_metrics.items():
            print(f"  {k}: {v:.6f}")

        if dev_metrics["f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["f1"]
            best_epoch = epoch
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            print(f"  [Best model updated] epoch={epoch}, dev_f1={best_dev_f1:.6f}")

    if best_state_dict is None:
        raise RuntimeError("No best checkpoint was saved.")

    print("\n===== Training finished =====")
    print(f"Best epoch: {best_epoch}")
    print(f"Best dev F1: {best_dev_f1:.6f}")

    final_model = BaselineMLP(input_dim=args.input_dim, hidden_dim=args.hidden_dim).to(device)
    final_model.load_state_dict(best_state_dict)

    final_test_metrics = evaluate(
        model=final_model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\nFinal test metrics:")
    for k, v in final_test_metrics.items():
        print(f"  {k}: {v:.6f}")

    results_dir = Path(args.results_dir) / args.run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.save(best_state_dict, results_dir / "best_model.pt")

    summary = {
        "run_name": args.run_name,
        "device": str(device),
        "manifest_type": args.manifest_type,
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_dev_f1": best_dev_f1,
        "final_test_metrics": final_test_metrics,
    }

    save_json(history, results_dir / "history.json")
    save_json(summary, results_dir / "summary.json")

    print("\nSaved files:")
    print(f"  {results_dir / 'best_model.pt'}")
    print(f"  {results_dir / 'history.json'}")
    print(f"  {results_dir / 'summary.json'}")


if __name__ == "__main__":
    main()