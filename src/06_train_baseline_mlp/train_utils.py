from __future__ import annotations

from typing import Dict

import torch

from metrics import compute_metrics_from_logits


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Returns
    -------
    dict
        Training summary with average loss and metrics.
    """
    model.train()

    total_loss = 0.0
    total_examples = 0

    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        batch_metrics = compute_metrics_from_logits(logits.detach(), y_batch.detach(), threshold=0.5)
        total_tp += batch_metrics["tp"]
        total_tn += batch_metrics["tn"]
        total_fp += batch_metrics["fp"]
        total_fn += batch_metrics["fn"]

    avg_loss = total_loss / total_examples

    denom_f1 = 2 * total_tp + total_fp + total_fn
    f1 = 0.0 if denom_f1 == 0 else (2 * total_tp) / denom_f1

    denom_far = total_fp + total_tn
    far = 0.0 if denom_far == 0 else total_fp / denom_far

    denom_miss = total_fn + total_tp
    miss_rate = 0.0 if denom_miss == 0 else total_fn / denom_miss

    return {
        "loss": avg_loss,
        "f1": f1,
        "far": far,
        "miss_rate": miss_rate,
    }


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate the model on a validation or test split.

    Returns
    -------
    dict
        Evaluation summary with average loss and metrics.
    """
    model.eval()

    total_loss = 0.0
    total_examples = 0

    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        batch_size = x_batch.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        batch_metrics = compute_metrics_from_logits(logits, y_batch, threshold=0.5)
        total_tp += batch_metrics["tp"]
        total_tn += batch_metrics["tn"]
        total_fp += batch_metrics["fp"]
        total_fn += batch_metrics["fn"]

    avg_loss = total_loss / total_examples

    denom_f1 = 2 * total_tp + total_fp + total_fn
    f1 = 0.0 if denom_f1 == 0 else (2 * total_tp) / denom_f1

    denom_far = total_fp + total_tn
    far = 0.0 if denom_far == 0 else total_fp / denom_far

    denom_miss = total_fn + total_tp
    miss_rate = 0.0 if denom_miss == 0 else total_fn / denom_miss

    return {
        "loss": avg_loss,
        "f1": f1,
        "far": far,
        "miss_rate": miss_rate,
    }