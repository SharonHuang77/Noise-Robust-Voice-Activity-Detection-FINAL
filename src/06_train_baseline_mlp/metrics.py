from __future__ import annotations

import torch


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert raw logits to probabilities using sigmoid.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model outputs

    Returns
    -------
    torch.Tensor
        Probabilities in [0, 1]
    """
    return torch.sigmoid(logits)


def probs_to_preds(probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Convert probabilities to binary predictions.

    Parameters
    ----------
    probs : torch.Tensor
        Probabilities in [0, 1]
    threshold : float
        Decision threshold

    Returns
    -------
    torch.Tensor
        Binary predictions: 0 or 1
    """
    return (probs >= threshold).to(torch.int64)


def compute_confusion_counts(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
):
    """
    Compute TP, TN, FP, FN for binary classification.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground-truth labels, values in {0, 1}
    y_pred : torch.Tensor
        Predicted labels, values in {0, 1}

    Returns
    -------
    tuple
        (tp, tn, fp, fn)
    """
    y_true = y_true.to(torch.int64)
    y_pred = y_pred.to(torch.int64)

    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    tn = ((y_true == 0) & (y_pred == 0)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()

    return tp, tn, fp, fn


def compute_f1(tp: int, fp: int, fn: int) -> float:
    """
    Compute F1 score from confusion counts.
    """
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return (2 * tp) / denom


def compute_far(fp: int, tn: int) -> float:
    """
    Compute False Alarm Rate (FAR).

    FAR = FP / (FP + TN)
    """
    denom = fp + tn
    if denom == 0:
        return 0.0
    return fp / denom


def compute_miss_rate(fn: int, tp: int) -> float:
    """
    Compute Miss Rate.

    Miss Rate = FN / (FN + TP)
    """
    denom = fn + tp
    if denom == 0:
        return 0.0
    return fn / denom


def compute_metrics_from_logits(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    threshold: float = 0.5,
):
    """
    Compute F1, FAR, and Miss Rate directly from logits.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model outputs
    y_true : torch.Tensor
        Ground-truth labels
    threshold : float
        Decision threshold

    Returns
    -------
    dict
        {
            "f1": ...,
            "far": ...,
            "miss_rate": ...,
            "tp": ...,
            "tn": ...,
            "fp": ...,
            "fn": ...
        }
    """
    probs = logits_to_probs(logits)
    preds = probs_to_preds(probs, threshold=threshold)

    tp, tn, fp, fn = compute_confusion_counts(y_true, preds)

    return {
        "f1": compute_f1(tp, fp, fn),
        "far": compute_far(fp, tn),
        "miss_rate": compute_miss_rate(fn, tp),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }