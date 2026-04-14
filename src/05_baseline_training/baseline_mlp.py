from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class BaselineMLP(nn.Module):
    """Simple frame-level baseline MLP for VAD."""

    def __init__(
        self,
        input_dim: int = 1331,
        hidden_dims=(512, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        dims: List[int] = [input_dim, *list(hidden_dims)]
        layers: List[nn.Module] = []

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.backbone(x)).squeeze(-1)
        return logits
