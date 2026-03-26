from __future__ import annotations

import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """
    Baseline MLP for frame-level VAD.

    Input:
        stacked acoustic feature vector of shape [1331]

    Output:
        one logit for binary speech / non-speech classification
    """

    def __init__(
        self,
        input_dim: int = 1331,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [B, 1331]

        Returns
        -------
        torch.Tensor
            Shape [B]
            Raw logits (not passed through sigmoid)
        """
        logits = self.net(x)          # shape [B, 1]
        logits = logits.squeeze(1)    # shape [B]
        return logits