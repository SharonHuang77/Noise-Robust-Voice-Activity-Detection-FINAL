from __future__ import annotations

import torch
from torch import nn


class CRNNVAD(nn.Module):
    """CRNN for frame-level VAD on sequence inputs.

    Input shape: [B, T, F]
    Output shape: [B, T] logits
    """

    def __init__(
        self,
        input_dim: int = 121,
        conv_channels: tuple[int, int] = (64, 128),
        conv_kernel_size: int = 5,
        rnn_hidden_size: int = 128,
        rnn_layers: int = 1,
        rnn_bidirectional: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if len(conv_channels) != 2:
            raise ValueError("conv_channels must contain exactly two values")

        padding = conv_kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels[0], kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm1d(conv_channels[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=conv_kernel_size, padding=padding),
            nn.BatchNorm1d(conv_channels[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.rnn = nn.GRU(
            input_size=conv_channels[1],
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
            bidirectional=rnn_bidirectional,
        )

        rnn_out_dim = rnn_hidden_size * (2 if rnn_bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_out_dim, rnn_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run CRNN forward pass.

        Parameters
        ----------
        x:
            Tensor of shape [B, T, F].

        Returns
        -------
        torch.Tensor
            Frame logits of shape [B, T].
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B, T, F], got {tuple(x.shape)}")

        # Conv1d expects [B, C, T], where C is feature dim.
        x = x.transpose(1, 2)
        x = self.conv(x)

        # Back to [B, T, C] for GRU.
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)

        logits = self.classifier(x).squeeze(-1)
        return logits