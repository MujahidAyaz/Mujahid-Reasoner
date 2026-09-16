from __future__ import annotations

import torch
from torch import Tensor, nn

from src.model.config import ModelConfig


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network.

    Architecture:

        x
        │
        ├── gate_proj ──→ SiLU ──┐
        │                         ×
        └── up_proj ─────────────┘
                                  │
                              down_proj
                                  │
                                output

    Formula:

        SwiGLU(x) = down_proj(
            SiLU(gate_proj(x)) * up_proj(x)
        )

    Input:
        [batch, sequence_length, hidden_size]

    Output:
        [batch, sequence_length, hidden_size]
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )

        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )

        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

        self.activation = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the SwiGLU feed-forward transformation.

        Args:
            x:
                Tensor with shape
                [batch, sequence_length, hidden_size].

        Returns:
            Tensor with shape
            [batch, sequence_length, hidden_size].
        """

        if x.ndim != 3:
            raise ValueError(
                "SwiGLU input must have shape "
                "[batch, sequence_length, hidden_size]."
            )

        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                "Input hidden size does not match "
                "the configured hidden size."
            )

        gate = self.activation(
            self.gate_proj(x)
        )

        up = self.up_proj(x)

        hidden = gate * up

        return self.down_proj(hidden)