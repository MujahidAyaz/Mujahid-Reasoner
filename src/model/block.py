from __future__ import annotations

import torch
from torch import Tensor, nn

from src.model.attention import GroupedQueryAttention
from src.model.config import ModelConfig
from src.model.mlp import SwiGLU
from src.model.norm import RMSNorm


class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer block.

    Architecture:

        x
        │
        ├──────────────────────────────┐
        │                              │
        ▼                              │
      RMSNorm                          │
        │                              │
        ▼                              │
    GQA Attention                      │
        │                              │
        └───────────────► Residual Add ◄┘
                              │
                              ▼
                            RMSNorm
                              │
                              ▼
                            SwiGLU
                              │
        ┌─────────────────────┘
        │
        └───────────────► Residual Add

    Input:
        [batch, sequence_length, hidden_size]

    Output:
        [batch, sequence_length, hidden_size]
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
        )

        self.self_attn = GroupedQueryAttention(
            config,
        )

        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
        )

        self.mlp = SwiGLU(
            config,
        )

    def forward(
        self,
        x: Tensor,
        position_offset: int = 0,
    ) -> Tensor:
        """
        Apply one Pre-Norm Transformer block.

        Args:
            x:
                Input tensor with shape
                [batch, sequence_length, hidden_size].

            position_offset:
                Starting position for RoPE.

        Returns:
            Tensor with shape
            [batch, sequence_length, hidden_size].
        """

        if x.ndim != 3:
            raise ValueError(
                "Transformer block input must have shape "
                "[batch, sequence_length, hidden_size]."
            )

        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                "Input hidden size does not match "
                "the configured hidden size."
            )

        # ------------------------------------------------------------
        # Attention sub-layer
        # ------------------------------------------------------------

        residual = x

        x = self.input_layernorm(x)

        x = self.self_attn(
            x,
            position_offset=position_offset,
        )

        x = residual + x

        # ------------------------------------------------------------
        # Feed-forward sub-layer
        # ------------------------------------------------------------

        residual = x

        x = self.post_attention_layernorm(x)

        x = self.mlp(x)

        x = residual + x

        return x