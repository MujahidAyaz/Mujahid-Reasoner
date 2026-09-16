from __future__ import annotations

import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Applies position-dependent rotations to query and key vectors.

    Expected input shape:

        [batch, heads, sequence_length, head_dimension]

    The rotation is applied independently to each pair of dimensions.
    """

    def __init__(
        self,
        head_dimension: int,
        max_sequence_length: int,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()

        if head_dimension <= 0:
            raise ValueError(
                "head_dimension must be positive."
            )

        if head_dimension % 2 != 0:
            raise ValueError(
                "head_dimension must be even for RoPE."
            )

        if max_sequence_length <= 0:
            raise ValueError(
                "max_sequence_length must be positive."
            )

        if theta <= 0:
            raise ValueError(
                "theta must be positive."
            )

        self.head_dimension = head_dimension
        self.max_sequence_length = max_sequence_length
        self.theta = theta

        inverse_frequency = 1.0 / (
            theta
            ** (
                torch.arange(
                    0,
                    head_dimension,
                    2,
                    dtype=torch.float32,
                )
                / head_dimension
            )
        )

        positions = torch.arange(
            max_sequence_length,
            dtype=torch.float32,
        )

        frequencies = torch.outer(
            positions,
            inverse_frequency,
        )

        cos = frequencies.cos()
        sin = frequencies.sin()

        self.register_buffer(
            "cos_cached",
            cos,
            persistent=False,
        )

        self.register_buffer(
            "sin_cached",
            sin,
            persistent=False,
        )

    def forward(
        self,
        x: Tensor,
        position_offset: int = 0,
    ) -> Tensor:
        """
        Apply rotary positional encoding.

        Args:
            x:
                Tensor with shape
                [batch, heads, sequence_length, head_dimension].

            position_offset:
                Starting position used during cached decoding.

        Returns:
            Tensor with the same shape and dtype as x.
        """

        if x.ndim != 4:
            raise ValueError(
                "RoPE input must have shape "
                "[batch, heads, sequence_length, head_dimension]."
            )

        batch, heads, sequence_length, head_dimension = (
            x.shape
        )

        if head_dimension != self.head_dimension:
            raise ValueError(
                "Input head dimension does not match "
                "the configured head dimension."
            )

        if position_offset < 0:
            raise ValueError(
                "position_offset cannot be negative."
            )

        end_position = (
            position_offset + sequence_length
        )

        if end_position > self.max_sequence_length:
            raise ValueError(
                "Sequence exceeds the configured "
                "maximum RoPE length."
            )

        cos = self.cos_cached[
            position_offset:end_position
        ]

        sin = self.sin_cached[
            position_offset:end_position
        ]

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        x_float = x.float()

        x_even = x_float[..., 0::2]
        x_odd = x_float[..., 1::2]

        rotated_even = (
            x_even * cos
            - x_odd * sin
        )

        rotated_odd = (
            x_even * sin
            + x_odd * cos
        )

        output = torch.empty_like(
            x_float
        )

        output[..., 0::2] = rotated_even
        output[..., 1::2] = rotated_odd

        return output.to(x.dtype)