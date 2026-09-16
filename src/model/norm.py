from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm normalizes activations using their root-mean-square value
    without subtracting the mean.

    Formula:

        RMS(x) = sqrt(mean(x²) + eps)

        y = x / RMS(x) * weight
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError(
                "hidden_size must be positive."
            )

        if eps <= 0:
            raise ValueError(
                "eps must be positive."
            )

        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(hidden_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization."""

        input_dtype = x.dtype

        x_float = x.float()

        variance = x_float.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        x_normalized = x_float * torch.rsqrt(
            variance + self.eps
        )

        return (
            self.weight * x_normalized
        ).to(input_dtype)