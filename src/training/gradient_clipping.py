from __future__ import annotations

import torch
from torch import nn


def clip_gradients(
    model: nn.Module,
    *,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> float:
    """
    Clip model gradients by global norm.

    Args:
        model:
            Model whose gradients should be clipped.

        max_norm:
            Maximum allowed global gradient norm.

        norm_type:
            Type of norm used for gradient clipping.

    Returns:
        Total gradient norm before clipping.

    Raises:
        ValueError:
            If max_norm is not positive.
        ValueError:
            If norm_type is not positive.
    """

    if max_norm <= 0:
        raise ValueError("max_norm must be > 0.")

    if norm_type <= 0:
        raise ValueError("norm_type must be > 0.")

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and parameter.grad is not None
    ]

    if not parameters:
        return 0.0

    total_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=max_norm,
        norm_type=norm_type,
    )

    return float(total_norm.item())