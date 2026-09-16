from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.optim import AdamW


def create_adamw_optimizer(
    model: nn.Module,
    *,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> AdamW:
    """
    Create an AdamW optimizer with parameter groups.

    Weight decay is applied to matrix-like parameters such as
    Linear and Embedding weights, while biases and normalization
    parameters are excluded from weight decay.

    Args:
        model:
            Model whose parameters will be optimized.

        learning_rate:
            Initial learning rate.

        weight_decay:
            AdamW weight decay coefficient.

        betas:
            AdamW beta coefficients.

        eps:
            Numerical stability term.

    Returns:
        Configured AdamW optimizer.
    """

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be positive."
        )

    if weight_decay < 0:
        raise ValueError(
            "weight_decay cannot be negative."
        )

    beta1, beta2 = betas

    if not 0.0 <= beta1 < 1.0:
        raise ValueError(
            "betas[0] must be in [0, 1)."
        )

    if not 0.0 <= beta2 < 1.0:
        raise ValueError(
            "betas[1] must be in [0, 1)."
        )

    if eps <= 0:
        raise ValueError(
            "eps must be positive."
        )

    decay_parameters: list[nn.Parameter] = []
    no_decay_parameters: list[nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    parameter_groups = [
        {
            "params": decay_parameters,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ]

    return AdamW(
        parameter_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )