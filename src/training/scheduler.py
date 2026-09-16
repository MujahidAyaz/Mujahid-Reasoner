from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def create_warmup_cosine_scheduler(
    optimizer: Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Create a learning-rate scheduler with:

        1. Linear warmup
        2. Cosine decay
        3. Configurable minimum learning-rate ratio

    Schedule:

        LR
        │
        │       ┌──────── peak LR
        │      / \
        │     /   \
        │    /     \
        │   /       \____ minimum LR
        │
        └──────────────────────> steps
           warmup       total

    Args:
        optimizer:
            Optimizer whose learning rate will be scheduled.

        warmup_steps:
            Number of steps used for linear warmup.

        total_steps:
            Total number of training steps.

        min_lr_ratio:
            Final learning rate as a fraction of the initial
            optimizer learning rate.

            Example:
                0.1 → final LR = 10% of peak LR.

    Returns:
        PyTorch LambdaLR scheduler.
    """

    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0.")

    if total_steps <= 0:
        raise ValueError("total_steps must be > 0.")

    if warmup_steps > total_steps:
        raise ValueError(
            "warmup_steps cannot be greater than total_steps."
        )

    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(
            "min_lr_ratio must be between 0.0 and 1.0."
        )

    def lr_lambda(step: int) -> float:
        # -------------------------
        # Warmup
        # -------------------------
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        # -------------------------
        # After warmup
        # -------------------------
        if total_steps == warmup_steps:
            return min_lr_ratio

        progress = (
            float(step - warmup_steps)
            / float(total_steps - warmup_steps)
        )

        progress = min(max(progress, 0.0), 1.0)

        # Cosine decay:
        # 1.0 → min_lr_ratio
        cosine_decay = 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        return (
            min_lr_ratio
            + (1.0 - min_lr_ratio) * cosine_decay
        )

    return LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )