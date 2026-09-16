from __future__ import annotations

import torch
from torch import Tensor, nn


class CausalLanguageModelLoss(nn.Module):
    """
    Cross-entropy loss for causal language modeling.

    The model predicts the next token at every position.

    Example:

        Input:
            [The, cat, sat]

        Targets:
            [cat, sat, down]

    Logits:
        [batch, sequence_length, vocab_size]

    Targets:
        [batch, sequence_length]

    The sequence and vocabulary dimensions are flattened before
    applying cross-entropy.
    """

    def __init__(
        self,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()

        self.ignore_index = ignore_index

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
        )

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        Calculate next-token prediction loss.

        Args:
            logits:
                Model predictions with shape:
                [batch, sequence_length, vocab_size]

            targets:
                Target token IDs with shape:
                [batch, sequence_length]

        Returns:
            Scalar cross-entropy loss.
        """

        if logits.ndim != 3:
            raise ValueError(
                "logits must have shape "
                "[batch, sequence_length, vocab_size]."
            )

        if targets.ndim != 2:
            raise ValueError(
                "targets must have shape "
                "[batch, sequence_length]."
            )

        if logits.shape[:2] != targets.shape:
            raise ValueError(
                "logits and targets must have matching "
                "batch and sequence dimensions."
            )

        if targets.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "targets must use an integer dtype."
            )

        if logits.shape[-1] <= 1:
            raise ValueError(
                "vocabulary size must be greater than 1."
            )

        logits = logits.reshape(
            -1,
            logits.shape[-1],
        )

        targets = targets.reshape(-1)

        return self.loss_fn(
            logits,
            targets,
        )