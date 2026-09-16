from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass
class TrainingMetrics:
    """Track training and validation metrics."""

    train_loss: float = 0.0
    validation_loss: float = math.inf

    learning_rate: float = 0.0
    gradient_norm: float = 0.0

    total_tokens: int = 0
    global_step: int = 0

    elapsed_seconds: float = 0.0

    @property
    def train_perplexity(self) -> float:
        return loss_to_perplexity(self.train_loss)

    @property
    def validation_perplexity(self) -> float:
        return loss_to_perplexity(
            self.validation_loss
        )

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0

        return (
            self.total_tokens
            / self.elapsed_seconds
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "train_perplexity": self.train_perplexity,
            "validation_perplexity": (
                self.validation_perplexity
            ),
            "learning_rate": self.learning_rate,
            "gradient_norm": self.gradient_norm,
            "total_tokens": self.total_tokens,
            "global_step": self.global_step,
            "elapsed_seconds": self.elapsed_seconds,
            "tokens_per_second": self.tokens_per_second,
        }


def loss_to_perplexity(loss: float) -> float:
    """
    Convert causal language-model loss to perplexity.

    PPL = exp(loss)
    """

    if not math.isfinite(loss):
        return math.inf

    # Prevent overflow for pathological losses.
    if loss > 100:
        return math.inf

    return math.exp(loss)


class TrainingTimer:
    """Simple monotonic timer for training runs."""

    def __init__(self) -> None:
        self._start_time: float | None = None

    def start(self) -> None:
        self._start_time = time.perf_counter()

    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0

        return (
            time.perf_counter()
            - self._start_time
        )