from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.metrics import (
    TrainingMetrics,
    TrainingTimer,
    loss_to_perplexity,
)


def test_loss_to_perplexity() -> None:
    assert loss_to_perplexity(0.0) == pytest.approx(1.0)
    assert loss_to_perplexity(1.0) == pytest.approx(
        math.e
    )


def test_perplexity_is_exponential_of_loss() -> None:
    loss = 2.0

    assert loss_to_perplexity(loss) == pytest.approx(
        math.exp(2.0)
    )


def test_invalid_loss_returns_infinity() -> None:
    assert loss_to_perplexity(math.inf) == math.inf
    assert loss_to_perplexity(math.nan) == math.inf


def test_extremely_large_loss() -> None:
    assert loss_to_perplexity(101.0) == math.inf


def test_training_metrics_defaults() -> None:
    metrics = TrainingMetrics()

    assert metrics.train_loss == 0.0
    assert metrics.validation_loss == math.inf
    assert metrics.total_tokens == 0
    assert metrics.global_step == 0


def test_train_perplexity() -> None:
    metrics = TrainingMetrics(
        train_loss=2.0
    )

    assert metrics.train_perplexity == pytest.approx(
        math.exp(2.0)
    )


def test_validation_perplexity() -> None:
    metrics = TrainingMetrics(
        validation_loss=3.0
    )

    assert metrics.validation_perplexity == pytest.approx(
        math.exp(3.0)
    )


def test_tokens_per_second() -> None:
    metrics = TrainingMetrics(
        total_tokens=1000,
        elapsed_seconds=10.0,
    )

    assert metrics.tokens_per_second == pytest.approx(
        100.0
    )


def test_zero_elapsed_time() -> None:
    metrics = TrainingMetrics(
        total_tokens=1000,
        elapsed_seconds=0.0,
    )

    assert metrics.tokens_per_second == 0.0


def test_to_dict() -> None:
    metrics = TrainingMetrics(
        train_loss=2.0,
        validation_loss=3.0,
        learning_rate=0.0003,
        gradient_norm=1.2,
        total_tokens=1024,
        global_step=10,
        elapsed_seconds=2.0,
    )

    result = metrics.to_dict()

    assert result["train_loss"] == 2.0
    assert result["validation_loss"] == 3.0
    assert result["global_step"] == 10
    assert result["total_tokens"] == 1024
    assert "train_perplexity" in result
    assert "validation_perplexity" in result
    assert "tokens_per_second" in result


def test_timer_before_start() -> None:
    timer = TrainingTimer()

    assert timer.elapsed() == 0.0


def test_timer_runs() -> None:
    timer = TrainingTimer()

    timer.start()

    elapsed = timer.elapsed()

    assert elapsed >= 0.0