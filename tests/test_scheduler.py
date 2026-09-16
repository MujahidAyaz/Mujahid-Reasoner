from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.scheduler import create_warmup_cosine_scheduler


@pytest.fixture
def optimizer() -> torch.optim.Optimizer:
    model = nn.Linear(4, 2)

    return torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )


def test_returns_lambda_lr(optimizer: torch.optim.Optimizer) -> None:
    scheduler = create_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=10,
        total_steps=100,
    )

    assert isinstance(
        scheduler,
        torch.optim.lr_scheduler.LambdaLR,
    )


def test_warmup_increases_learning_rate(
    optimizer: torch.optim.Optimizer,
) -> None:
    scheduler = create_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=10,
        total_steps=100,
    )

    learning_rates = []

    for _ in range(10):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates[-1] > learning_rates[0]


def test_warmup_reaches_peak_learning_rate(
    optimizer: torch.optim.Optimizer,
) -> None:
    peak_lr = 1e-3

    scheduler = create_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=10,
        total_steps=100,
    )

    for _ in range(10):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        peak_lr,
        rel=1e-6,
    )


def test_cosine_decay_decreases_learning_rate(
    optimizer: torch.optim.Optimizer,
) -> None:
    scheduler = create_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=10,
        total_steps=100,
    )

    for _ in range(10):
        optimizer.step()
        scheduler.step()

    peak_lr = optimizer.param_groups[0]["lr"]

    for _ in range(20):
        optimizer.step()
        scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]

    assert current_lr < peak_lr


def test_reaches_minimum_learning_rate(
    optimizer: torch.optim.Optimizer,
) -> None:
    peak_lr = 1e-3
    min_lr_ratio = 0.1

    scheduler = create_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=10,
        total_steps=100,
        min_lr_ratio=min_lr_ratio,
    )

    for _ in range(100):
        optimizer.step()
        scheduler.step()

    expected_lr = peak_lr * min_lr_ratio

    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        expected_lr,
        rel=1e-6,
    )


def test_minimum_lr_ratio(
    optimizer: torch.optim.Optimizer,
) -> None:
    scheduler = create_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=10,
        total_steps=100,
        min_lr_ratio=0.2,
    )

    for _ in range(100):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        2e-4,
        rel=1e-6,
    )


def test_zero_warmup(
    optimizer: torch.optim.Optimizer,
) -> None:
    scheduler = create_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=0,
        total_steps=100,
    )

    optimizer.step()
    scheduler.step()

    assert optimizer.param_groups[0]["lr"] < 1e-3


def test_full_warmup(
    optimizer: torch.optim.Optimizer,
) -> None:
    scheduler = create_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=100,
        total_steps=100,
    )

    for _ in range(100):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        1e-4,
        rel=1e-6,
    )


def test_learning_rate_never_exceeds_peak(
    optimizer: torch.optim.Optimizer,
) -> None:
    scheduler = create_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=10,
        total_steps=100,
    )

    learning_rates = []

    for _ in range(120):
        optimizer.step()
        scheduler.step()
        learning_rates.append(
            optimizer.param_groups[0]["lr"]
        )

    assert max(learning_rates) <= 1e-3 + 1e-9

def test_invalid_warmup_steps(
    optimizer: torch.optim.Optimizer,
) -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        create_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=-1,
            total_steps=100,
        )


def test_invalid_total_steps(
    optimizer: torch.optim.Optimizer,
) -> None:
    with pytest.raises(ValueError, match="total_steps"):
        create_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=10,
            total_steps=0,
        )


def test_warmup_greater_than_total_steps(
    optimizer: torch.optim.Optimizer,
) -> None:
    with pytest.raises(
        ValueError,
        match="warmup_steps",
    ):
        create_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=101,
            total_steps=100,
        )


def test_invalid_min_lr_ratio(
    optimizer: torch.optim.Optimizer,
) -> None:
    with pytest.raises(
        ValueError,
        match="min_lr_ratio",
    ):
        create_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=10,
            total_steps=100,
            min_lr_ratio=1.5,
        )


def test_negative_min_lr_ratio(
    optimizer: torch.optim.Optimizer,
) -> None:
    with pytest.raises(
        ValueError,
        match="min_lr_ratio",
    ):
        create_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=10,
            total_steps=100,
            min_lr_ratio=-0.1,
        )