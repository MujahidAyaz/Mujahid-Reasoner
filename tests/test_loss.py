from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
import torch.nn.functional as F

from src.training.loss import CausalLanguageModelLoss


@pytest.fixture
def loss_fn() -> CausalLanguageModelLoss:
    return CausalLanguageModelLoss()


def test_loss_returns_scalar(
    loss_fn: CausalLanguageModelLoss,
) -> None:
    logits = torch.randn(2, 8, 100)
    targets = torch.randint(0, 100, (2, 8))

    loss = loss_fn(logits, targets)

    assert loss.ndim == 0


def test_loss_is_finite(
    loss_fn: CausalLanguageModelLoss,
) -> None:
    logits = torch.randn(2, 8, 100)
    targets = torch.randint(0, 100, (2, 8))

    loss = loss_fn(logits, targets)

    assert torch.isfinite(loss)


def test_loss_matches_pytorch_cross_entropy(
    loss_fn: CausalLanguageModelLoss,
) -> None:
    torch.manual_seed(42)

    logits = torch.randn(2, 8, 100)
    targets = torch.randint(0, 100, (2, 8))

    actual = loss_fn(logits, targets)

    expected = F.cross_entropy(
        logits.reshape(-1, 100),
        targets.reshape(-1),
    )

    assert torch.allclose(
        actual,
        expected,
    )


def test_loss_with_perfect_predictions(
    loss_fn: CausalLanguageModelLoss,
) -> None:
    batch_size = 2
    sequence_length = 4
    vocab_size = 10

    targets = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]
    )

    logits = torch.full(
        (
            batch_size,
            sequence_length,
            vocab_size,
        ),
        -100.0,
    )

    for batch in range(batch_size):
        for position in range(sequence_length):
            logits[
                batch,
                position,
                targets[batch, position],
            ] = 100.0

    loss = loss_fn(logits, targets)

    assert loss.item() < 1e-6


def test_loss_with_uniform_predictions(
    loss_fn: CausalLanguageModelLoss,
) -> None:
    vocab_size = 100

    logits = torch.zeros(
        2,
        8,
        vocab_size,
    )

    targets = torch.randint(
        0,
        vocab_size,
        (2, 8),
    )

    loss = loss_fn(logits, targets)

    expected = torch.log(
        torch.tensor(float(vocab_size))
    )

    assert torch.allclose(
        loss,
        expected,
        atol=1e-6,
    )


def test_ignore_index(
    loss_fn: CausalLanguageModelLoss,
) -> None:
    torch.manual_seed(42)

    logits = torch.randn(
        2,
        8,
        100,
    )

    targets = torch.randint(
        0,
        100,
        (2, 8),
    )

    targets[0, 0] = -100
    targets[1, 3] = -100

    actual = loss_fn(
        logits,
        targets,
    )

    valid_logits = logits.reshape(-1, 100)
    valid_targets = targets.reshape(-1)

    expected = F.cross_entropy(
        valid_logits,
        valid_targets,
        ignore_index=-100,
    )

    assert torch.allclose(
        actual,
        expected,
    )


def test_gradients_flow(
    loss_fn: CausalLanguageModelLoss,
) -> None:
    logits = torch.randn(
        2,
        8,
        100,
        requires_grad=True,
    )

    targets = torch.randint(
        0,
        100,
        (2, 8),
    )

    loss = loss_fn(
        logits,
        targets,
    )

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(
        logits.grad
    ).all()


def test_different_sequence_lengths(
    loss_fn: CausalLanguageModelLoss,
) -> None:
    for sequence_length in [1, 2, 8, 32]:
        logits = torch.randn(
            2,
            sequence_length,
            100,
        )

        targets = torch.randint(
            0,
            100,
            (2, sequence_length),
        )

        loss = loss_fn(
            logits,
            targets,
        )

        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_invalid_logits_dimensions(
    loss_fn: CausalLanguageModelLoss,
) -> None:
    logits = torch.randn(
        2,
        8,
    )

    targets = torch.randint(
        0,
        100,
        (2, 8),
    )

    with pytest.raises(ValueError):
        loss_fn(logits, targets)


def test_invalid_targets_dimensions(
    loss_fn: CausalLanguageModelLoss,
) -> None:
    logits = torch.randn(
        2,
        8,
        100,
    )

    targets = torch.randint(
        0,
        100,
        (2, 8, 1),
    )

    with pytest.raises(ValueError):
        loss_fn(logits, targets)


def test_mismatched_shapes(
    loss_fn: CausalLanguageModelLoss,
) -> None:
    logits = torch.randn(
        2,
        8,
        100,
    )

    targets = torch.randint(
        0,
        100,
        (2, 7),
    )

    with pytest.raises(ValueError):
        loss_fn(logits, targets)


def test_invalid_target_dtype(
    loss_fn: CausalLanguageModelLoss,
) -> None:
    logits = torch.randn(
        2,
        8,
        100,
    )

    targets = torch.randn(
        2,
        8,
    )

    with pytest.raises(ValueError):
        loss_fn(logits, targets)


def test_invalid_vocabulary_size(
    loss_fn: CausalLanguageModelLoss,
) -> None:
    logits = torch.randn(
        2,
        8,
        1,
    )

    targets = torch.zeros(
        2,
        8,
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        loss_fn(logits, targets)


def test_deterministic_loss(
    loss_fn: CausalLanguageModelLoss,
) -> None:
    torch.manual_seed(42)

    logits = torch.randn(
        2,
        8,
        100,
    )

    targets = torch.randint(
        0,
        100,
        (2, 8),
    )

    loss_a = loss_fn(
        logits,
        targets,
    )

    loss_b = loss_fn(
        logits,
        targets,
    )

    assert torch.equal(
        loss_a,
        loss_b,
    )