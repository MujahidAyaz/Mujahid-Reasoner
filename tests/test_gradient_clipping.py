from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.gradient_clipping import clip_gradients


class GradientTestModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.fixture
def model() -> GradientTestModel:
    torch.manual_seed(42)

    model = GradientTestModel()

    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 2)

    outputs = model(inputs)
    loss = nn.functional.mse_loss(outputs, targets)

    loss.backward()

    return model


def _gradient_norm(
    model: nn.Module,
    norm_type: float = 2.0,
) -> float:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    if not gradients:
        return 0.0

    total = torch.stack(
        [
            gradient.detach().norm(norm_type)
            for gradient in gradients
        ]
    )

    return float(total.norm(norm_type).item())


def test_returns_gradient_norm(
    model: GradientTestModel,
) -> None:
    norm = clip_gradients(
        model,
        max_norm=1.0,
    )

    assert isinstance(norm, float)
    assert norm > 0.0


def test_clips_large_gradients(
    model: GradientTestModel,
) -> None:
    original_norm = _gradient_norm(model)

    assert original_norm > 1.0

    clip_gradients(
        model,
        max_norm=1.0,
    )

    clipped_norm = _gradient_norm(model)

    assert clipped_norm <= 1.0 + 1e-6


def test_does_not_change_small_gradients(
    model: GradientTestModel,
) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.fill_(0.01)

    original_norm = _gradient_norm(model)

    returned_norm = clip_gradients(
        model,
        max_norm=1.0,
    )

    clipped_norm = _gradient_norm(model)

    assert clipped_norm == pytest.approx(
        original_norm,
        rel=1e-6,
    )

    assert returned_norm == pytest.approx(
        original_norm,
        rel=1e-6,
    )


def test_returned_norm_is_pre_clipping_norm(
    model: GradientTestModel,
) -> None:
    original_norm = _gradient_norm(model)

    returned_norm = clip_gradients(
        model,
        max_norm=1.0,
    )

    assert returned_norm == pytest.approx(
        original_norm,
        rel=1e-6,
    )


def test_no_gradients_returns_zero() -> None:
    model = GradientTestModel()

    norm = clip_gradients(
        model,
        max_norm=1.0,
    )

    assert norm == 0.0


def test_frozen_parameters_are_ignored(
    model: GradientTestModel,
) -> None:
    model.linear.weight.requires_grad = False

    # Remove the frozen parameter's gradient.
    model.linear.weight.grad = None

    norm = clip_gradients(
        model,
        max_norm=1.0,
    )

    assert isinstance(norm, float)
    assert norm >= 0.0


def test_custom_max_norm(
    model: GradientTestModel,
) -> None:
    max_norm = 0.5

    clip_gradients(
        model,
        max_norm=max_norm,
    )

    clipped_norm = _gradient_norm(model)

    assert clipped_norm <= max_norm + 1e-6


def test_custom_norm_type(
    model: GradientTestModel,
) -> None:
    norm = clip_gradients(
        model,
        max_norm=1.0,
        norm_type=1.0,
    )

    assert isinstance(norm, float)
    assert norm >= 0.0


def test_invalid_max_norm(
    model: GradientTestModel,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_norm",
    ):
        clip_gradients(
            model,
            max_norm=0.0,
        )


def test_negative_max_norm(
    model: GradientTestModel,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_norm",
    ):
        clip_gradients(
            model,
            max_norm=-1.0,
        )


def test_invalid_norm_type(
    model: GradientTestModel,
) -> None:
    with pytest.raises(
        ValueError,
        match="norm_type",
    ):
        clip_gradients(
            model,
            max_norm=1.0,
            norm_type=0.0,
        )


def test_negative_norm_type(
    model: GradientTestModel,
) -> None:
    with pytest.raises(
        ValueError,
        match="norm_type",
    ):
        clip_gradients(
            model,
            max_norm=1.0,
            norm_type=-2.0,
        )