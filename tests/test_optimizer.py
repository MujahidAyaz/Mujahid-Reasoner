from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
from torch import nn
from torch.optim import AdamW

from src.training.optimizer import create_adamw_optimizer


class OptimizerTestModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.linear = nn.Linear(
            16,
            8,
            bias=True,
        )

        self.norm = nn.LayerNorm(8)

        self.embedding = nn.Embedding(
            100,
            16,
        )


@pytest.fixture
def model() -> OptimizerTestModel:
    torch.manual_seed(42)
    return OptimizerTestModel()


def test_returns_adamw(
    model: OptimizerTestModel,
) -> None:
    optimizer = create_adamw_optimizer(model)

    assert isinstance(
        optimizer,
        AdamW,
    )


def test_learning_rate(
    model: OptimizerTestModel,
) -> None:
    optimizer = create_adamw_optimizer(
        model,
        learning_rate=1e-4,
    )

    for group in optimizer.param_groups:
        assert group["lr"] == 1e-4


def test_weight_decay_configuration(
    model: OptimizerTestModel,
) -> None:
    optimizer = create_adamw_optimizer(
        model,
        weight_decay=0.1,
    )

    decay_values = {
        group["weight_decay"]
        for group in optimizer.param_groups
    }

    assert 0.1 in decay_values
    assert 0.0 in decay_values


def test_matrix_parameters_use_weight_decay(
    model: OptimizerTestModel,
) -> None:
    optimizer = create_adamw_optimizer(
        model,
        weight_decay=0.1,
    )

    decay_group = next(
        group
        for group in optimizer.param_groups
        if group["weight_decay"] == 0.1
    )

    decay_parameters = set(
        decay_group["params"]
    )

    assert (
        model.linear.weight
        in decay_parameters
    )

    assert (
        model.embedding.weight
        in decay_parameters
    )


def test_one_dimensional_parameters_skip_weight_decay(
    model: OptimizerTestModel,
) -> None:
    optimizer = create_adamw_optimizer(
        model,
        weight_decay=0.1,
    )

    no_decay_group = next(
        group
        for group in optimizer.param_groups
        if group["weight_decay"] == 0.0
    )

    no_decay_parameters = set(
        no_decay_group["params"]
    )

    assert (
        model.linear.bias
        in no_decay_parameters
    )

    assert (
        model.norm.weight
        in no_decay_parameters
    )

    assert (
        model.norm.bias
        in no_decay_parameters
    )


def test_all_trainable_parameters_are_in_optimizer(
    model: OptimizerTestModel,
) -> None:
    optimizer = create_adamw_optimizer(model)

    optimizer_parameters = {
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    model_parameters = {
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    }

    assert optimizer_parameters == model_parameters


def test_frozen_parameters_are_excluded(
    model: OptimizerTestModel,
) -> None:
    model.linear.weight.requires_grad = False

    optimizer = create_adamw_optimizer(model)

    optimizer_parameters = {
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert (
        model.linear.weight
        not in optimizer_parameters
    )


def test_optimizer_updates_parameters(
    model: OptimizerTestModel,
) -> None:
    optimizer = create_adamw_optimizer(
        model,
        learning_rate=1e-3,
    )

    input_tensor = torch.randn(4, 16)

    output = model.linear(input_tensor)
    loss = output.pow(2).mean()

    loss.backward()

    before = model.linear.weight.detach().clone()

    optimizer.step()

    after = model.linear.weight.detach()

    assert not torch.equal(
        before,
        after,
    )


def test_gradients_are_cleared(
    model: OptimizerTestModel,
) -> None:
    optimizer = create_adamw_optimizer(model)

    input_tensor = torch.randn(4, 16)

    output = model.linear(input_tensor)
    loss = output.mean()

    loss.backward()

    assert (
        model.linear.weight.grad
        is not None
    )

    optimizer.zero_grad()

    assert (
        model.linear.weight.grad
        is None
    )


def test_custom_betas(
    model: OptimizerTestModel,
) -> None:
    optimizer = create_adamw_optimizer(
        model,
        betas=(0.9, 0.98),
    )

    for group in optimizer.param_groups:
        assert group["betas"] == (
            0.9,
            0.98,
        )


def test_custom_epsilon(
    model: OptimizerTestModel,
) -> None:
    optimizer = create_adamw_optimizer(
        model,
        eps=1e-7,
    )

    for group in optimizer.param_groups:
        assert group["eps"] == 1e-7


def test_invalid_learning_rate(
    model: OptimizerTestModel,
) -> None:
    with pytest.raises(ValueError):
        create_adamw_optimizer(
            model,
            learning_rate=0.0,
        )


def test_invalid_weight_decay(
    model: OptimizerTestModel,
) -> None:
    with pytest.raises(ValueError):
        create_adamw_optimizer(
            model,
            weight_decay=-0.1,
        )


def test_invalid_betas(
    model: OptimizerTestModel,
) -> None:
    with pytest.raises(ValueError):
        create_adamw_optimizer(
            model,
            betas=(1.0, 0.95),
        )


def test_invalid_epsilon(
    model: OptimizerTestModel,
) -> None:
    with pytest.raises(ValueError):
        create_adamw_optimizer(
            model,
            eps=0.0,
        )