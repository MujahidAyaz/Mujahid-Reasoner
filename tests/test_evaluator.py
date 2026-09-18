from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.evaluator import (
    EvaluationResult,
    ModelEvaluator,
)


class DummyModel(torch.nn.Module):
    """Small deterministic model for evaluator tests."""

    def __init__(self, vocab_size: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape

        logits = torch.zeros(
            batch_size,
            sequence_length,
            self.vocab_size,
            dtype=torch.float32,
            device=input_ids.device,
        )

        # Always predict token 0 with high confidence.
        logits[..., 0] = 10.0

        return logits


def test_evaluation_result_fields() -> None:
    result = EvaluationResult(
        loss=2.5,
        perplexity=math.exp(2.5),
        tokens=100,
        batches=5,
    )

    assert result.loss == 2.5
    assert result.perplexity == math.exp(2.5)
    assert result.tokens == 100
    assert result.batches == 5


def test_evaluate_tuple_batch() -> None:
    model = DummyModel()

    input_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ],
        dtype=torch.long,
    )

    target_ids = torch.zeros_like(input_ids)

    dataset = TensorDataset(input_ids, target_ids)
    dataloader = DataLoader(dataset, batch_size=2)

    evaluator = ModelEvaluator(model)

    result = evaluator.evaluate(dataloader)

    assert isinstance(result, EvaluationResult)
    assert result.tokens == 8
    assert result.batches == 1
    assert result.loss > 0.0
    assert result.perplexity > 1.0


def test_evaluate_multiple_batches() -> None:
    model = DummyModel()

    input_ids = torch.tensor(
        [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ],
        dtype=torch.long,
    )

    target_ids = torch.zeros_like(input_ids)

    dataset = TensorDataset(input_ids, target_ids)
    dataloader = DataLoader(dataset, batch_size=2)

    evaluator = ModelEvaluator(model)

    result = evaluator.evaluate(dataloader)

    assert result.tokens == 8
    assert result.batches == 2


def test_evaluate_dictionary_target_ids() -> None:
    model = DummyModel()

    batch = {
        "input_ids": torch.tensor(
            [[1, 2, 3, 4]],
            dtype=torch.long,
        ),
        "target_ids": torch.zeros(
            (1, 4),
            dtype=torch.long,
        ),
    }

    dataloader = [batch]

    evaluator = ModelEvaluator(model)

    result = evaluator.evaluate(dataloader)

    assert result.tokens == 4
    assert result.batches == 1


def test_evaluate_dictionary_labels() -> None:
    model = DummyModel()

    batch = {
        "input_ids": torch.tensor(
            [[1, 2, 3, 4]],
            dtype=torch.long,
        ),
        "labels": torch.zeros(
            (1, 4),
            dtype=torch.long,
        ),
    }

    dataloader = [batch]

    evaluator = ModelEvaluator(model)

    result = evaluator.evaluate(dataloader)

    assert result.tokens == 4
    assert result.batches == 1


def test_empty_dataloader_raises() -> None:
    model = DummyModel()

    dataloader = []

    evaluator = ModelEvaluator(model)

    with pytest.raises(
        RuntimeError,
        match="No evaluation tokens were processed",
    ):
        evaluator.evaluate(dataloader)


def test_invalid_dictionary_batch_raises() -> None:
    model = DummyModel()

    batch = {
        "input_ids": torch.tensor(
            [[1, 2]],
            dtype=torch.long,
        )
    }

    evaluator = ModelEvaluator(model)

    with pytest.raises(
        KeyError,
        match="target_ids.*labels",
    ):
        evaluator.evaluate([batch])


def test_invalid_tuple_batch_raises() -> None:
    model = DummyModel()

    batch = (
        torch.tensor([[1, 2]], dtype=torch.long),
    )

    evaluator = ModelEvaluator(model)

    with pytest.raises(
        ValueError,
        match="input and target",
    ):
        evaluator.evaluate([batch])


def test_invalid_batch_type_raises() -> None:
    model = DummyModel()

    evaluator = ModelEvaluator(model)

    with pytest.raises(
        TypeError,
        match="Unsupported batch format",
    ):
        evaluator.evaluate([123])