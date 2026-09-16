from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch

from src.model.model import MujahidReasonerModel
from src.model.norm import RMSNorm
from src.model.block import TransformerBlock


@pytest.fixture
def config():
    from src.model.config import ModelConfig

    return ModelConfig(
        name="mujahid-reasoner-test",
        vocab_size=32000,
        hidden_size=256,
        num_layers=6,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=768,
        max_sequence_length=512,
        dropout=0.0,
        rope_theta=10000.0,
        norm_type="rmsnorm",
        activation="swiglu",
        position_embedding="rope",
        attention_type="gqa",
        tie_word_embeddings=True,
        initializer_range=0.02,
    )


@pytest.fixture
def model(config):
    torch.manual_seed(42)
    return MujahidReasonerModel(config)


def test_model_output_shape(
    model,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (2, 16),
    )

    logits = model(input_ids)

    assert logits.shape == (
        2,
        16,
        32000,
    )


def test_embedding_shape(
    model,
) -> None:
    assert model.token_embedding.weight.shape == (
        32000,
        256,
    )


def test_number_of_transformer_layers(
    model,
) -> None:
    assert len(model.layers) == 6

    for layer in model.layers:
        assert isinstance(
            layer,
            TransformerBlock,
        )


def test_final_layernorm(
    model,
) -> None:
    assert isinstance(
        model.final_layernorm,
        RMSNorm,
    )


def test_lm_head_shape(
    model,
) -> None:
    assert model.lm_head.weight.shape == (
        32000,
        256,
    )


def test_weight_tying(
    model,
) -> None:
    assert (
        model.lm_head.weight.data_ptr()
        == model.token_embedding.weight.data_ptr()
    )


def test_parameter_count(
    model,
) -> None:
    count = model.parameter_count()

    assert count > 0
    assert count == sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_parameter_count_all(
    model,
) -> None:
    trainable = model.parameter_count(
        trainable_only=True,
    )

    total = model.parameter_count(
        trainable_only=False,
    )

    assert trainable == total


def test_varying_sequence_lengths(
    model,
) -> None:
    for sequence_length in [1, 4, 16, 32]:
        input_ids = torch.randint(
            0,
            32000,
            (2, sequence_length),
        )

        logits = model(input_ids)

        assert logits.shape == (
            2,
            sequence_length,
            32000,
        )


def test_position_offset(
    model,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (2, 8),
    )

    logits = model(
        input_ids,
        position_offset=10,
    )

    assert logits.shape == (
        2,
        8,
        32000,
    )


def test_invalid_input_dimensions(
    model,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (2, 8, 1),
    )

    with pytest.raises(ValueError):
        model(input_ids)


def test_invalid_sequence_length(
    model,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (2, 513),
    )

    with pytest.raises(ValueError):
        model(input_ids)


def test_invalid_position_offset(
    model,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (2, 8),
    )

    with pytest.raises(ValueError):
        model(
            input_ids,
            position_offset=510,
        )


def test_invalid_input_dtype(
    model,
) -> None:
    input_ids = torch.randn(
        2,
        8,
    )

    with pytest.raises(ValueError):
        model(input_ids)


def test_negative_token_id(
    model,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (2, 8),
    )

    input_ids[0, 0] = -1

    with pytest.raises(ValueError):
        model(input_ids)


def test_out_of_vocabulary_token_id(
    model,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (2, 8),
    )

    input_ids[0, 0] = 32000

    with pytest.raises(ValueError):
        model(input_ids)


def test_dtype_preservation(
    model,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (2, 8),
    )

    logits = model(input_ids)

    assert logits.dtype == torch.float32


def test_deterministic_output(
    config,
) -> None:
    torch.manual_seed(42)
    model_a = MujahidReasonerModel(config)

    torch.manual_seed(42)
    model_b = MujahidReasonerModel(config)

    input_ids = torch.randint(
        0,
        32000,
        (2, 8),
    )

    logits_a = model_a(input_ids)
    logits_b = model_b(input_ids)

    assert torch.allclose(
        logits_a,
        logits_b,
    )


def test_gradients_flow(
    model,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (2, 8),
    )

    logits = model(input_ids)
    loss = logits.mean()

    loss.backward()

    assert (
        model.token_embedding.weight.grad
        is not None
    )

    assert (
        model.final_layernorm.weight.grad
        is not None
    )


def test_forward_produces_finite_values(
    model,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (2, 8),
    )

    logits = model(input_ids)

    assert torch.isfinite(logits).all()