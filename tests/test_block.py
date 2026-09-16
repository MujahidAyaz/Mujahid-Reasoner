from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pytest
import torch

from src.model.attention import GroupedQueryAttention
from src.model.block import TransformerBlock
from src.model.config import ModelConfig
from src.model.mlp import SwiGLU
from src.model.norm import RMSNorm


@pytest.fixture
def config() -> ModelConfig:
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
def block(config: ModelConfig) -> TransformerBlock:
    torch.manual_seed(42)
    return TransformerBlock(config)


def test_output_shape(
    block: TransformerBlock,
) -> None:
    x = torch.randn(2, 16, 256)

    output = block(x)

    assert output.shape == (2, 16, 256)


def test_attention_module(
    block: TransformerBlock,
) -> None:
    assert isinstance(
        block.self_attn,
        GroupedQueryAttention,
    )


def test_mlp_module(
    block: TransformerBlock,
) -> None:
    assert isinstance(
        block.mlp,
        SwiGLU,
    )


def test_input_layernorm(
    block: TransformerBlock,
) -> None:
    assert isinstance(
        block.input_layernorm,
        RMSNorm,
    )


def test_post_attention_layernorm(
    block: TransformerBlock,
) -> None:
    assert isinstance(
        block.post_attention_layernorm,
        RMSNorm,
    )


def test_hidden_size(
    block: TransformerBlock,
) -> None:
    assert block.hidden_size == 256


def test_residual_connection(
    block: TransformerBlock,
) -> None:
    torch.manual_seed(42)

    x = torch.randn(2, 8, 256)

    output = block(x)

    assert output.shape == x.shape
    assert not torch.allclose(output, x)


def test_varying_sequence_lengths(
    block: TransformerBlock,
) -> None:
    for sequence_length in [1, 4, 16, 32]:
        x = torch.randn(2, sequence_length, 256)

        output = block(x)

        assert output.shape == (
            2,
            sequence_length,
            256,
        )


def test_position_offset(
    block: TransformerBlock,
) -> None:
    x = torch.randn(2, 8, 256)

    output = block(
        x,
        position_offset=10,
    )

    assert output.shape == (2, 8, 256)


def test_dtype_preservation(
    block: TransformerBlock,
) -> None:
    x = torch.randn(
        2,
        8,
        256,
        dtype=torch.float32,
    )

    output = block(x)

    assert output.dtype == x.dtype


def test_invalid_dimensions(
    block: TransformerBlock,
) -> None:
    x = torch.randn(2, 256)

    with pytest.raises(ValueError):
        block(x)


def test_invalid_hidden_size(
    block: TransformerBlock,
) -> None:
    x = torch.randn(2, 8, 128)

    with pytest.raises(ValueError):
        block(x)


def test_deterministic_output(
    config: ModelConfig,
) -> None:
    torch.manual_seed(42)
    block_a = TransformerBlock(config)

    torch.manual_seed(42)
    block_b = TransformerBlock(config)

    x = torch.randn(2, 8, 256)

    output_a = block_a(x)
    output_b = block_b(x)

    assert torch.allclose(
        output_a,
        output_b,
    )


def test_gradients_flow(
    block: TransformerBlock,
) -> None:
    x = torch.randn(
        2,
        8,
        256,
        requires_grad=True,
    )

    output = block(x)
    loss = output.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()