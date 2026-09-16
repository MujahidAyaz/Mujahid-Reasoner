from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.attention import GroupedQueryAttention
from src.model.config import ModelConfig


def create_config() -> ModelConfig:
    """Create the development GQA configuration."""

    return ModelConfig(
        name="mujahid-reasoner-test",
        vocab_size=32_000,
        hidden_size=256,
        num_layers=6,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=768,
        max_sequence_length=512,
        dropout=0.0,
        rope_theta=10_000.0,
        norm_type="rmsnorm",
        activation="swiglu",
        position_embedding="rope",
        attention_type="gqa",
        tie_word_embeddings=True,
        initializer_range=0.02,
    )


def create_attention() -> GroupedQueryAttention:
    """Create the development GQA attention module."""

    return GroupedQueryAttention(create_config())


def test_attention_output_shape() -> None:
    """Attention output must preserve the input shape."""

    attention = create_attention()

    x = torch.randn(
        2,
        32,
        256,
    )

    output = attention(x)

    assert output.shape == x.shape


def test_attention_projection_shapes() -> None:
    """Verify Q, K, V, and output projection dimensions."""

    attention = create_attention()

    assert attention.q_proj.weight.shape == (
        256,
        256,
    )

    assert attention.k_proj.weight.shape == (
        128,
        256,
    )

    assert attention.v_proj.weight.shape == (
        128,
        256,
    )

    assert attention.o_proj.weight.shape == (
        256,
        256,
    )


def test_gqa_head_configuration() -> None:
    """Verify that query heads and KV heads use GQA."""

    attention = create_attention()

    assert attention.num_attention_heads == 8
    assert attention.num_key_value_heads == 4
    assert attention.kv_group_size == 2
    assert attention.head_dimension == 32


def test_query_reshape() -> None:
    """Query projection must produce the expected head layout."""

    attention = create_attention()

    x = torch.randn(
        2,
        16,
        256,
    )

    query = attention.q_proj(x)
    query = attention._reshape_query(query)

    assert query.shape == (
        2,
        8,
        16,
        32,
    )


def test_key_value_reshape() -> None:
    """Key/value projections must use fewer KV heads."""

    attention = create_attention()

    x = torch.randn(
        2,
        16,
        256,
    )

    key = attention.k_proj(x)
    value = attention.v_proj(x)

    key = attention._reshape_key_value(key)
    value = attention._reshape_key_value(value)

    assert key.shape == (
        2,
        4,
        16,
        32,
    )

    assert value.shape == (
        2,
        4,
        16,
        32,
    )


def test_key_value_head_repetition() -> None:
    """Each KV head must be repeated according to the GQA ratio."""

    attention = create_attention()

    x = torch.randn(
        2,
        4,
        16,
        32,
    )

    repeated = attention._repeat_key_value(x)

    assert repeated.shape == (
        2,
        8,
        16,
        32,
    )

    assert torch.allclose(
        repeated[:, 0],
        x[:, 0],
    )

    assert torch.allclose(
        repeated[:, 1],
        x[:, 0],
    )

    assert torch.allclose(
        repeated[:, 2],
        x[:, 1],
    )

    assert torch.allclose(
        repeated[:, 3],
        x[:, 1],
    )


def test_causal_attention() -> None:
    """
    Verify the causal mask prevents attention to future tokens.

    The causal mask must contain:
        True  → current/past token
        False → future token
    """

    attention = create_attention()

    mask = attention.causal_mask[:4, :4]

    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ],
        dtype=torch.bool,
    )

    assert torch.equal(
        mask,
        expected,
    )


def test_position_offset() -> None:
    """Attention must support non-zero RoPE position offsets."""

    attention = create_attention()

    x = torch.randn(
        2,
        8,
        256,
    )

    output = attention(
        x,
        position_offset=10,
    )

    assert output.shape == x.shape


def test_different_sequence_lengths() -> None:
    """Attention must support different valid sequence lengths."""

    attention = create_attention()

    for sequence_length in (1, 8, 32, 128):
        x = torch.randn(
            2,
            sequence_length,
            256,
        )

        output = attention(x)

        assert output.shape == x.shape


def test_attention_preserves_dtype() -> None:
    """Attention output should preserve float32 dtype."""

    attention = create_attention()

    x = torch.randn(
        2,
        16,
        256,
        dtype=torch.float32,
    )

    output = attention(x)

    assert output.dtype == x.dtype


def test_invalid_input_dimensions() -> None:
    """Attention must reject tensors that are not 3-dimensional."""

    attention = create_attention()

    x = torch.randn(
        2,
        16,
    )

    with pytest.raises(ValueError):
        attention(x)


def test_invalid_hidden_size() -> None:
    """Attention must reject an incorrect hidden dimension."""

    attention = create_attention()

    x = torch.randn(
        2,
        16,
        128,
    )

    with pytest.raises(ValueError):
        attention(x)


def test_sequence_overflow() -> None:
    """Attention must reject sequences beyond the configured context."""

    attention = create_attention()

    x = torch.randn(
        1,
        513,
        256,
    )

    with pytest.raises(ValueError):
        attention(x)


def test_attention_is_deterministic() -> None:
    """With dropout disabled, repeated inference must be deterministic."""

    attention = create_attention()

    x = torch.randn(
        2,
        16,
        256,
    )

    output_1 = attention(x)
    output_2 = attention(x)

    assert torch.allclose(
        output_1,
        output_2,
        atol=1e-6,
    )