from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.mlp import SwiGLU
from src.model.config import ModelConfig


def create_config() -> ModelConfig:
    """Create the development SwiGLU configuration."""

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


def create_mlp() -> SwiGLU:
    """Create the development SwiGLU module."""

    return SwiGLU(create_config())


def test_mlp_output_shape() -> None:
    """SwiGLU output must preserve the input shape."""

    mlp = create_mlp()

    x = torch.randn(
        2,
        32,
        256,
    )

    output = mlp(x)

    assert output.shape == x.shape


def test_projection_shapes() -> None:
    """Verify gate, up, and down projection dimensions."""

    mlp = create_mlp()

    assert mlp.gate_proj.weight.shape == (
        768,
        256,
    )

    assert mlp.up_proj.weight.shape == (
        768,
        256,
    )

    assert mlp.down_proj.weight.shape == (
        256,
        768,
    )


def test_intermediate_size() -> None:
    """Verify the configured intermediate dimension."""

    mlp = create_mlp()

    assert mlp.hidden_size == 256
    assert mlp.intermediate_size == 768


def test_activation_is_silu() -> None:
    """SwiGLU must use SiLU as its gating activation."""

    mlp = create_mlp()

    assert isinstance(
        mlp.activation,
        torch.nn.SiLU,
    )


def test_gate_and_up_shapes() -> None:
    """Gate and up projections must produce matching tensors."""

    mlp = create_mlp()

    x = torch.randn(
        2,
        16,
        256,
    )

    gate = mlp.gate_proj(x)
    up = mlp.up_proj(x)

    assert gate.shape == (
        2,
        16,
        768,
    )

    assert up.shape == (
        2,
        16,
        768,
    )


def test_swiglu_computation() -> None:
    """Verify the complete SwiGLU computation."""

    mlp = create_mlp()

    x = torch.randn(
        2,
        8,
        256,
    )

    gate = mlp.activation(
        mlp.gate_proj(x)
    )

    up = mlp.up_proj(x)

    expected = mlp.down_proj(
        gate * up
    )

    output = mlp(x)

    assert torch.allclose(
        output,
        expected,
        atol=1e-6,
    )


def test_zero_input() -> None:
    """Zero input should produce zero output with zero biases."""

    mlp = create_mlp()

    x = torch.zeros(
        2,
        8,
        256,
    )

    output = mlp(x)

    assert torch.allclose(
        output,
        torch.zeros_like(output),
        atol=1e-6,
    )


def test_different_sequence_lengths() -> None:
    """SwiGLU must support different sequence lengths."""

    mlp = create_mlp()

    for sequence_length in (1, 8, 32, 128):
        x = torch.randn(
            2,
            sequence_length,
            256,
        )

        output = mlp(x)

        assert output.shape == x.shape


def test_dtype_preservation() -> None:
    """SwiGLU should preserve float32 output dtype."""

    mlp = create_mlp()

    x = torch.randn(
        2,
        16,
        256,
        dtype=torch.float32,
    )

    output = mlp(x)

    assert output.dtype == x.dtype


def test_invalid_dimensions() -> None:
    """SwiGLU must reject non-3D input tensors."""

    mlp = create_mlp()

    x = torch.randn(
        2,
        256,
    )

    with pytest.raises(ValueError):
        mlp(x)


def test_invalid_hidden_size() -> None:
    """SwiGLU must reject an incorrect hidden dimension."""

    mlp = create_mlp()

    x = torch.randn(
        2,
        16,
        128,
    )

    with pytest.raises(ValueError):
        mlp(x)


def test_deterministic_output() -> None:
    """Repeated inference must produce identical results."""

    mlp = create_mlp()

    x = torch.randn(
        2,
        16,
        256,
    )

    output_1 = mlp(x)
    output_2 = mlp(x)

    assert torch.allclose(
        output_1,
        output_2,
        atol=1e-6,
    )