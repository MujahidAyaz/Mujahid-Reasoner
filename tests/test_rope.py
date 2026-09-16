from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.rope import RotaryEmbedding


HEAD_DIMENSION = 32
MAX_SEQUENCE_LENGTH = 512


def create_rope() -> RotaryEmbedding:
    """Create the standard RoPE configuration."""

    return RotaryEmbedding(
        head_dimension=HEAD_DIMENSION,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        theta=10_000.0,
    )


def test_rope_output_shape() -> None:
    """Output shape must match input shape."""

    rope = create_rope()

    x = torch.randn(2, 8, 128, HEAD_DIMENSION)

    output = rope(x)

    assert output.shape == x.shape


def test_rope_preserves_dtype() -> None:
    """RoPE must preserve the input dtype."""

    rope = create_rope()

    x = torch.randn(
        2,
        8,
        128,
        HEAD_DIMENSION,
        dtype=torch.float32,
    )

    output = rope(x)

    assert output.dtype == x.dtype


def test_rope_cache_shape() -> None:
    """Verify precomputed cosine and sine cache shapes."""

    rope = create_rope()

    expected_shape = (
        MAX_SEQUENCE_LENGTH,
        HEAD_DIMENSION // 2,
    )

    assert rope.cos_cached.shape == expected_shape
    assert rope.sin_cached.shape == expected_shape


def test_rope_position_zero() -> None:
    """Position zero should apply an identity rotation."""

    rope = create_rope()

    x = torch.randn(
        2,
        8,
        1,
        HEAD_DIMENSION,
    )

    output = rope(x)

    assert torch.allclose(
        output,
        x,
        atol=1e-6,
    )


def test_rope_changes_nonzero_positions() -> None:
    """Nonzero positions should rotate the representation."""

    rope = create_rope()

    x = torch.randn(
        1,
        1,
        8,
        HEAD_DIMENSION,
    )

    output = rope(x)

    assert not torch.allclose(
        output[:, :, 1:],
        x[:, :, 1:],
    )


def test_position_offset() -> None:
    """Verify cached position offsets produce the correct rotation."""

    rope = create_rope()

    x = torch.randn(
        1,
        1,
        4,
        HEAD_DIMENSION,
    )

    offset_output = rope(
        x,
        position_offset=10,
    )

    full_output = rope(
        torch.cat(
            [
                torch.zeros(
                    1,
                    1,
                    10,
                    HEAD_DIMENSION,
                ),
                x,
            ],
            dim=2,
        )
    )

    assert torch.allclose(
        offset_output,
        full_output[:, :, 10:],
        atol=1e-6,
    )


def test_odd_head_dimension_rejected() -> None:
    """RoPE requires an even head dimension."""

    with pytest.raises(ValueError):
        RotaryEmbedding(
            head_dimension=31,
            max_sequence_length=512,
        )


def test_sequence_overflow_rejected() -> None:
    """Sequences longer than the configured cache must fail."""

    rope = create_rope()

    x = torch.randn(
        1,
        1,
        MAX_SEQUENCE_LENGTH + 1,
        HEAD_DIMENSION,
    )

    with pytest.raises(ValueError):
        rope(x)