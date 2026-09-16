from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.norm import RMSNorm


def test_rmsnorm_output_shape() -> None:
    """Output shape must match input shape."""

    norm = RMSNorm(hidden_size=256)

    x = torch.randn(2, 512, 256)

    output = norm(x)

    assert output.shape == x.shape


def test_rmsnorm_parameter_shape() -> None:
    """RMSNorm must have one learnable weight per hidden dimension."""

    norm = RMSNorm(hidden_size=256)

    assert norm.weight.shape == (256,)
    assert isinstance(norm.weight, torch.nn.Parameter)


def test_rmsnorm_normalizes_rms() -> None:
    """Verify normalized activations have approximately unit RMS."""

    norm = RMSNorm(hidden_size=256)

    x = torch.randn(2, 512, 256)

    output = norm(x)

    rms = torch.sqrt(
        output.pow(2).mean(dim=-1)
    )

    assert torch.allclose(
        rms,
        torch.ones_like(rms),
        atol=1e-5,
    )


def test_rmsnorm_preserves_dtype() -> None:
    """Verify output dtype matches input dtype."""

    norm = RMSNorm(hidden_size=256)

    x = torch.randn(
        2,
        512,
        256,
        dtype=torch.float32,
    )

    output = norm(x)

    assert output.dtype == x.dtype


def test_invalid_hidden_size() -> None:
    """Verify invalid hidden dimensions are rejected."""

    with pytest.raises(ValueError):
        RMSNorm(hidden_size=0)


def test_invalid_epsilon() -> None:
    """Verify invalid epsilon is rejected."""

    with pytest.raises(ValueError):
        RMSNorm(
            hidden_size=256,
            eps=0.0,
        )