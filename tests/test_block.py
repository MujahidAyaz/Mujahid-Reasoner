from __future__ import annotations

import torch
import pytest

from src.model.block import TransformerBlock
from src.model.cache import KVCache
from src.model.config import ModelConfig


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

    output, cache = block(x)

    assert output.shape == (2, 16, 256)
    assert cache is None


def test_residual_connection(
    block: TransformerBlock,
) -> None:
    torch.manual_seed(42)

    x = torch.randn(2, 8, 256)

    output, cache = block(x)

    assert output.shape == x.shape
    assert cache is None


def test_varying_sequence_lengths(
    block: TransformerBlock,
) -> None:
    for sequence_length in [1, 4, 16, 32]:
        x = torch.randn(2, sequence_length, 256)

        output, cache = block(x)

        assert output.shape == (
            2,
            sequence_length,
            256,
        )
        assert cache is None


def test_position_offset(
    block: TransformerBlock,
) -> None:
    x = torch.randn(2, 8, 256)

    output, cache = block(
        x,
        position_offset=10,
    )

    assert output.shape == (2, 8, 256)
    assert cache is None


def test_dtype_preservation(
    block: TransformerBlock,
) -> None:
    x = torch.randn(
        2,
        8,
        256,
        dtype=torch.float32,
    )

    output, cache = block(x)

    assert output.dtype == x.dtype
    assert cache is None


def test_deterministic_output(
    config: ModelConfig,
) -> None:
    torch.manual_seed(42)
    block_a = TransformerBlock(config)

    torch.manual_seed(42)
    block_b = TransformerBlock(config)

    x = torch.randn(2, 8, 256)

    output_a, cache_a = block_a(x)
    output_b, cache_b = block_b(x)

    assert torch.allclose(
        output_a,
        output_b,
    )

    assert cache_a is None
    assert cache_b is None


def test_gradients_flow(
    block: TransformerBlock,
) -> None:
    x = torch.randn(
        2,
        8,
        256,
        requires_grad=True,
    )

    output, cache = block(x)

    loss = output.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert cache is None


def test_cache_is_returned_when_enabled(
    block: TransformerBlock,
) -> None:
    x = torch.randn(2, 8, 256)

    output, cache = block(
        x,
        use_cache=True,
    )

    assert output.shape == (2, 8, 256)
    assert isinstance(cache, KVCache)
    assert cache.sequence_length == 8


def test_cached_forward_accepts_previous_cache(
    block: TransformerBlock,
) -> None:
    torch.manual_seed(42)

    prompt = torch.randn(2, 8, 256)
    next_token = torch.randn(2, 1, 256)

    _, cache = block(
        prompt,
        use_cache=True,
    )

    assert cache is not None

    output, updated_cache = block(
        next_token,
        position_offset=8,
        cache=cache,
        use_cache=True,
    )

    assert output.shape == (2, 1, 256)
    assert isinstance(updated_cache, KVCache)
    assert updated_cache.sequence_length == 9


def test_invalid_input_rank(
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


def test_cache_equivalence(
    config: ModelConfig,
) -> None:
    """
    Verify that cached token-by-token execution produces
    the same hidden states as normal full-context execution.
    """

    torch.manual_seed(42)

    block = TransformerBlock(config)
    block.eval()

    prompt = torch.randn(1, 4, 256)
    continuation = torch.randn(1, 3, 256)

    full_input = torch.cat(
        (prompt, continuation),
        dim=1,
    )

    full_output, _ = block(full_input)

    prompt_output, cache = block(
        prompt,
        use_cache=True,
    )

    assert cache is not None

    cached_outputs = [prompt_output]

    for step in range(continuation.size(1)):
        token = continuation[:, step : step + 1]

        token_output, cache = block(
            token,
            position_offset=4 + step,
            cache=cache,
            use_cache=True,
        )

        cached_outputs.append(token_output)

    cached_output = torch.cat(
        cached_outputs,
        dim=1,
    )

    assert torch.allclose(
        full_output,
        cached_output,
        atol=1e-5,
        rtol=1e-5,
    )