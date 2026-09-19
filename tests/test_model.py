from __future__ import annotations

import pytest
import torch

from src.model.cache import LayerKVCache
from src.model.config import ModelConfig
from src.model.model import MujahidReasonerModel


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
def model(config: ModelConfig) -> MujahidReasonerModel:
    torch.manual_seed(42)
    return MujahidReasonerModel(config)


def test_model_output_shape(
    model: MujahidReasonerModel,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (2, 16),
    )

    logits, cache = model(input_ids)

    assert logits.shape == (
        2,
        16,
        32000,
    )
    assert cache is None


def test_parameter_count(
    model: MujahidReasonerModel,
) -> None:
    assert model.parameter_count() > 0
    assert (
        model.parameter_count()
        == model.parameter_count(trainable_only=False)
    )


def test_weight_tying(
    model: MujahidReasonerModel,
) -> None:
    assert (
        model.token_embedding.weight
        is model.lm_head.weight
    )


def test_varying_sequence_lengths(
    model: MujahidReasonerModel,
) -> None:
    for sequence_length in [1, 4, 16, 32]:
        input_ids = torch.randint(
            0,
            32000,
            (2, sequence_length),
        )

        logits, cache = model(input_ids)

        assert logits.shape == (
            2,
            sequence_length,
            32000,
        )
        assert cache is None


def test_position_offset(
    model: MujahidReasonerModel,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (2, 8),
    )

    logits, cache = model(
        input_ids,
        position_offset=10,
    )

    assert logits.shape == (
        2,
        8,
        32000,
    )
    assert cache is None


def test_invalid_input_rank(
    model: MujahidReasonerModel,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (8,),
    )

    with pytest.raises(ValueError):
        model(input_ids)


def test_empty_sequence(
    model: MujahidReasonerModel,
) -> None:
    input_ids = torch.empty(
        (2, 0),
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        model(input_ids)


def test_invalid_position_offset(
    model: MujahidReasonerModel,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (2, 8),
    )

    with pytest.raises(ValueError):
        model(
            input_ids,
            position_offset=-1,
        )


def test_sequence_exceeds_context(
    model: MujahidReasonerModel,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (2, 513),
    )

    with pytest.raises(ValueError):
        model(input_ids)


def test_invalid_input_dtype(
    model: MujahidReasonerModel,
) -> None:
    input_ids = torch.randn(2, 8)

    with pytest.raises(ValueError):
        model(input_ids)


def test_negative_token_id(
    model: MujahidReasonerModel,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (2, 8),
    )

    input_ids[0, 0] = -1

    with pytest.raises(ValueError):
        model(input_ids)


def test_token_id_out_of_range(
    model: MujahidReasonerModel,
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
    model: MujahidReasonerModel,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (2, 8),
    )

    logits, cache = model(input_ids)

    assert logits.dtype == torch.float32
    assert cache is None


def test_deterministic_output(
    config: ModelConfig,
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

    logits_a, cache_a = model_a(input_ids)
    logits_b, cache_b = model_b(input_ids)

    assert torch.allclose(
        logits_a,
        logits_b,
    )

    assert cache_a is None
    assert cache_b is None


def test_gradients_flow(
    model: MujahidReasonerModel,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (2, 8),
    )

    logits, cache = model(input_ids)

    loss = logits.mean()
    loss.backward()

    assert cache is None

    assert (
        model.token_embedding.weight.grad
        is not None
    )

    assert torch.isfinite(
        model.token_embedding.weight.grad
    ).all()


def test_forward_produces_finite_values(
    model: MujahidReasonerModel,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (2, 8),
    )

    logits, cache = model(input_ids)

    assert torch.isfinite(logits).all()
    assert cache is None


def test_cache_is_returned(
    model: MujahidReasonerModel,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (2, 8),
    )

    logits, cache = model(
        input_ids,
        use_cache=True,
    )

    assert logits.shape == (
        2,
        8,
        32000,
    )

    assert isinstance(
        cache,
        LayerKVCache,
    )

    assert len(cache) == 6

    for layer_cache in cache.layers:
        assert layer_cache is not None
        assert layer_cache.sequence_length == 8


def test_cache_batch_size(
    model: MujahidReasonerModel,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (3, 8),
    )

    _, cache = model(
        input_ids,
        use_cache=True,
    )

    assert cache is not None

    for layer_cache in cache.layers:
        assert layer_cache is not None
        assert layer_cache.key.size(0) == 3
        assert layer_cache.value.size(0) == 3


def test_cache_sequence_growth(
    model: MujahidReasonerModel,
) -> None:
    torch.manual_seed(42)

    prompt = torch.randint(
        0,
        32000,
        (1, 8),
    )

    next_token = torch.randint(
        0,
        32000,
        (1, 1),
    )

    _, cache = model(
        prompt,
        use_cache=True,
    )

    assert cache is not None

    _, updated_cache = model(
        next_token,
        position_offset=8,
        cache=cache,
        use_cache=True,
    )

    assert updated_cache is not None

    for layer_cache in updated_cache.layers:
        assert layer_cache is not None
        assert layer_cache.sequence_length == 9


def test_full_model_cache_equivalence(
    model: MujahidReasonerModel,
) -> None:
    """
    Verify that cached token-by-token decoding produces
    the same logits as normal full-context execution.
    """

    model.eval()

    torch.manual_seed(123)

    prompt = torch.randint(
        0,
        32000,
        (1, 4),
    )

    continuation = torch.randint(
        0,
        32000,
        (1, 3),
    )

    full_input = torch.cat(
        (
            prompt,
            continuation,
        ),
        dim=1,
    )

    full_logits, _ = model(full_input)

    prompt_logits, cache = model(
        prompt,
        use_cache=True,
    )

    assert cache is not None

    cached_logits = [
        prompt_logits,
    ]

    for step in range(
        continuation.size(1)
    ):
        token = continuation[
            :,
            step : step + 1,
        ]

        token_logits, cache = model(
            token,
            position_offset=4 + step,
            cache=cache,
            use_cache=True,
        )

        cached_logits.append(
            token_logits
        )

    cached_logits = torch.cat(
        cached_logits,
        dim=1,
    )

    assert cached_logits.shape == full_logits.shape

    assert torch.allclose(
        full_logits,
        cached_logits,
        atol=1e-5,
        rtol=1e-5,
    )


def test_cache_rejects_wrong_position(
    model: MujahidReasonerModel,
) -> None:
    input_ids = torch.randint(
        0,
        32000,
        (1, 8),
    )

    _, cache = model(
        input_ids,
        use_cache=True,
    )

    assert cache is not None

    next_token = torch.randint(
        0,
        32000,
        (1, 1),
    )

    with pytest.raises(ValueError):
        model(
            next_token,
            position_offset=7,
            cache=cache,
            use_cache=True,
        )


# ---------------------------------------------------------------------------
# Gradient Checkpointing Tests
# ---------------------------------------------------------------------------


def test_gradient_checkpointing_state(
    model: MujahidReasonerModel,
) -> None:
    assert not model.is_gradient_checkpointing_enabled()

    model.enable_gradient_checkpointing()

    assert model.is_gradient_checkpointing_enabled()

    model.disable_gradient_checkpointing()

    assert not model.is_gradient_checkpointing_enabled()


def test_gradient_checkpointing_output_equivalence(
    config: ModelConfig,
) -> None:
    """
    Gradient checkpointing must not change the forward result.
    """

    torch.manual_seed(123)

    model_normal = MujahidReasonerModel(config)

    torch.manual_seed(123)

    model_checkpointed = MujahidReasonerModel(config)

    model_normal.train()
    model_checkpointed.train()

    model_checkpointed.enable_gradient_checkpointing()

    input_ids = torch.randint(
        0,
        32000,
        (2, 8),
    )

    normal_logits, normal_cache = model_normal(
        input_ids,
        use_cache=False,
    )

    checkpointed_logits, checkpointed_cache = (
        model_checkpointed(
            input_ids,
            use_cache=False,
        )
    )

    assert normal_cache is None
    assert checkpointed_cache is None

    assert torch.allclose(
        normal_logits,
        checkpointed_logits,
        atol=1e-5,
        rtol=1e-5,
    )


def test_gradient_checkpointing_backward(
    model: MujahidReasonerModel,
) -> None:
    """
    Verify gradients propagate through checkpointed layers.
    """

    model.train()
    model.enable_gradient_checkpointing()

    input_ids = torch.randint(
        0,
        32000,
        (2, 8),
    )

    logits, cache = model(
        input_ids,
        use_cache=False,
    )

    loss = logits.mean()

    loss.backward()

    assert cache is None

    embedding_gradient = (
        model.token_embedding.weight.grad
    )

    assert embedding_gradient is not None

    assert torch.isfinite(
        embedding_gradient
    ).all()

    assert (
        embedding_gradient.abs().sum().item()
        > 0.0
    )


def test_gradient_checkpointing_is_training_only(
    model: MujahidReasonerModel,
) -> None:
    """
    Verify the externally observable training/evaluation behavior.

    Training with checkpointing must produce valid gradients.
    Evaluation with checkpointing enabled must still produce
    valid inference output without requiring gradients.
    """

    input_ids = torch.randint(
        0,
        32000,
        (2, 8),
    )

    model.train()
    model.enable_gradient_checkpointing()

    train_logits, train_cache = model(
        input_ids,
        use_cache=False,
    )

    assert train_logits.shape == (
        2,
        8,
        32000,
    )

    assert train_cache is None
    assert train_logits.requires_grad

    train_loss = train_logits.mean()
    train_loss.backward()

    assert (
        model.token_embedding.weight.grad
        is not None
    )

    model.zero_grad(set_to_none=True)

    model.eval()

    with torch.no_grad():
        eval_logits, eval_cache = model(
            input_ids,
            use_cache=False,
        )

    assert eval_logits.shape == (
        2,
        8,
        32000,
    )

    assert eval_cache is None
    assert not eval_logits.requires_grad

    assert torch.isfinite(eval_logits).all()


def test_gradient_checkpointing_with_kv_cache(
    model: MujahidReasonerModel,
) -> None:
    """
    Enabling gradient checkpointing must not break KV-cache
    inference.
    """

    model.train()
    model.enable_gradient_checkpointing()

    input_ids = torch.randint(
        0,
        32000,
        (2, 8),
    )

    logits, cache = model(
        input_ids,
        use_cache=True,
    )

    assert logits.shape == (
        2,
        8,
        32000,
    )

    assert isinstance(
        cache,
        LayerKVCache,
    )

    assert len(cache) == 6

    for layer_cache in cache.layers:
        assert layer_cache is not None
        assert layer_cache.sequence_length == 8


def test_gradient_checkpointing_cache_growth(
    model: MujahidReasonerModel,
) -> None:
    """
    KV-cache continuation must remain functional when
    checkpointing is enabled.
    """

    model.train()
    model.enable_gradient_checkpointing()

    prompt = torch.randint(
        0,
        32000,
        (1, 4),
    )

    continuation = torch.randint(
        0,
        32000,
        (1, 1),
    )

    _, cache = model(
        prompt,
        use_cache=True,
    )

    assert cache is not None

    _, updated_cache = model(
        continuation,
        position_offset=4,
        cache=cache,
        use_cache=True,
    )

    assert updated_cache is not None

    for layer_cache in updated_cache.layers:
        assert layer_cache is not None
        assert layer_cache.sequence_length == 5


def test_gradient_checkpointing_position_offset_restriction(
    model: MujahidReasonerModel,
) -> None:
    """
    Checkpointed training currently operates only from
    position offset zero.
    """

    model.train()
    model.enable_gradient_checkpointing()

    input_ids = torch.randint(
        0,
        32000,
        (2, 8),
    )

    with pytest.raises(
        RuntimeError,
        match="position_offset=0",
    ):
        model(
            input_ids,
            position_offset=1,
            use_cache=False,
        )


def test_gradient_checkpointing_can_be_disabled_after_training(
    model: MujahidReasonerModel,
) -> None:
    """
    Disabling checkpointing must restore the normal path.
    """

    model.train()

    input_ids = torch.randint(
        0,
        32000,
        (2, 8),
    )

    model.enable_gradient_checkpointing()

    checkpointed_logits, checkpointed_cache = model(
        input_ids,
        use_cache=False,
    )

    assert checkpointed_cache is None

    model.disable_gradient_checkpointing()

    normal_logits, normal_cache = model(
        input_ids,
        use_cache=False,
    )

    assert normal_cache is None

    assert torch.allclose(
        checkpointed_logits,
        normal_logits,
        atol=1e-5,
        rtol=1e-5,
    )