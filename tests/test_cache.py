from __future__ import annotations

import pytest
import torch

from src.model.cache import KVCache, LayerKVCache


def test_kv_cache_properties() -> None:
    key = torch.randn(2, 4, 16, 32)
    value = torch.randn(2, 4, 16, 32)

    cache = KVCache(
        key=key,
        value=value,
        sequence_length=8,
    )

    assert cache.sequence_length == 8
    assert cache.capacity == 16
    assert cache.device == key.device
    assert cache.dtype == key.dtype
    assert cache.batch_size == 2
    assert cache.num_kv_heads == 4
    assert cache.head_dim == 32


def test_kv_cache_append() -> None:
    key = torch.zeros(2, 4, 16, 32)
    value = torch.zeros(2, 4, 16, 32)

    cache = KVCache(
        key=key,
        value=value,
        sequence_length=8,
    )

    new_key = torch.randn(2, 4, 1, 32)
    new_value = torch.randn(2, 4, 1, 32)

    original_key_storage = cache.key

    updated = cache.append(
        key=new_key,
        value=new_value,
    )

    assert updated is cache
    assert updated.key is original_key_storage
    assert updated.sequence_length == 9

    cached_key, cached_value = updated.get()

    assert cached_key.shape == (2, 4, 9, 32)
    assert cached_value.shape == (2, 4, 9, 32)

    assert torch.equal(
        cached_key[:, :, 8:9, :],
        new_key,
    )

    assert torch.equal(
        cached_value[:, :, 8:9, :],
        new_value,
    )


def test_kv_cache_get_returns_populated_region() -> None:
    key = torch.randn(2, 4, 16, 32)
    value = torch.randn(2, 4, 16, 32)

    cache = KVCache(
        key=key,
        value=value,
        sequence_length=5,
    )

    cached_key, cached_value = cache.get()

    assert cached_key.shape == (2, 4, 5, 32)
    assert cached_value.shape == (2, 4, 5, 32)


def test_kv_cache_capacity_error() -> None:
    key = torch.zeros(1, 2, 4, 8)
    value = torch.zeros(1, 2, 4, 8)

    cache = KVCache(
        key=key,
        value=value,
        sequence_length=4,
    )

    new_key = torch.randn(1, 2, 1, 8)
    new_value = torch.randn(1, 2, 1, 8)

    with pytest.raises(ValueError, match="capacity exceeded"):
        cache.append(
            key=new_key,
            value=new_value,
        )


def test_kv_cache_shape_validation() -> None:
    key = torch.zeros(1, 2, 8, 16)
    value = torch.zeros(1, 2, 8, 16)

    cache = KVCache(
        key=key,
        value=value,
        sequence_length=8,
    )

    bad_key = torch.randn(1, 3, 1, 16)
    bad_value = torch.randn(1, 3, 1, 16)

    with pytest.raises(
        ValueError,
        match="Number of KV heads mismatch",
    ):
        cache.append(
            key=bad_key,
            value=bad_value,
        )


def test_layer_kv_cache_empty() -> None:
    cache = LayerKVCache.empty(6)

    assert len(cache) == 6
    assert all(layer is None for layer in cache.layers)


def test_layer_kv_cache_set_get() -> None:
    cache = LayerKVCache.empty(2)

    key = torch.zeros(1, 2, 16, 8)
    value = torch.zeros(1, 2, 16, 8)

    layer_cache = KVCache(
        key=key,
        value=value,
        sequence_length=4,
    )

    cache[0] = layer_cache

    assert cache[0] is layer_cache
    assert cache[1] is None


def test_invalid_sequence_length() -> None:
    key = torch.zeros(1, 2, 8, 16)
    value = torch.zeros(1, 2, 8, 16)

    with pytest.raises(
        ValueError,
        match="sequence_length",
    ):
        KVCache(
            key=key,
            value=value,
            sequence_length=9,
        )