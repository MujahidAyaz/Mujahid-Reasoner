from __future__ import annotations

import pytest
import torch

from src.model.cache import KVCache, LayerKVCache


def test_kv_cache_properties() -> None:
    key = torch.randn(2, 4, 8, 32)
    value = torch.randn(2, 4, 8, 32)

    cache = KVCache(key=key, value=value)

    assert cache.sequence_length == 8
    assert cache.device == key.device
    assert cache.dtype == key.dtype


def test_kv_cache_append() -> None:
    key = torch.randn(2, 4, 8, 32)
    value = torch.randn(2, 4, 8, 32)

    new_key = torch.randn(2, 4, 1, 32)
    new_value = torch.randn(2, 4, 1, 32)

    cache = KVCache(key=key, value=value)

    updated = cache.append(
        key=new_key,
        value=new_value,
    )

    assert updated.key.shape == (2, 4, 9, 32)
    assert updated.value.shape == (2, 4, 9, 32)

    assert torch.equal(
        updated.key[:, :, :8],
        key,
    )

    assert torch.equal(
        updated.key[:, :, 8:],
        new_key,
    )


def test_kv_cache_rejects_invalid_rank() -> None:
    key = torch.randn(2, 4, 8)
    value = torch.randn(2, 4, 8)

    cache = KVCache(
        key=torch.randn(2, 4, 8, 32),
        value=torch.randn(2, 4, 8, 32),
    )

    with pytest.raises(ValueError):
        cache.append(key, value)


def test_kv_cache_rejects_shape_mismatch() -> None:
    cache = KVCache(
        key=torch.randn(2, 4, 8, 32),
        value=torch.randn(2, 4, 8, 32),
    )

    with pytest.raises(ValueError):
        cache.append(
            torch.randn(2, 4, 1, 32),
            torch.randn(2, 4, 1, 16),
        )


def test_layer_cache_empty() -> None:
    cache = LayerKVCache.empty(6)

    assert len(cache) == 6
    assert all(layer is None for layer in cache.layers)


def test_layer_cache_assignment() -> None:
    cache = LayerKVCache.empty(6)

    layer_cache = KVCache(
        key=torch.randn(2, 4, 8, 32),
        value=torch.randn(2, 4, 8, 32),
    )

    cache[0] = layer_cache

    assert cache[0] is layer_cache
    assert cache[1] is None