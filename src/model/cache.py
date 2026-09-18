from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class KVCache:
    """
    Stores key/value tensors for one Transformer layer.

    Keys and values use shape:

        [batch, num_kv_heads, sequence_length, head_dim]
    """

    key: Tensor
    value: Tensor

    @property
    def sequence_length(self) -> int:
        return self.key.size(2)

    @property
    def device(self) -> torch.device:
        return self.key.device

    @property
    def dtype(self) -> torch.dtype:
        return self.key.dtype

    def append(
        self,
        key: Tensor,
        value: Tensor,
    ) -> KVCache:
        """Return a new cache containing the appended K/V states."""

        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                "Key and value tensors must have shape "
                "[batch, heads, sequence, head_dim]."
            )

        if key.shape != value.shape:
            raise ValueError(
                "Key and value tensors must have identical shapes."
            )

        if key.size(0) != self.key.size(0):
            raise ValueError("Batch size mismatch.")

        if key.size(1) != self.key.size(1):
            raise ValueError("Number of KV heads mismatch.")

        if key.size(3) != self.key.size(3):
            raise ValueError("Head dimension mismatch.")

        return KVCache(
            key=torch.cat((self.key, key), dim=2),
            value=torch.cat((self.value, value), dim=2),
        )


@dataclass
class LayerKVCache:
    """
    KV cache for all Transformer layers.
    """

    layers: list[KVCache | None]

    @classmethod
    def empty(cls, num_layers: int) -> LayerKVCache:
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        return cls(
            layers=[None] * num_layers
        )

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> KVCache | None:
        return self.layers[index]

    def __setitem__(
        self,
        index: int,
        cache: KVCache,
    ) -> None:
        self.layers[index] = cache