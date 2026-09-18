from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class KVCache:
    """
    Preallocated key/value cache for one Transformer layer.

    Storage shape:

        [batch, num_kv_heads, max_sequence_length, head_dim]

    `sequence_length` tracks how many positions are currently populated.
    """

    key: Tensor
    value: Tensor
    sequence_length: int = 0

    def __post_init__(self) -> None:
        if self.key.ndim != 4 or self.value.ndim != 4:
            raise ValueError(
                "Key and value tensors must have shape "
                "[batch, heads, sequence, head_dim]."
            )

        if self.key.shape != self.value.shape:
            raise ValueError(
                "Key and value tensors must have identical shapes."
            )

        if not 0 <= self.sequence_length <= self.key.size(2):
            raise ValueError(
                "sequence_length must be within the cache capacity."
            )

    @property
    def capacity(self) -> int:
        return self.key.size(2)

    @property
    def device(self) -> torch.device:
        return self.key.device

    @property
    def dtype(self) -> torch.dtype:
        return self.key.dtype

    @property
    def batch_size(self) -> int:
        return self.key.size(0)

    @property
    def num_kv_heads(self) -> int:
        return self.key.size(1)

    @property
    def head_dim(self) -> int:
        return self.key.size(3)

    def append(
        self,
        key: Tensor,
        value: Tensor,
    ) -> KVCache:
        """
        Write new K/V states into the preallocated cache.

        No torch.cat is used. Existing cache storage is reused.
        """

        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                "Key and value tensors must have shape "
                "[batch, heads, sequence, head_dim]."
            )

        if key.shape != value.shape:
            raise ValueError(
                "Key and value tensors must have identical shapes."
            )

        if key.size(0) != self.batch_size:
            raise ValueError("Batch size mismatch.")

        if key.size(1) != self.num_kv_heads:
            raise ValueError("Number of KV heads mismatch.")

        if key.size(3) != self.head_dim:
            raise ValueError("Head dimension mismatch.")

        new_length = (
            self.sequence_length + key.size(2)
        )

        if new_length > self.capacity:
            raise ValueError(
                f"KV cache capacity exceeded: "
                f"{new_length} > {self.capacity}."
            )

        start = self.sequence_length
        end = new_length

        self.key[:, :, start:end, :] = key
        self.value[:, :, start:end, :] = value
        self.sequence_length = new_length

        return self

    def get(self) -> tuple[Tensor, Tensor]:
        """
        Return only the populated portion of the cache.
        """

        return (
            self.key[:, :, :self.sequence_length, :],
            self.value[:, :, :self.sequence_length, :],
        )


@dataclass
class LayerKVCache:
    """
    Preallocated KV cache for all Transformer layers.
    """

    layers: list[KVCache | None]

    @classmethod
    def empty(
        cls,
        num_layers: int,
    ) -> LayerKVCache:
        if num_layers <= 0:
            raise ValueError(
                "num_layers must be positive."
            )

        return cls(
            layers=[None] * num_layers
        )

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(
        self,
        index: int,
    ) -> KVCache | None:
        return self.layers[index]

    def __setitem__(
        self,
        index: int,
        cache: KVCache,
    ) -> None:
        self.layers[index] = cache