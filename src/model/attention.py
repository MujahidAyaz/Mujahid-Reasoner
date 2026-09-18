from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.model.cache import KVCache
from src.model.config import ModelConfig
from src.model.rope import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with RoPE and preallocated KV caching.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dimension = config.head_dimension
        self.kv_group_size = config.kv_group_size
        self.max_sequence_length = config.max_sequence_length

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dimension,
            bias=False,
        )

        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dimension,
            bias=False,
        )

        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dimension,
            bias=False,
        )

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dimension,
            config.hidden_size,
            bias=False,
        )

        self.rope = RotaryEmbedding(
            head_dimension=self.head_dimension,
            max_sequence_length=config.max_sequence_length,
            theta=config.rope_theta,
        )

        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    config.max_sequence_length,
                    config.max_sequence_length,
                    dtype=torch.bool,
                )
            ),
            persistent=False,
        )

    def _reshape_query(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, _ = x.shape

        x = x.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.head_dimension,
        )

        return x.transpose(1, 2)

    def _reshape_key_value(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, _ = x.shape

        x = x.view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dimension,
        )

        return x.transpose(1, 2)

    def _repeat_key_value(self, x: Tensor) -> Tensor:
        if self.kv_group_size == 1:
            return x

        return x.repeat_interleave(
            self.kv_group_size,
            dim=1,
        )

    def _validate_cache(
        self,
        cache: KVCache,
        batch_size: int,
        position_offset: int,
    ) -> None:
        if cache.batch_size != batch_size:
            raise ValueError(
                "KV cache batch size does not match "
                "the attention input."
            )

        if cache.num_kv_heads != self.num_key_value_heads:
            raise ValueError(
                "KV cache head count does not match "
                "the configured number of KV heads."
            )

        if cache.head_dim != self.head_dimension:
            raise ValueError(
                "KV cache head dimension does not match "
                "the configured head dimension."
            )

        if cache.sequence_length != position_offset:
            raise ValueError(
                "KV cache sequence length must match "
                "position_offset."
            )

        if cache.capacity != self.max_sequence_length:
            raise ValueError(
                "KV cache capacity does not match "
                "the configured maximum sequence length."
            )

    def forward(
        self,
        x: Tensor,
        position_offset: int = 0,
        cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KVCache | None]:

        if x.ndim != 3:
            raise ValueError(
                "Attention input must have shape "
                "[batch, sequence_length, hidden_size]."
            )

        batch_size, sequence_length, hidden_size = x.shape

        if hidden_size != self.hidden_size:
            raise ValueError(
                "Input hidden size does not match "
                "the configured hidden size."
            )

        if sequence_length <= 0:
            raise ValueError(
                "Sequence length must be positive."
            )

        if position_offset < 0:
            raise ValueError(
                "position_offset must be non-negative."
            )

        if (
            position_offset + sequence_length
            > self.max_sequence_length
        ):
            raise ValueError(
                "Attention sequence exceeds the "
                "configured maximum sequence length."
            )

        # ------------------------------------------------------------
        # 1. Project input into Q, K and V.
        # ------------------------------------------------------------

        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        # ------------------------------------------------------------
        # 2. Split projections into attention heads.
        # ------------------------------------------------------------

        query = self._reshape_query(query)
        key = self._reshape_key_value(key)
        value = self._reshape_key_value(value)

        # ------------------------------------------------------------
        # 3. Apply RoPE.
        # ------------------------------------------------------------

        query = self.rope(
            query,
            position_offset=position_offset,
        )

        key = self.rope(
            key,
            position_offset=position_offset,
        )

        # ------------------------------------------------------------
        # 4. Prepare KV cache.
        # ------------------------------------------------------------

        if cache is not None:
            self._validate_cache(
                cache,
                batch_size=batch_size,
                position_offset=position_offset,
            )

            # Write the newly computed K/V states into the
            # preallocated cache.
            cache.append(
                key=key,
                value=value,
            )

            key, value = cache.get()

        elif use_cache:
            # Allocate cache storage once for the entire
            # maximum sequence length.
            cache = KVCache(
                key=torch.empty(
                    batch_size,
                    self.num_key_value_heads,
                    self.max_sequence_length,
                    self.head_dimension,
                    device=key.device,
                    dtype=key.dtype,
                ),
                value=torch.empty(
                    batch_size,
                    self.num_key_value_heads,
                    self.max_sequence_length,
                    self.head_dimension,
                    device=value.device,
                    dtype=value.dtype,
                ),
                sequence_length=0,
            )

            cache.append(
                key=key,
                value=value,
            )

            key, value = cache.get()

        total_sequence_length = key.size(2)

        if total_sequence_length > self.max_sequence_length:
            raise ValueError(
                "KV cache exceeds the configured "
                "maximum sequence length."
            )

        # ------------------------------------------------------------
        # 5. Expand K/V heads for GQA.
        # ------------------------------------------------------------

        key = self._repeat_key_value(key)
        value = self._repeat_key_value(value)

        # ------------------------------------------------------------
        # 6. Compute scaled dot-product attention.
        # ------------------------------------------------------------

        scale = 1.0 / math.sqrt(
            self.head_dimension
        )

        attention_scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        ) * scale

        # ------------------------------------------------------------
        # 7. Apply causal masking.
        # ------------------------------------------------------------

        if cache is None or position_offset == 0:
            mask = self.causal_mask[
                :sequence_length,
                :total_sequence_length,
            ]

        else:
            cache_length = position_offset

            current_mask = self.causal_mask[
                :sequence_length,
                :sequence_length,
            ]

            prefix_mask = torch.ones(
                sequence_length,
                cache_length,
                dtype=torch.bool,
                device=x.device,
            )

            mask = torch.cat(
                (
                    prefix_mask,
                    current_mask,
                ),
                dim=1,
            )

        attention_scores = attention_scores.masked_fill(
            ~mask,
            torch.finfo(attention_scores.dtype).min,
        )

        # ------------------------------------------------------------
        # 8. Attention probabilities.
        # ------------------------------------------------------------

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        # ------------------------------------------------------------
        # 9. Weighted combination of values.
        # ------------------------------------------------------------

        attention_output = torch.matmul(
            attention_weights,
            value,
        )

        # ------------------------------------------------------------
        # 10. Merge attention heads.
        # ------------------------------------------------------------

        attention_output = attention_output.transpose(
            1,
            2,
        ).contiguous()

        attention_output = attention_output.view(
            batch_size,
            sequence_length,
            self.num_attention_heads
            * self.head_dimension,
        )

        # ------------------------------------------------------------
        # 11. Final projection.
        # ------------------------------------------------------------

        output = self.o_proj(
            attention_output
        )

        if use_cache:
            return output, cache

        return output