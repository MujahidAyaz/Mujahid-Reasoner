from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.model.config import ModelConfig
from src.model.rope import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).

    Query heads can be more numerous than key/value heads.

    Example:
        Query heads     = 8
        Key/Value heads = 4

    Each key/value head is shared by two query heads.

    Input:
        [batch, sequence_length, hidden_size]

    Output:
        [batch, sequence_length, hidden_size]
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

    def _reshape_query(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Convert projected queries from:

            [B, T, H * D]

        to:

            [B, H, T, D]
        """

        batch_size, sequence_length, _ = x.shape

        x = x.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.head_dimension,
        )

        return x.transpose(1, 2)

    def _reshape_key_value(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Convert projected keys/values from:

            [B, T, KV * D]

        to:

            [B, KV, T, D]
        """

        batch_size, sequence_length, _ = x.shape

        x = x.view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dimension,
        )

        return x.transpose(1, 2)

    def _repeat_key_value(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Expand key/value heads to match query heads.

        Example:

            KV heads = 4
            Q heads  = 8

        Each KV head is repeated twice.

        [B, 4, T, D]
            ↓
        [B, 8, T, D]
        """

        if self.kv_group_size == 1:
            return x

        return x.repeat_interleave(
            self.kv_group_size,
            dim=1,
        )

    def forward(
        self,
        x: Tensor,
        position_offset: int = 0,
    ) -> Tensor:
        """
        Apply grouped query attention.

        Args:
            x:
                Input tensor with shape
                [batch, sequence_length, hidden_size].

            position_offset:
                Starting position for RoPE.

        Returns:
            Tensor with shape
            [batch, sequence_length, hidden_size].
        """

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

        if (
            position_offset < 0
            or position_offset + sequence_length
            > self.max_sequence_length
        ):
            raise ValueError(
                "Attention sequence exceeds the "
                "configured maximum sequence length."
            )

        # ------------------------------------------------------------
        # 1. Project input into queries, keys, and values.
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
        # 3. Apply Rotary Position Embeddings.
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
        # 4. Expand K/V heads for Grouped Query Attention.
        # ------------------------------------------------------------

        key = self._repeat_key_value(key)
        value = self._repeat_key_value(value)

        # ------------------------------------------------------------
        # 5. Compute scaled dot-product attention.
        #
        # Q: [B, H, T, D]
        # K: [B, H, T, D]
        #
        # scores:
        #     [B, H, T, T]
        # ------------------------------------------------------------

        scale = 1.0 / math.sqrt(
            self.head_dimension
        )

        attention_scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        ) * scale

        # ------------------------------------------------------------
        # 6. Apply causal masking.
        #
        # A token may only attend to itself and previous tokens.
        # ------------------------------------------------------------

        mask = self.causal_mask[
            :sequence_length,
            :sequence_length,
        ]

        attention_scores = attention_scores.masked_fill(
            ~mask,
            torch.finfo(attention_scores.dtype).min,
        )

        # ------------------------------------------------------------
        # 7. Convert scores into attention probabilities.
        # ------------------------------------------------------------

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        # ------------------------------------------------------------
        # 8. Weighted combination of value vectors.
        # ------------------------------------------------------------

        attention_output = torch.matmul(
            attention_weights,
            value,
        )

        # ------------------------------------------------------------
        # 9. Merge attention heads.
        #
        # [B, H, T, D]
        #       ↓
        # [B, T, H * D]
        # ------------------------------------------------------------

        attention_output = attention_output.transpose(
            1,
            2,
        ).contiguous()

        attention_output = attention_output.view(
            batch_size,
            sequence_length,
            self.num_attention_heads * self.head_dimension,
        )

        # ------------------------------------------------------------
        # 10. Final output projection.
        # ------------------------------------------------------------

        return self.o_proj(attention_output)