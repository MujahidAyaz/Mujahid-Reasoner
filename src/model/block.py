from __future__ import annotations

from torch import Tensor, nn

from src.model.attention import GroupedQueryAttention
from src.model.cache import KVCache
from src.model.config import ModelConfig
from src.model.mlp import SwiGLU
from src.model.norm import RMSNorm


class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer block with optional KV caching.

    Architecture:

        x
        │
        ├── RMSNorm
        │
        ├── GQA + RoPE + optional KV cache
        │
        └── residual
                │
                ├── RMSNorm
                │
                ├── SwiGLU
                │
                └── residual

    Normal forward:
        Input  -> [batch, sequence_length, hidden_size]
        Output -> [batch, sequence_length, hidden_size]

    Cached forward:
        Input  -> new tokens only
        Cache  -> previous key/value states
        Output -> new tokens only
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
        )

        self.self_attn = GroupedQueryAttention(
            config,
        )

        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
        )

        self.mlp = SwiGLU(
            config,
        )

    def forward(
        self,
        x: Tensor,
        position_offset: int = 0,
        cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KVCache | None]:
        """
        Apply one Pre-Norm Transformer block.

        Args:
            x:
                Input tensor with shape
                [batch, sequence_length, hidden_size].

            position_offset:
                Starting position for RoPE.

            cache:
                Optional KV cache from the previous forward pass.

            use_cache:
                Whether to return an updated KV cache.

        Returns:
            A tuple containing:

            output:
                [batch, sequence_length, hidden_size]

            updated_cache:
                Updated KV cache, or None when caching is disabled.
        """

        if x.ndim != 3:
            raise ValueError(
                "Transformer block input must have shape "
                "[batch, sequence_length, hidden_size]."
            )

        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                "Input hidden size does not match "
                "the configured hidden size."
            )

        # ------------------------------------------------------------
        # Attention sub-layer
        # ------------------------------------------------------------

        residual = x

        x = self.input_layernorm(x)

        attention_output_result = self.self_attn(
            x,
            position_offset=position_offset,
            cache=cache,
            use_cache=use_cache,
        )

        if use_cache or cache is not None:
            attention_output, updated_cache = attention_output_result
        else:
            attention_output = attention_output_result
            updated_cache = None

        x = residual + attention_output

        # ------------------------------------------------------------
        # Feed-forward sub-layer
        # ------------------------------------------------------------

        residual = x

        x = self.post_attention_layernorm(x)

        x = self.mlp(x)

        x = residual + x

        return x, updated_cache