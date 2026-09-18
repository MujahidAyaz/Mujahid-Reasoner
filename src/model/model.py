from __future__ import annotations

import torch
from torch import Tensor, nn

from src.model.block import TransformerBlock
from src.model.cache import LayerKVCache
from src.model.config import ModelConfig
from src.model.norm import RMSNorm


class MujahidReasonerModel(nn.Module):
    """
    Decoder-only Transformer language model with optional KV caching.

    Architecture:

        input_ids
            │
            ▼
        Token Embedding
            │
            ▼
        Transformer Block × N
            │
            ▼
        Final RMSNorm
            │
            ▼
        LM Head
            │
            ▼
        Logits

    Normal forward:
        Input:
            [batch, sequence_length]

        Output:
            [batch, sequence_length, vocab_size]

    Cached forward:
        Input:
            New token(s) only

        Cache:
            Previous K/V states for every Transformer layer

        Output:
            Logits for new token(s) only
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        self.final_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._initialize_weights)

    def _initialize_weights(
        self,
        module: nn.Module,
    ) -> None:
        """
        Initialize model parameters.

        Linear and embedding weights use a normal distribution
        controlled by initializer_range.

        RMSNorm weights are initialized to one.
        """

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: Tensor,
        position_offset: int = 0,
        cache: LayerKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, LayerKVCache | None]:
        """
        Run the decoder-only Transformer.

        Args:
            input_ids:
                Token IDs with shape
                [batch, sequence_length].

            position_offset:
                Starting position for RoPE.

            cache:
                Optional KV cache containing previous states
                for every Transformer layer.

            use_cache:
                Whether to return an updated KV cache.

        Returns:
            logits:
                Tensor with shape
                [batch, sequence_length, vocab_size].

            updated_cache:
                Updated cache for all Transformer layers,
                or None when caching is disabled.
        """

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape "
                "[batch, sequence_length]."
            )

        batch_size, sequence_length = input_ids.shape

        if sequence_length <= 0:
            raise ValueError(
                "Sequence length must be positive."
            )

        if (
            position_offset < 0
            or position_offset + sequence_length
            > self.config.max_sequence_length
        ):
            raise ValueError(
                "Sequence exceeds the configured "
                "maximum sequence length."
            )

        if input_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "input_ids must use an integer dtype."
            )

        if torch.any(input_ids < 0):
            raise ValueError(
                "input_ids cannot contain negative token IDs."
            )

        if torch.any(
            input_ids >= self.config.vocab_size
        ):
            raise ValueError(
                "input_ids contain a token ID outside "
                "the configured vocabulary."
            )

        if cache is not None:
            if len(cache) != len(self.layers):
                raise ValueError(
                    "KV cache layer count does not match "
                    "the number of Transformer layers."
                )

            if cache.layers[0] is not None:
                if cache.layers[0].key.size(0) != batch_size:
                    raise ValueError(
                        "KV cache batch size does not match "
                        "the input batch size."
                    )

                if (
                    cache.layers[0].sequence_length
                    != position_offset
                ):
                    raise ValueError(
                        "KV cache sequence length must match "
                        "position_offset."
                    )

        x = self.token_embedding(input_ids)

        updated_cache = None

        if use_cache:
            updated_cache = LayerKVCache.empty(
                num_layers=len(self.layers),
            )

        for layer_index, layer in enumerate(self.layers):
            layer_cache = (
                cache[layer_index]
                if cache is not None
                else None
            )

            x, layer_updated_cache = layer(
                x,
                position_offset=position_offset,
                cache=layer_cache,
                use_cache=use_cache,
            )

            if use_cache:
                if layer_updated_cache is None:
                    raise RuntimeError(
                        "Layer did not return a KV cache "
                        "when use_cache=True."
                    )

                updated_cache[layer_index] = (
                    layer_updated_cache
                )

        x = self.final_layernorm(x)

        logits = self.lm_head(x)

        return logits, updated_cache

    def parameter_count(
        self,
        trainable_only: bool = True,
    ) -> int:
        """
        Return the number of model parameters.
        """

        if trainable_only:
            return sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )

        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )