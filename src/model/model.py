from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from src.model.block import TransformerBlock
from src.model.cache import LayerKVCache
from src.model.config import ModelConfig
from src.model.norm import RMSNorm


class MujahidReasonerModel(nn.Module):
    """
    Decoder-only Transformer language model.

    Features:

        - RMSNorm
        - RoPE
        - Grouped Query Attention
        - SwiGLU
        - Optional KV caching
        - Optional gradient checkpointing

    Gradient checkpointing is a training-time memory optimization.
    When enabled, intermediate activations inside Transformer blocks
    are not retained for backward. They are recomputed during the
    backward pass.

    KV caching and gradient checkpointing are intentionally separated:

        Training:
            gradient checkpointing may be enabled.

        Inference:
            KV caching is supported normally.

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

        # Runtime training optimization.
        #
        # This is deliberately kept as model state rather than adding
        # it to ModelConfig because gradient checkpointing is a runtime
        # execution strategy, not an architectural property of the model.
        self.gradient_checkpointing = False

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

    def enable_gradient_checkpointing(self) -> None:
        """
        Enable activation/gradient checkpointing.

        When enabled, Transformer blocks are recomputed during the
        backward pass instead of retaining all intermediate activations.

        This reduces activation memory at the cost of additional
        computation.

        Checkpointing is only used during training and only when
        KV caching is disabled.
        """

        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        """
        Disable activation/gradient checkpointing.
        """

        self.gradient_checkpointing = False

    def is_gradient_checkpointing_enabled(self) -> bool:
        """
        Return whether gradient checkpointing is enabled.
        """

        return self.gradient_checkpointing

    def _checkpoint_layer(
        self,
        layer: TransformerBlock,
        x: Tensor,
    ) -> Tensor:
        """
        Execute one Transformer block through activation checkpointing.

        The checkpointed function returns only the hidden-state tensor.

        KV-cache handling is deliberately excluded because cache
        mutation is stateful and should never be replayed during
        backward recomputation.
        """

        def forward_without_cache(
            hidden_states: Tensor,
        ) -> Tensor:
            output, updated_cache = layer(
                hidden_states,
                position_offset=0,
                cache=None,
                use_cache=False,
            )

            if updated_cache is not None:
                raise RuntimeError(
                    "Gradient-checkpointed Transformer blocks "
                    "must not produce a KV cache."
                )

            return output

        return checkpoint(
            forward_without_cache,
            x,
            use_reentrant=False,
        )

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

        # Gradient checkpointing is a training-only optimization.
        #
        # Never use it when:
        #   - the model is in evaluation mode
        #   - KV caching is requested
        #   - an existing cache is being consumed
        #
        # This keeps autoregressive inference completely independent
        # from the checkpointing mechanism.
        use_gradient_checkpointing = (
            self.training
            and self.gradient_checkpointing
            and not use_cache
            and cache is None
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

            if use_gradient_checkpointing:
                if position_offset != 0:
                    raise RuntimeError(
                        "Gradient checkpointing currently requires "
                        "position_offset=0."
                    )

                x = self._checkpoint_layer(
                    layer,
                    x,
                )

                continue

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