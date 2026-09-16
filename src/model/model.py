from __future__ import annotations

import torch
from torch import Tensor, nn

from src.model.block import TransformerBlock
from src.model.config import ModelConfig
from src.model.norm import RMSNorm


class MujahidReasonerModel(nn.Module):
    """
    Decoder-only Transformer language model.

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

    Input:
        [batch, sequence_length]

    Output:
        [batch, sequence_length, vocab_size]
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
    ) -> Tensor:
        """
        Run the decoder-only Transformer.

        Args:
            input_ids:
                Token IDs with shape
                [batch, sequence_length].

            position_offset:
                Starting position for RoPE.

        Returns:
            Logits with shape
            [batch, sequence_length, vocab_size].
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

        x = self.token_embedding(input_ids)

        for layer in self.layers:
            x = layer(
                x,
                position_offset=position_offset,
            )

        x = self.final_layernorm(x)

        logits = self.lm_head(x)

        return logits

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