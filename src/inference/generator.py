from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from tokenizers import Tokenizer

from src.model.cache import LayerKVCache
from src.model.model import MujahidReasonerModel


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9


class TextGenerator:
    """
    Autoregressive text generator with optional KV caching.

    KV caching avoids recomputing attention keys and values
    for tokens that have already been processed.
    """

    def __init__(
        self,
        model: MujahidReasonerModel,
        tokenizer: Tokenizer,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = torch.device(device)

        self.model.eval()

        self.bos_token_id = tokenizer.token_to_id("<bos>")
        self.eos_token_id = tokenizer.token_to_id("<eos>")

        if self.bos_token_id is None:
            raise ValueError(
                "Tokenizer does not contain a <bos> token."
            )

        if self.eos_token_id is None:
            raise ValueError(
                "Tokenizer does not contain an <eos> token."
            )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        use_cache: bool = True,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt:
                Input text.

            config:
                Generation configuration.

            use_cache:
                Whether to use KV caching.

        Returns:
            Generated text including the original prompt.
        """

        if not isinstance(prompt, str):
            raise TypeError(
                "prompt must be a string."
            )

        if not prompt.strip():
            raise ValueError(
                "prompt must not be empty."
            )

        if config is None:
            config = GenerationConfig()

        self._validate_generation_config(config)

        encoded = self.tokenizer.encode(prompt)

        token_ids = encoded.ids

        if not token_ids:
            raise ValueError(
                "Tokenizer produced no tokens."
            )

        if len(token_ids) > self.model.config.max_sequence_length:
            token_ids = token_ids[
                -self.model.config.max_sequence_length :
            ]

        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )

        if use_cache:
            generated_ids = self._generate_with_cache(
                input_ids,
                config,
            )
        else:
            generated_ids = self._generate_without_cache(
                input_ids,
                config,
            )

        return self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

    def _generate_with_cache(
        self,
        input_ids: Tensor,
        config: GenerationConfig,
    ) -> list[int]:
        """
        Generate using incremental KV-cache decoding.
        """

        generated_ids = input_ids[0].tolist()

        logits, cache = self.model(
            input_ids,
            use_cache=True,
        )

        if cache is None:
            raise RuntimeError(
                "Model did not return a KV cache."
            )

        next_logits = logits[:, -1, :]

        for _ in range(config.max_new_tokens):
            if (
                cache.layers[0] is not None
                and cache.layers[0].sequence_length
                >= self.model.config.max_sequence_length
            ):
                break

            next_token = self._sample_token(
                next_logits,
                config,
            )

            token_id = int(
                next_token.item()
            )

            generated_ids.append(token_id)

            if token_id == self.eos_token_id:
                break

            position_offset = (
                cache.layers[0].sequence_length
                if cache.layers[0] is not None
                else len(generated_ids) - 1
            )

            next_input = next_token.view(
                1,
                1,
            )

            logits, cache = self.model(
                next_input,
                position_offset=position_offset,
                cache=cache,
                use_cache=True,
            )

            if cache is None:
                raise RuntimeError(
                    "Model stopped returning the KV cache."
                )

            next_logits = logits[:, -1, :]

        return generated_ids

    def _generate_without_cache(
        self,
        input_ids: Tensor,
        config: GenerationConfig,
    ) -> list[int]:
        """
        Generate without KV caching.

        This path is retained as a correctness and benchmarking
        reference implementation.
        """

        generated_ids = input_ids[0].tolist()

        for _ in range(config.max_new_tokens):
            context_ids = generated_ids[
                -self.model.config.max_sequence_length :
            ]

            context = torch.tensor(
                [context_ids],
                dtype=torch.long,
                device=self.device,
            )

            logits, _ = self.model(
                context,
                use_cache=False,
            )

            next_logits = logits[:, -1, :]

            next_token = self._sample_token(
                next_logits,
                config,
            )

            token_id = int(
                next_token.item()
            )

            generated_ids.append(token_id)

            if token_id == self.eos_token_id:
                break

        return generated_ids

    @staticmethod
    def _sample_token(
        logits: Tensor,
        config: GenerationConfig,
    ) -> Tensor:
        """
        Apply temperature, top-k and top-p sampling.
        """

        if logits.ndim != 2:
            raise ValueError(
                "Logits must have shape [batch, vocab_size]."
            )

        logits = logits / config.temperature

        if config.top_k > 0:
            logits = TextGenerator._apply_top_k(
                logits,
                config.top_k,
            )

        if config.top_p < 1.0:
            logits = TextGenerator._apply_top_p(
                logits,
                config.top_p,
            )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        return torch.multinomial(
            probabilities,
            num_samples=1,
        )

    @staticmethod
    def _apply_top_k(
        logits: Tensor,
        top_k: int,
    ) -> Tensor:
        vocab_size = logits.size(-1)

        top_k = min(
            top_k,
            vocab_size,
        )

        threshold = torch.topk(
            logits,
            top_k,
            dim=-1,
        ).values[..., -1, None]

        return logits.masked_fill(
            logits < threshold,
            float("-inf"),
        )

    @staticmethod
    def _apply_top_p(
        logits: Tensor,
        top_p: float,
    ) -> Tensor:
        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
            dim=-1,
        )

        sorted_probabilities = torch.softmax(
            sorted_logits,
            dim=-1,
        )

        cumulative_probabilities = torch.cumsum(
            sorted_probabilities,
            dim=-1,
        )

        remove_mask = (
            cumulative_probabilities > top_p
        )

        remove_mask[..., 1:] = (
            remove_mask[..., :-1].clone()
        )

        remove_mask[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(
            remove_mask,
            float("-inf"),
        )

        return torch.zeros_like(logits).scatter(
            -1,
            sorted_indices,
            sorted_logits,
        )

    @staticmethod
    def _validate_generation_config(
        config: GenerationConfig,
    ) -> None:
        if config.max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens must be non-negative."
            )

        if config.temperature <= 0:
            raise ValueError(
                "temperature must be greater than zero."
            )

        if config.top_k < 0:
            raise ValueError(
                "top_k must be non-negative."
            )

        if not 0 < config.top_p <= 1:
            raise ValueError(
                "top_p must be in the range (0, 1]."
            )