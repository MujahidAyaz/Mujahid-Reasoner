from __future__ import annotations

from collections.abc import Iterator
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

    Supports:

    - Standard full-text generation.
    - Incremental streaming generation.
    - Preallocated KV-cache decoding.
    - Temperature sampling.
    - Top-k sampling.
    - Top-p / nucleus sampling.

    The existing `generate()` API is preserved so current
    callers remain compatible.
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
            raise ValueError("Tokenizer does not contain a <bos> token.")

        if self.eos_token_id is None:
            raise ValueError("Tokenizer does not contain an <eos> token.")

    # ------------------------------------------------------------------
    # Standard generation API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        use_cache: bool = True,
    ) -> str:
        """
        Generate complete text from a prompt.

        Returns:
            Generated text including the original prompt.
        """

        self._validate_prompt(prompt)

        if config is None:
            config = GenerationConfig()

        self._validate_generation_config(config)

        encoded = self.tokenizer.encode(prompt)
        token_ids = encoded.ids

        if not token_ids:
            raise ValueError("Tokenizer produced no tokens.")

        token_ids = self._truncate_prompt(token_ids)

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

    # ------------------------------------------------------------------
    # Streaming generation API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        use_cache: bool = True,
    ) -> Iterator[str]:
        """
        Stream generated text incrementally.

        Each yielded value contains only newly available text.

        The original prompt is not yielded.
        """

        self._validate_prompt(prompt)

        if config is None:
            config = GenerationConfig()

        self._validate_generation_config(config)

        encoded = self.tokenizer.encode(prompt)
        token_ids = encoded.ids

        if not token_ids:
            raise ValueError("Tokenizer produced no tokens.")

        token_ids = self._truncate_prompt(token_ids)

        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )

        if use_cache:
            yield from self._stream_with_cache(
                input_ids,
                config,
            )
        else:
            yield from self._stream_without_cache(
                input_ids,
                config,
            )

    # ------------------------------------------------------------------
    # Cached generation
    # ------------------------------------------------------------------

    def _generate_with_cache(
        self,
        input_ids: Tensor,
        config: GenerationConfig,
    ) -> list[int]:
        """Generate using incremental KV-cache decoding."""

        generated_ids = input_ids[0].tolist()

        logits, cache = self.model(
            input_ids,
            use_cache=True,
        )

        if cache is None:
            raise RuntimeError("Model did not return a KV cache.")

        next_logits = logits[:, -1, :]

        for _ in range(config.max_new_tokens):
            if self._cache_is_full(cache):
                break

            next_token = self._sample_token(
                next_logits,
                config,
            )

            token_id = int(next_token.item())

            generated_ids.append(token_id)

            if token_id == self.eos_token_id:
                break

            position_offset = self._get_cache_length(
                cache,
                fallback=len(generated_ids) - 1,
            )

            next_input = next_token.view(1, 1)

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

    # ------------------------------------------------------------------
    # Cached streaming generation
    # ------------------------------------------------------------------

    def _stream_with_cache(
        self,
        input_ids: Tensor,
        config: GenerationConfig,
    ) -> Iterator[str]:
        """
        Stream generation using incremental KV-cache decoding.
        """

        logits, cache = self.model(
            input_ids,
            use_cache=True,
        )

        if cache is None:
            raise RuntimeError("Model did not return a KV cache.")

        next_logits = logits[:, -1, :]

        generated_ids: list[int] = []
        streamed_text = ""

        for _ in range(config.max_new_tokens):
            if self._cache_is_full(cache):
                break

            next_token = self._sample_token(
                next_logits,
                config,
            )

            token_id = int(next_token.item())

            generated_ids.append(token_id)

            if token_id == self.eos_token_id:
                break

            chunk, streamed_text = self._decode_generated_chunk(
                generated_ids,
                previous_text=streamed_text,
            )

            if chunk:
                yield chunk

            position_offset = self._get_cache_length(
                cache,
                fallback=len(input_ids[0]) + len(generated_ids) - 1,
            )

            next_input = next_token.view(1, 1)

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

    # ------------------------------------------------------------------
    # Non-cached generation
    # ------------------------------------------------------------------

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
                -self.model.config.max_sequence_length:
            ]

            context = torch.tensor(
                [context_ids],
                dtype=torch.long,
                device=self.device,
            )

            model_output = self.model(
                context,
                use_cache=False,
            )

            logits = (
                model_output[0]
                if isinstance(model_output, tuple)
                else model_output
            )

            next_logits = logits[:, -1, :]

            next_token = self._sample_token(
                next_logits,
                config,
            )

            token_id = int(next_token.item())

            generated_ids.append(token_id)

            if token_id == self.eos_token_id:
                break

        return generated_ids

    # ------------------------------------------------------------------
    # Non-cached streaming generation
    # ------------------------------------------------------------------

    def _stream_without_cache(
        self,
        input_ids: Tensor,
        config: GenerationConfig,
    ) -> Iterator[str]:
        """
        Stream generation without KV caching.

        This path exists primarily for correctness testing and
        benchmarking. Cached streaming should be preferred.
        """

        generated_ids = input_ids[0].tolist()
        streamed_ids: list[int] = []
        streamed_text = ""

        for _ in range(config.max_new_tokens):
            context_ids = generated_ids[
                -self.model.config.max_sequence_length:
            ]

            context = torch.tensor(
                [context_ids],
                dtype=torch.long,
                device=self.device,
            )

            model_output = self.model(
                context,
                use_cache=False,
            )

            logits = (
                model_output[0]
                if isinstance(model_output, tuple)
                else model_output
            )

            next_logits = logits[:, -1, :]

            next_token = self._sample_token(
                next_logits,
                config,
            )

            token_id = int(next_token.item())

            generated_ids.append(token_id)

            if token_id == self.eos_token_id:
                break

            streamed_ids.append(token_id)

            chunk, streamed_text = self._decode_generated_chunk(
                streamed_ids,
                previous_text=streamed_text,
            )

            if chunk:
                yield chunk

    # ------------------------------------------------------------------
    # Decoding helpers
    # ------------------------------------------------------------------

    def _decode_generated_chunk(
        self,
        generated_ids: list[int],
        previous_text: str = "",
    ) -> tuple[str, str]:
        """
        Decode the generated token prefix and return only the newly
        available text.

        Returns:
            A tuple containing:

            - newly generated text chunk
            - complete decoded generated text so far
        """

        current_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        if current_text.startswith(previous_text):
            chunk = current_text[len(previous_text):]
        else:
            chunk = current_text

        return chunk, current_text

    def _truncate_prompt(
        self,
        token_ids: list[int],
    ) -> list[int]:
        """
        Keep only the most recent tokens that fit the context window.
        """

        max_length = self.model.config.max_sequence_length

        if len(token_ids) <= max_length:
            return token_ids

        return token_ids[-max_length:]

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_is_full(
        self,
        cache: LayerKVCache,
    ) -> bool:
        if not cache.layers:
            return False

        first_layer = cache.layers[0]

        if first_layer is None:
            return False

        return (
            first_layer.sequence_length
            >= self.model.config.max_sequence_length
        )

    @staticmethod
    def _get_cache_length(
        cache: LayerKVCache,
        fallback: int,
    ) -> int:
        if cache.layers:
            first_layer = cache.layers[0]

            if first_layer is not None:
                return first_layer.sequence_length

        return fallback

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

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

        remove_mask = cumulative_probabilities > top_p

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

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_prompt(
        prompt: str,
    ) -> None:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")

        if not prompt.strip():
            raise ValueError("prompt must not be empty.")

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