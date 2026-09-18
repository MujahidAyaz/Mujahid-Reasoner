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
    do_sample: bool = True
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: tuple[str, ...] = ()


class TextGenerator:
    """
    Autoregressive text generator with optional KV caching.

    Supports:

    - Standard full-text generation.
    - Incremental streaming generation.
    - Preallocated KV-cache decoding.
    - Greedy decoding.
    - Temperature sampling.
    - Top-k sampling.
    - Top-p / nucleus sampling.
    - Repetition penalties.
    - Frequency penalties.
    - Presence penalties.
    - Stop sequences.

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
            raise ValueError(
                "Tokenizer does not contain a <bos> token."
            )

        if self.eos_token_id is None:
            raise ValueError(
                "Tokenizer does not contain an <eos> token."
            )

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
            raise ValueError(
                "Tokenizer produced no tokens."
            )

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

        Partial stop-sequence prefixes are buffered internally so
        they are never leaked to the caller.
        """

        self._validate_prompt(prompt)

        if config is None:
            config = GenerationConfig()

        self._validate_generation_config(config)

        encoded = self.tokenizer.encode(prompt)
        token_ids = encoded.ids

        if not token_ids:
            raise ValueError(
                "Tokenizer produced no tokens."
            )

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
            raise RuntimeError(
                "Model did not return a KV cache."
            )

        next_logits = logits[:, -1, :]

        for _ in range(config.max_new_tokens):
            if self._cache_is_full(cache):
                break

            next_token = self._sample_token(
                next_logits,
                config,
                generated_ids=generated_ids,
            )

            token_id = int(next_token.item())

            generated_ids.append(token_id)

            current_text = self.tokenizer.decode(
                generated_ids[len(input_ids[0]):],
                skip_special_tokens=True,
            )

            if self._contains_stop_sequence(
                current_text,
                config.stop_sequences,
            ):
                break

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

        Stop-sequence prefixes are buffered until they can be
        determined to be safe text or a complete stop sequence.
        """

        logits, cache = self.model(
            input_ids,
            use_cache=True,
        )

        if cache is None:
            raise RuntimeError(
                "Model did not return a KV cache."
            )

        next_logits = logits[:, -1, :]

        generated_ids: list[int] = []
        emitted_text = ""

        for _ in range(config.max_new_tokens):
            if self._cache_is_full(cache):
                break

            next_token = self._sample_token(
                next_logits,
                config,
                generated_ids=generated_ids,
            )

            token_id = int(next_token.item())

            generated_ids.append(token_id)

            current_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )

            safe_text, should_stop = (
                self._get_safe_stream_text(
                    current_text,
                    config.stop_sequences,
                )
            )

            chunk = safe_text[len(emitted_text):]

            if chunk:
                yield chunk

            emitted_text = safe_text

            if should_stop:
                break

            if token_id == self.eos_token_id:
                break

            position_offset = self._get_cache_length(
                cache,
                fallback=(
                    len(input_ids[0])
                    + len(generated_ids)
                    - 1
                ),
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
                generated_ids=generated_ids,
            )

            token_id = int(next_token.item())

            generated_ids.append(token_id)

            current_text = self.tokenizer.decode(
                generated_ids[len(input_ids[0]):],
                skip_special_tokens=True,
            )

            if self._contains_stop_sequence(
                current_text,
                config.stop_sequences,
            ):
                break

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
        emitted_text = ""

        prompt_length = len(input_ids[0])

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
                generated_ids=generated_ids,
            )

            token_id = int(next_token.item())

            generated_ids.append(token_id)

            generated_text = self.tokenizer.decode(
                generated_ids[prompt_length:],
                skip_special_tokens=True,
            )

            safe_text, should_stop = (
                self._get_safe_stream_text(
                    generated_text,
                    config.stop_sequences,
                )
            )

            chunk = safe_text[len(emitted_text):]

            if chunk:
                yield chunk

            emitted_text = safe_text

            if should_stop:
                break

            if token_id == self.eos_token_id:
                break

    # ------------------------------------------------------------------
    # Stream text helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_safe_stream_text(
        text: str,
        stop_sequences: tuple[str, ...],
    ) -> tuple[str, bool]:
        """
        Return text that is safe to stream.

        A suffix that could still become a stop sequence is held
        back until enough text arrives to determine its meaning.

        Returns:
            Tuple containing:

            - safe text that may be emitted
            - whether a complete stop sequence was found
        """

        if not stop_sequences:
            return text, False

        # First check for a complete stop sequence.
        earliest_stop_index: int | None = None

        for sequence in stop_sequences:
            stop_index = text.find(sequence)

            if stop_index == -1:
                continue

            if (
                earliest_stop_index is None
                or stop_index < earliest_stop_index
            ):
                earliest_stop_index = stop_index

        if earliest_stop_index is not None:
            return (
                text[:earliest_stop_index],
                True,
            )

        # No complete stop sequence exists yet.
        #
        # Hold back the longest suffix that is also a prefix
        # of a configured stop sequence.
        max_pending_length = 0

        for sequence in stop_sequences:
            max_length = min(
                len(sequence) - 1,
                len(text),
            )

            for length in range(
                max_length,
                0,
                -1,
            ):
                if text.endswith(sequence[:length]):
                    max_pending_length = max(
                        max_pending_length,
                        length,
                    )
                    break

        if max_pending_length == 0:
            return text, False

        return (
            text[:-max_pending_length],
            False,
        )

    @staticmethod
    def _contains_stop_sequence(
        text: str,
        stop_sequences: tuple[str, ...],
    ) -> bool:
        """Return True when text contains a complete stop sequence."""

        return any(
            sequence in text
            for sequence in stop_sequences
        )

    @staticmethod
    def _decode_generated_chunk(
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
        generated_ids: list[int] | None = None,
    ) -> Tensor:
        """
        Select the next token using either greedy decoding or sampling.

        Greedy decoding:
            Selects the token with the highest probability.

        Sampling:
            Applies temperature, repetition penalty,
            frequency/presence penalties, top-k, and top-p,
            then samples from the resulting probability
            distribution.
        """

        if logits.ndim != 2:
            raise ValueError(
                "Logits must have shape [batch, vocab_size]."
            )

        if not config.do_sample:
            return torch.argmax(
                logits,
                dim=-1,
                keepdim=True,
            )

        logits = logits / config.temperature

        if config.repetition_penalty > 1.0:
            logits = TextGenerator._apply_repetition_penalty(
                logits,
                generated_ids,
                config.repetition_penalty,
            )

        if (
            config.frequency_penalty > 0.0
            or config.presence_penalty > 0.0
        ):
            logits = (
                TextGenerator
                ._apply_frequency_presence_penalties(
                    logits,
                    generated_ids,
                    config.frequency_penalty,
                    config.presence_penalty,
                )
            )

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
    def _apply_repetition_penalty(
        logits: Tensor,
        generated_ids: list[int] | None,
        penalty: float,
    ) -> Tensor:
        """
        Penalize tokens that have already appeared.

        Positive logits are divided by the penalty.
        Negative logits are multiplied by the penalty.

        A penalty of 1.0 leaves logits unchanged.
        """

        if penalty < 1.0:
            raise ValueError(
                "repetition_penalty must be greater than or equal to 1.0."
            )

        if not generated_ids:
            return logits

        penalized_logits = logits.clone()

        token_ids = torch.tensor(
            list(set(generated_ids)),
            dtype=torch.long,
            device=logits.device,
        )

        selected_logits = penalized_logits[:, token_ids]

        selected_logits = torch.where(
            selected_logits > 0,
            selected_logits / penalty,
            selected_logits * penalty,
        )

        penalized_logits[:, token_ids] = selected_logits

        return penalized_logits

    @staticmethod
    def _apply_frequency_presence_penalties(
        logits: Tensor,
        generated_ids: list[int] | None,
        frequency_penalty: float,
        presence_penalty: float,
    ) -> Tensor:
        """
        Apply frequency and presence penalties.

        Frequency penalty:
            Penalizes tokens proportionally to how many times
            they have already appeared.

        Presence penalty:
            Applies a fixed penalty to every token that has
            appeared at least once.

        Formula:
            adjusted_logit =
                logit
                - frequency_penalty * count(token)
                - presence_penalty * I(token appeared)
        """

        if frequency_penalty < 0.0:
            raise ValueError(
                "frequency_penalty must be greater than or equal to 0.0."
            )

        if presence_penalty < 0.0:
            raise ValueError(
                "presence_penalty must be greater than or equal to 0.0."
            )

        if not generated_ids:
            return logits

        if (
            frequency_penalty == 0.0
            and presence_penalty == 0.0
        ):
            return logits

        penalized_logits = logits.clone()

        token_ids = torch.tensor(
            generated_ids,
            dtype=torch.long,
            device=logits.device,
        )

        counts = torch.bincount(
            token_ids,
            minlength=logits.size(-1),
        ).to(logits.dtype)

        penalty = (
            frequency_penalty * counts
            + presence_penalty
            * (counts > 0).to(logits.dtype)
        )

        penalized_logits = (
            penalized_logits
            - penalty.unsqueeze(0)
        )

        return penalized_logits

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
            raise TypeError(
                "prompt must be a string."
            )

        if not prompt.strip():
            raise ValueError(
                "prompt must not be empty."
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

        if not isinstance(config.do_sample, bool):
            raise TypeError(
                "do_sample must be a boolean."
            )

        if config.repetition_penalty < 1.0:
            raise ValueError(
                "repetition_penalty must be greater than or equal to 1.0."
            )

        if config.frequency_penalty < 0.0:
            raise ValueError(
                "frequency_penalty must be greater than or equal to 0.0."
            )

        if config.presence_penalty < 0.0:
            raise ValueError(
                "presence_penalty must be greater than or equal to 0.0."
            )

        if not isinstance(config.stop_sequences, tuple):
            raise TypeError(
                "stop_sequences must be a tuple of strings."
            )

        if any(
            not isinstance(sequence, str)
            for sequence in config.stop_sequences
        ):
            raise TypeError(
                "Every stop sequence must be a string."
            )

        if any(
            not sequence
            for sequence in config.stop_sequences
        ):
            raise ValueError(
                "stop_sequences cannot contain empty strings."
            )