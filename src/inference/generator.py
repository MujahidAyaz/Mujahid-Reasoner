from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from tokenizers import Tokenizer

from src.inference.batching import (
    BatchGenerationOutput,
    group_prompt_indices_by_length,
)
from src.inference.config import GenerationConfig
from src.inference.sampling import (
    apply_top_k,
    apply_top_p,
    sample_token,
)
from src.model.cache import LayerKVCache
from src.model.model import MujahidReasonerModel


@dataclass(frozen=True)
class GenerationResult:
    """
    Result of single-prompt generation.

    `text` contains the original prompt followed by generated text.

    `generated_token_ids` contains only newly generated token IDs,
    excluding the prompt.

    `generated_token_count` is the exact number of tokens produced
    by the model.
    """

    text: str
    generated_token_ids: tuple[int, ...]
    generated_token_count: int

    def __post_init__(self) -> None:
        if self.generated_token_count < 0:
            raise ValueError(
                "generated_token_count must be non-negative."
            )

        if self.generated_token_count != len(
            self.generated_token_ids
        ):
            raise ValueError(
                "generated_token_count must match "
                "generated_token_ids length."
            )


@dataclass(frozen=True)
class BatchGenerationStats:
    """
    Exact statistics for batched generation.

    `generated_token_counts[i]` corresponds to prompt `i`.
    """

    output: BatchGenerationOutput
    generated_token_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.output.texts) != len(
            self.generated_token_counts
        ):
            raise ValueError(
                "Number of generated token counts must match "
                "number of outputs."
            )

        if any(
            count < 0
            for count in self.generated_token_counts
        ):
            raise ValueError(
                "generated token counts must be non-negative."
            )

    @property
    def total_generated_tokens(self) -> int:
        """Return the exact number of generated tokens across the batch."""

        return sum(self.generated_token_counts)


class TextGenerator:
    """
    Autoregressive text generator with optional KV caching.

    Supports:

    - Standard full-text generation.
    - Exact generation statistics.
    - Batched generation.
    - Exact batched generation statistics.
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
    - Minimum generation length.
    - Maximum generation length.
    - Special-token suppression.

    The existing `generate()` and `generate_batch()` APIs are
    preserved.
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

        self.pad_token_id = tokenizer.token_to_id("<pad>")
        self.bos_token_id = tokenizer.token_to_id("<bos>")
        self.eos_token_id = tokenizer.token_to_id("<eos>")
        self.unk_token_id = tokenizer.token_to_id("<unk>")

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
        """Generate complete text from a prompt."""

        result = self.generate_with_stats(
            prompt,
            config=config,
            use_cache=use_cache,
        )

        return result.text

    @torch.inference_mode()
    def generate_with_stats(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        use_cache: bool = True,
    ) -> GenerationResult:
        """
        Generate text and return exact generation statistics.

        The returned text contains the prompt followed by generated
        text.

        The returned token IDs and token count contain only newly
        generated tokens and exclude the prompt.
        """

        self._validate_prompt(prompt)

        if config is None:
            config = GenerationConfig()

        self._validate_generation_config(config)

        token_ids = self._encode_prompt(prompt)
        token_ids = self._truncate_prompt(token_ids)

        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )

        if use_cache:
            full_generated_ids = self._generate_with_cache(
                input_ids,
                config,
            )
        else:
            full_generated_ids = self._generate_without_cache(
                input_ids,
                config,
            )

        generated_ids = full_generated_ids[
            len(token_ids):
        ]

        text = self.tokenizer.decode(
            full_generated_ids,
            skip_special_tokens=True,
        )

        return GenerationResult(
            text=text,
            generated_token_ids=tuple(
                generated_ids
            ),
            generated_token_count=len(
                generated_ids
            ),
        )

    # ------------------------------------------------------------------
    # Batched generation API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: tuple[str, ...],
        config: GenerationConfig | None = None,
    ) -> BatchGenerationOutput:
        """
        Generate text for multiple prompts.

        Prompts are grouped by tokenized length so each group can safely
        share the current KV-cache implementation.
        """

        result = self.generate_batch_with_stats(
            prompts,
            config=config,
        )

        return result.output

    @torch.inference_mode()
    def generate_batch_with_stats(
        self,
        prompts: tuple[str, ...],
        config: GenerationConfig | None = None,
    ) -> BatchGenerationStats:
        """
        Generate multiple prompts and return exact token statistics.

        Prompts with equal tokenized lengths are processed together.
        Results are restored to the original prompt ordering.
        """

        self._validate_prompts(prompts)

        if config is None:
            config = GenerationConfig()

        self._validate_generation_config(config)

        encoded_prompts = [
            self._truncate_prompt(
                self._encode_prompt(prompt)
            )
            for prompt in prompts
        ]

        groups = group_prompt_indices_by_length(
            encoded_prompts,
        )

        results: list[str | None] = [
            None
        ] * len(prompts)

        token_counts: list[int | None] = [
            None
        ] * len(prompts)

        for indices in groups.values():
            batch_input_ids = torch.tensor(
                [
                    encoded_prompts[index]
                    for index in indices
                ],
                dtype=torch.long,
                device=self.device,
            )

            generated_ids = self._generate_batch_with_cache(
                batch_input_ids,
                config,
            )

            prompt_length = batch_input_ids.size(1)

            for row, original_index in enumerate(indices):
                results[original_index] = self.tokenizer.decode(
                    generated_ids[row],
                    skip_special_tokens=True,
                )

                token_counts[original_index] = (
                    len(generated_ids[row])
                    - prompt_length
                )

        if any(
            result is None
            for result in results
        ):
            raise RuntimeError(
                "Batched generation failed to produce "
                "all outputs."
            )

        if any(
            count is None
            for count in token_counts
        ):
            raise RuntimeError(
                "Batched generation failed to produce "
                "all token counts."
            )

        output = BatchGenerationOutput(
            texts=tuple(
                result
                for result in results
                if result is not None
            ),
        )

        return BatchGenerationStats(
            output=output,
            generated_token_counts=tuple(
                count
                for count in token_counts
                if count is not None
            ),
        )

    def _generate_batch_with_cache(
        self,
        input_ids: Tensor,
        config: GenerationConfig,
    ) -> list[list[int]]:
        """
        Autoregressively generate multiple sequences in parallel.

        All rows must have the same prompt length.

        Finished rows receive EOS internally so the shared KV cache
        remains synchronized.
        """

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape "
                "[batch, sequence_length]."
            )

        batch_size = input_ids.size(0)

        if batch_size == 0:
            raise ValueError(
                "input_ids batch must not be empty."
            )

        generated_ids = [
            row.tolist()
            for row in input_ids
        ]

        prompt_length = input_ids.size(1)

        logits, cache = self.model(
            input_ids,
            use_cache=True,
        )

        if cache is None:
            raise RuntimeError(
                "Model did not return a KV cache."
            )

        next_logits = logits[:, -1, :]

        finished = [
            False
        ] * batch_size

        generated_token_counts = [
            0
        ] * batch_size

        forbidden_token_ids = (
            self._get_forbidden_token_ids()
        )

        for _ in range(
            config.max_new_tokens
        ):
            if self._cache_is_full(cache):
                break

            active_indices = [
                index
                for index, is_finished in enumerate(
                    finished
                )
                if not is_finished
            ]

            next_tokens = torch.full(
                (batch_size, 1),
                self.eos_token_id,
                dtype=torch.long,
                device=self.device,
            )

            if active_indices:
                active_logits = next_logits[
                    active_indices
                ]

                active_histories = [
                    generated_ids[index]
                    for index in active_indices
                ]

                sampled_tokens = sample_token(
                    active_logits,
                    config,
                    generated_ids=active_histories,
                    forbidden_token_ids=forbidden_token_ids,
                )

                for row, batch_index in enumerate(
                    active_indices
                ):
                    next_tokens[
                        batch_index,
                        0,
                    ] = sampled_tokens[
                        row,
                        0,
                    ]

            for index in range(batch_size):
                if finished[index]:
                    continue

                token_id = int(
                    next_tokens[
                        index,
                        0,
                    ].item()
                )

                generated_ids[index].append(
                    token_id
                )

                generated_token_counts[index] += 1

                generated_text = self.tokenizer.decode(
                    generated_ids[index][
                        prompt_length:
                    ],
                    skip_special_tokens=True,
                )

                can_stop = self._can_stop(
                    generated_token_counts[index],
                    config.min_new_tokens,
                )

                if (
                    can_stop
                    and self._contains_stop_sequence(
                        generated_text,
                        config.stop_sequences,
                    )
                ):
                    finished[index] = True

                if (
                    token_id == self.eos_token_id
                    and can_stop
                ):
                    finished[index] = True

            if all(finished):
                break

            position_offset = self._get_cache_length(
                cache,
                fallback=input_ids.size(1),
            )

            logits, cache = self.model(
                next_tokens,
                position_offset=position_offset,
                cache=cache,
                use_cache=True,
            )

            if cache is None:
                raise RuntimeError(
                    "Model stopped returning "
                    "the KV cache."
                )

            next_logits = logits[:, -1, :]

        return generated_ids

    # ------------------------------------------------------------------
    # Streaming generation API
    # ------------------------------------------------------------------

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

        Partial stop-sequence prefixes are buffered internally.
        """

        self._validate_prompt(prompt)

        if config is None:
            config = GenerationConfig()

        self._validate_generation_config(config)

        token_ids = self._encode_prompt(prompt)
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
        prompt_length = len(generated_ids)

        logits, cache = self.model(
            input_ids,
            use_cache=True,
        )

        if cache is None:
            raise RuntimeError(
                "Model did not return a KV cache."
            )

        next_logits = logits[:, -1, :]

        forbidden_token_ids = (
            self._get_forbidden_token_ids()
        )

        for _ in range(
            config.max_new_tokens
        ):
            if self._cache_is_full(cache):
                break

            next_token = sample_token(
                next_logits,
                config,
                generated_ids=generated_ids,
                forbidden_token_ids=forbidden_token_ids,
            )

            token_id = int(
                next_token.item()
            )

            generated_ids.append(
                token_id
            )

            generated_token_count = (
                len(generated_ids)
                - prompt_length
            )

            current_text = self.tokenizer.decode(
                generated_ids[
                    prompt_length:
                ],
                skip_special_tokens=True,
            )

            can_stop = self._can_stop(
                generated_token_count,
                config.min_new_tokens,
            )

            if (
                can_stop
                and self._contains_stop_sequence(
                    current_text,
                    config.stop_sequences,
                )
            ):
                break

            if (
                token_id == self.eos_token_id
                and can_stop
            ):
                break

            position_offset = self._get_cache_length(
                cache,
                fallback=len(generated_ids) - 1,
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
                    "Model stopped returning "
                    "the KV cache."
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
        """Stream generation using incremental KV-cache decoding."""

        logits, cache = self.model(
            input_ids,
            use_cache=True,
        )

        if cache is None:
            raise RuntimeError(
                "Model did not return a KV cache."
            )

        next_logits = logits[:, -1, :]

        generated_ids = input_ids[0].tolist()
        prompt_length = len(generated_ids)

        emitted_text = ""

        forbidden_token_ids = (
            self._get_forbidden_token_ids()
        )

        for _ in range(
            config.max_new_tokens
        ):
            if self._cache_is_full(cache):
                break

            next_token = sample_token(
                next_logits,
                config,
                generated_ids=generated_ids,
                forbidden_token_ids=forbidden_token_ids,
            )

            token_id = int(
                next_token.item()
            )

            generated_ids.append(
                token_id
            )

            generated_token_count = (
                len(generated_ids)
                - prompt_length
            )

            current_text = self.tokenizer.decode(
                generated_ids[
                    prompt_length:
                ],
                skip_special_tokens=True,
            )

            can_stop = self._can_stop(
                generated_token_count,
                config.min_new_tokens,
            )

            if can_stop:
                (
                    safe_text,
                    should_stop,
                ) = self._get_safe_stream_text(
                    current_text,
                    config.stop_sequences,
                )
            else:
                safe_text = current_text
                should_stop = False

            chunk = safe_text[
                len(emitted_text):
            ]

            if chunk:
                yield chunk

            emitted_text = safe_text

            if should_stop:
                break

            if (
                token_id == self.eos_token_id
                and can_stop
            ):
                break

            position_offset = self._get_cache_length(
                cache,
                fallback=(
                    len(input_ids[0])
                    + generated_token_count
                    - 1
                ),
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
                    "Model stopped returning "
                    "the KV cache."
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

        Retained as a correctness and benchmarking
        reference implementation.
        """

        generated_ids = input_ids[0].tolist()
        prompt_length = len(generated_ids)

        forbidden_token_ids = (
            self._get_forbidden_token_ids()
        )

        for _ in range(
            config.max_new_tokens
        ):
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
                if isinstance(
                    model_output,
                    tuple,
                )
                else model_output
            )

            next_logits = logits[:, -1, :]

            next_token = sample_token(
                next_logits,
                config,
                generated_ids=generated_ids,
                forbidden_token_ids=forbidden_token_ids,
            )

            token_id = int(
                next_token.item()
            )

            generated_ids.append(
                token_id
            )

            generated_token_count = (
                len(generated_ids)
                - prompt_length
            )

            current_text = self.tokenizer.decode(
                generated_ids[
                    prompt_length:
                ],
                skip_special_tokens=True,
            )

            can_stop = self._can_stop(
                generated_token_count,
                config.min_new_tokens,
            )

            if (
                can_stop
                and self._contains_stop_sequence(
                    current_text,
                    config.stop_sequences,
                )
            ):
                break

            if (
                token_id == self.eos_token_id
                and can_stop
            ):
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
        """Stream generation without KV caching."""

        generated_ids = input_ids[0].tolist()
        prompt_length = len(generated_ids)

        emitted_text = ""

        forbidden_token_ids = (
            self._get_forbidden_token_ids()
        )

        for _ in range(
            config.max_new_tokens
        ):
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
                if isinstance(
                    model_output,
                    tuple,
                )
                else model_output
            )

            next_logits = logits[:, -1, :]

            next_token = sample_token(
                next_logits,
                config,
                generated_ids=generated_ids,
                forbidden_token_ids=forbidden_token_ids,
            )

            token_id = int(
                next_token.item()
            )

            generated_ids.append(
                token_id
            )

            generated_token_count = (
                len(generated_ids)
                - prompt_length
            )

            generated_text = self.tokenizer.decode(
                generated_ids[
                    prompt_length:
                ],
                skip_special_tokens=True,
            )

            can_stop = self._can_stop(
                generated_token_count,
                config.min_new_tokens,
            )

            if can_stop:
                (
                    safe_text,
                    should_stop,
                ) = self._get_safe_stream_text(
                    generated_text,
                    config.stop_sequences,
                )
            else:
                safe_text = generated_text
                should_stop = False

            chunk = safe_text[
                len(emitted_text):
            ]

            if chunk:
                yield chunk

            emitted_text = safe_text

            if should_stop:
                break

            if (
                token_id == self.eos_token_id
                and can_stop
            ):
                break

    # ------------------------------------------------------------------
    # Prompt / token helpers
    # ------------------------------------------------------------------

    def _encode_prompt(
        self,
        prompt: str,
    ) -> list[int]:
        """Encode a prompt into tokenizer IDs."""

        encoded = self.tokenizer.encode(prompt)
        token_ids = encoded.ids

        if not token_ids:
            raise ValueError(
                "Tokenizer produced no tokens."
            )

        return token_ids

    def _truncate_prompt(
        self,
        token_ids: list[int],
    ) -> list[int]:
        """
        Keep tokens inside the model context window.

        If the prompt exceeds the model context window, retain the
        most recent tokens.
        """

        max_length = (
            self.model.config.max_sequence_length
        )

        if len(token_ids) <= max_length:
            return token_ids

        return token_ids[
            -max_length:
        ]

    def _get_forbidden_token_ids(
        self,
    ) -> tuple[int, ...]:
        """Return special tokens that must never be generated."""

        token_ids: list[int] = []

        if self.pad_token_id is not None:
            token_ids.append(
                self.pad_token_id
            )

        if self.bos_token_id is not None:
            token_ids.append(
                self.bos_token_id
            )

        return tuple(token_ids)

    @staticmethod
    def _can_stop(
        generated_token_count: int,
        min_new_tokens: int,
    ) -> bool:
        """Return whether generation is allowed to terminate."""

        return (
            generated_token_count
            >= min_new_tokens
        )

    # ------------------------------------------------------------------
    # Stream text helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_safe_stream_text(
        text: str,
        stop_sequences: tuple[str, ...],
    ) -> tuple[str, bool]:
        """Return text that is safe to stream."""

        if not stop_sequences:
            return text, False

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
                if text.endswith(
                    sequence[:length]
                ):
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
    # Backward-compatible sampling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_top_k(
        logits: Tensor,
        top_k: int,
    ) -> Tensor:
        """Backward-compatible wrapper for top-k filtering."""

        return apply_top_k(
            logits,
            top_k,
        )

    @staticmethod
    def _apply_top_p(
        logits: Tensor,
        top_p: float,
    ) -> Tensor:
        """Backward-compatible wrapper for top-p filtering."""

        return apply_top_p(
            logits,
            top_p,
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

    @classmethod
    def _validate_prompts(
        cls,
        prompts: Sequence[str],
    ) -> None:
        if not prompts:
            raise ValueError(
                "prompts must not be empty."
            )

        if not all(
            isinstance(prompt, str)
            for prompt in prompts
        ):
            raise ValueError(
                "all prompts must be strings."
            )

        if not all(
            prompt.strip()
            for prompt in prompts
        ):
            raise ValueError(
                "all prompts must be non-empty."
            )

    @staticmethod
    def _validate_generation_config(
        config: GenerationConfig,
    ) -> None:
        if config.min_new_tokens < 0:
            raise ValueError(
                "min_new_tokens must be non-negative."
            )

        if config.max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens must be non-negative."
            )

        if (
            config.max_new_tokens
            < config.min_new_tokens
        ):
            raise ValueError(
                "max_new_tokens must be greater than "
                "or equal to min_new_tokens."
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

        if not isinstance(
            config.do_sample,
            bool,
        ):
            raise TypeError(
                "do_sample must be a boolean."
            )

        if config.repetition_penalty < 1.0:
            raise ValueError(
                "repetition_penalty must be greater "
                "than or equal to 1.0."
            )

        if config.frequency_penalty < 0.0:
            raise ValueError(
                "frequency_penalty must be greater "
                "than or equal to 0.0."
            )

        if config.presence_penalty < 0.0:
            raise ValueError(
                "presence_penalty must be greater "
                "than or equal to 0.0."
            )

        if not isinstance(
            config.stop_sequences,
            tuple,
        ):
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
                "stop_sequences cannot contain "
                "empty strings."
            )