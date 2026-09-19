from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.inference.config import GenerationConfig
from src.inference.generator import TextGenerator


class DummyTokenizer:
    """Minimal tokenizer used to test batched generation."""

    def __init__(self) -> None:
        self.vocabulary = {
            "<pad>": 0,
            "<bos>": 11,
            "<eos>": 12,
            "<unk>": 13,
            "a": 1,
            "b": 2,
            "c": 3,
            "d": 4,
            "e": 5,
            "f": 6,
            "g": 7,
            "h": 8,
            "i": 9,
            "j": 10,
        }

    def token_to_id(self, token: str) -> int | None:
        return self.vocabulary.get(token)

    def encode(self, text: str):
        class Encoding:
            def __init__(self, ids: list[int]) -> None:
                self.ids = ids

        ids = [
            self.vocabulary.get(
                char,
                self.vocabulary["<unk>"],
            )
            for char in text
            if not char.isspace()
        ]

        return Encoding(ids)

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        inverse = {
            token_id: token
            for token, token_id in self.vocabulary.items()
        }

        special_tokens = {
            self.vocabulary["<pad>"],
            self.vocabulary["<bos>"],
            self.vocabulary["<eos>"],
            self.vocabulary["<unk>"],
        }

        tokens: list[str] = []

        for token_id in ids:
            if (
                skip_special_tokens
                and token_id in special_tokens
            ):
                continue

            tokens.append(inverse.get(token_id, ""))

        return "".join(tokens)


class DummyKVCache:
    """Minimal KV-cache layer used by the dummy model."""

    def __init__(self, sequence_length: int) -> None:
        self.sequence_length = sequence_length


class DummyLayerKVCache:
    """
    Minimal replacement for LayerKVCache.

    TextGenerator only needs the cache to expose a `layers`
    collection whose first layer exposes `sequence_length`.
    """

    def __init__(self, sequence_length: int) -> None:
        self.layers = [DummyKVCache(sequence_length)]


class DummyModel:
    """
    Small deterministic model used to test batching.

    It intentionally does not perform real language modeling.
    It always predicts token ID 1 ("a").
    """

    def __init__(self, vocab_size: int = 14) -> None:
        self.vocab_size = vocab_size
        self.device = torch.device("cpu")

        self.config = SimpleNamespace(
            max_sequence_length=512,
        )

    def to(
        self,
        device: str | torch.device,
    ) -> "DummyModel":
        self.device = torch.device(device)
        return self

    def eval(self) -> "DummyModel":
        return self

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        use_cache: bool = False,
        past_key_values=None,
        position_offset: int = 0,
        cache=None,
    ):
        batch_size, sequence_length = input_ids.shape

        logits = torch.full(
            (
                batch_size,
                sequence_length,
                self.vocab_size,
            ),
            -100.0,
            dtype=torch.float32,
            device=input_ids.device,
        )

        # Deterministic prediction:
        # token ID 1 corresponds to "a".
        logits[:, :, 1] = 10.0

        if not use_cache:
            return logits, past_key_values

        if cache is None:
            current_length = sequence_length
        else:
            current_length = (
                cache.layers[0].sequence_length
                + sequence_length
            )

        new_cache = DummyLayerKVCache(
            sequence_length=current_length,
        )

        return logits, new_cache


def build_generator() -> TextGenerator:
    """Create a TextGenerator with deterministic test components."""
    tokenizer = DummyTokenizer()
    model = DummyModel()

    return TextGenerator(
        model,
        tokenizer,
    )


def test_generate_batch_returns_one_output_per_prompt() -> None:
    generator = build_generator()

    result = generator.generate_batch(
        ("abc", "de"),
        GenerationConfig(
            do_sample=False,
            max_new_tokens=3,
        ),
    )

    assert len(result.texts) == 2


def test_generate_batch_preserves_prompt_order() -> None:
    generator = build_generator()

    result = generator.generate_batch(
        (
            "abc",
            "de",
            "fgh",
        ),
        GenerationConfig(
            do_sample=False,
            max_new_tokens=2,
        ),
    )

    assert result.texts[0].startswith("abc")
    assert result.texts[1].startswith("de")
    assert result.texts[2].startswith("fgh")


def test_generate_batch_handles_different_prompt_lengths() -> None:
    generator = build_generator()

    result = generator.generate_batch(
        (
            "a",
            "abcdef",
            "ab",
        ),
        GenerationConfig(
            do_sample=False,
            max_new_tokens=2,
        ),
    )

    assert result.texts[0].startswith("a")
    assert result.texts[1].startswith("abcdef")
    assert result.texts[2].startswith("ab")


def test_generate_batch_with_zero_new_tokens() -> None:
    generator = build_generator()

    result = generator.generate_batch(
        (
            "abc",
            "de",
        ),
        GenerationConfig(
            do_sample=False,
            max_new_tokens=0,
        ),
    )

    assert result.texts == (
        "abc",
        "de",
    )


def test_generate_batch_rejects_empty_prompts() -> None:
    generator = build_generator()

    with pytest.raises(ValueError):
        generator.generate_batch(
            (),
            GenerationConfig(
                do_sample=False,
                max_new_tokens=2,
            ),
        )


def test_generate_batch_rejects_non_string_prompts() -> None:
    generator = build_generator()

    with pytest.raises(ValueError):
        generator.generate_batch(
            (
                "abc",
                123,
            ),
            GenerationConfig(
                do_sample=False,
                max_new_tokens=2,
            ),
        )


def test_generate_batch_rejects_blank_prompt() -> None:
    generator = build_generator()

    with pytest.raises(ValueError):
        generator.generate_batch(
            (
                "abc",
                "   ",
            ),
            GenerationConfig(
                do_sample=False,
                max_new_tokens=2,
            ),
        )
