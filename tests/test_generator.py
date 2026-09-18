from __future__ import annotations

import pytest
import torch

from src.inference.generator import GenerationConfig, TextGenerator


class DummyTokenizer:
    def __init__(self) -> None:
        self._vocab = {
            "<bos>": 0,
            "<eos>": 1,
            "hello": 2,
            "world": 3,
        }

    def token_to_id(self, token: str) -> int | None:
        return self._vocab.get(token)

    def encode(self, text: str):
        class Encoded:
            ids = [2, 3]

        return Encoded()

    def decode(self, ids, skip_special_tokens=True) -> str:
        inverse = {value: key for key, value in self._vocab.items()}
        tokens = [inverse[token_id] for token_id in ids]
        if skip_special_tokens:
            tokens = [
                token
                for token in tokens
                if token not in {"<bos>", "<eos>"}
            ]
        return " ".join(tokens)


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 4) -> None:
        super().__init__()
        self.vocab_size = vocab_size

        class Config:
            max_sequence_length = 16

        self.config = Config()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_offset: int = 0,
        cache=None,
        use_cache: bool = False,
    ):
        batch_size, sequence_length = input_ids.shape

        logits = torch.zeros(
            batch_size,
            sequence_length,
            self.vocab_size,
            device=input_ids.device,
        )

        # Always select token 2 deterministically.
        logits[..., 2] = 10.0

        if not use_cache:
            return logits, None

        # Minimal fake KV cache compatible with the generator.
        class FakeLayerCache:
            def __init__(self, sequence_length: int):
                self.sequence_length = sequence_length

        class FakeCache:
            def __init__(self, sequence_length: int):
                self.layers = [
                    FakeLayerCache(sequence_length)
                ]

        previous_length = 0

        if cache is not None and cache.layers[0] is not None:
            previous_length = cache.layers[0].sequence_length

        updated_cache = FakeCache(
            previous_length + sequence_length
        )

        return logits, updated_cache


@pytest.fixture
def tokenizer():
    return DummyTokenizer()


@pytest.fixture
def model():
    return DummyModel()


@pytest.fixture
def generator(model, tokenizer):
    return TextGenerator(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )


def test_generation_config_defaults():
    config = GenerationConfig()

    assert config.max_new_tokens == 100
    assert config.temperature == 0.8
    assert config.top_k == 50
    assert config.top_p == 0.9


def test_empty_prompt_rejected(generator):
    with pytest.raises(ValueError):
        generator.generate("")


def test_whitespace_prompt_rejected(generator):
    with pytest.raises(ValueError):
        generator.generate("   ")


def test_non_string_prompt_rejected(generator):
    with pytest.raises(TypeError):
        generator.generate(123)  # type: ignore[arg-type]


def test_invalid_max_new_tokens(generator):
    config = GenerationConfig(max_new_tokens=-1)

    with pytest.raises(ValueError):
        generator.generate("hello", config)


def test_invalid_temperature(generator):
    config = GenerationConfig(temperature=0.0)

    with pytest.raises(ValueError):
        generator.generate("hello", config)


def test_invalid_top_k(generator):
    config = GenerationConfig(top_k=-1)

    with pytest.raises(ValueError):
        generator.generate("hello", config)


def test_invalid_top_p(generator):
    config = GenerationConfig(top_p=0.0)

    with pytest.raises(ValueError):
        generator.generate("hello", config)


def test_top_k():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])

    result = TextGenerator._apply_top_k(logits, top_k=2)

    assert torch.isneginf(result[0, 0])
    assert not torch.isneginf(result[0, 1])
    assert not torch.isneginf(result[0, 2])
    assert torch.isneginf(result[0, 3])


def test_top_p():
    logits = torch.tensor([[5.0, 4.0, 1.0, 0.0]])

    result = TextGenerator._apply_top_p(
        logits,
        top_p=0.80,
    )

    assert torch.isfinite(result).any()


def test_deterministic_generation_with_cache(generator):
    config = GenerationConfig(
        max_new_tokens=3,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
    )

    output = generator.generate(
        "hello",
        config=config,
        use_cache=True,
    )

    assert isinstance(output, str)
    assert "hello" in output
    assert "world" in output


def test_deterministic_generation_without_cache(generator):
    config = GenerationConfig(
        max_new_tokens=3,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
    )

    output = generator.generate(
        "hello",
        config=config,
        use_cache=False,
    )

    assert isinstance(output, str)
    assert "hello" in output
    assert "world" in output