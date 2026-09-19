from __future__ import annotations

import pytest
import torch

from src.inference.generator import GenerationConfig, TextGenerator
from src.inference.sampling import (
    apply_frequency_presence_penalties,
    apply_repetition_penalty,
    sample_token,
)



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
        inverse = {
            value: key
            for key, value in self._vocab.items()
        }

        tokens = [
            inverse[token_id]
            for token_id in ids
        ]

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

        # Always make token 2 the highest-probability token.
        logits[..., 2] = 10.0

        if not use_cache:
            return logits, None

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


# ----------------------------------------------------------------------
# Generation configuration
# ----------------------------------------------------------------------


def test_generation_config_defaults():
    config = GenerationConfig()

    assert config.max_new_tokens == 100
    assert config.temperature == 0.8
    assert config.top_k == 50
    assert config.top_p == 0.9
    assert config.do_sample is True


def test_greedy_generation_config():
    config = GenerationConfig(
        do_sample=False,
    )

    assert config.do_sample is False


# ----------------------------------------------------------------------
# Prompt validation
# ----------------------------------------------------------------------


def test_empty_prompt_rejected(generator):
    with pytest.raises(ValueError):
        generator.generate("")


def test_whitespace_prompt_rejected(generator):
    with pytest.raises(ValueError):
        generator.generate("   ")


def test_non_string_prompt_rejected(generator):
    with pytest.raises(TypeError):
        generator.generate(123)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Generation configuration validation
# ----------------------------------------------------------------------


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


def test_invalid_do_sample_type(generator):
    config = GenerationConfig(
        do_sample=1,  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError):
        generator.generate("hello", config)


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------


def test_greedy_decoding_selects_highest_logit():
    logits = torch.tensor(
        [[1.0, 5.0, 3.0, 2.0]]
    )

    config = GenerationConfig(
        do_sample=False,
    )

    result = sample_token(
        logits,
        config,
    )

    assert result.shape == (1, 1)
    assert result.item() == 1


def test_sampling_with_top_k():
    logits = torch.tensor(
        [[1.0, 5.0, 3.0, 2.0]]
    )

    result = TextGenerator._apply_top_k(
        logits,
        top_k=2,
    )

    assert torch.isneginf(result[0, 0])
    assert not torch.isneginf(result[0, 1])
    assert not torch.isneginf(result[0, 2])
    assert torch.isneginf(result[0, 3])


def test_sampling_with_top_p():
    logits = torch.tensor(
        [[5.0, 4.0, 1.0, 0.0]]
    )

    result = TextGenerator._apply_top_p(
        logits,
        top_p=0.80,
    )

    assert torch.isfinite(result).any()

def test_repetition_penalty_reduces_positive_logit():
    logits = torch.tensor(
        [[2.0, 5.0, 3.0, 1.0]]
    )

    result = apply_repetition_penalty(
        logits,
        generated_ids=[1, 2],
        penalty=2.0,
    )

    assert result[0, 0].item() == 2.0
    assert result[0, 1].item() == 2.5
    assert result[0, 2].item() == 1.5
    assert result[0, 3].item() == 1.0


def test_repetition_penalty_preserves_logits_at_one():
    logits = torch.tensor(
        [[2.0, -4.0, 3.0, 1.0]]
    )

    result = apply_repetition_penalty(
        logits,
        generated_ids=[0, 1, 2],
        penalty=1.0,
    )

    assert torch.equal(result, logits)


def test_invalid_repetition_penalty(generator):
    config = GenerationConfig(
        repetition_penalty=0.5,
    )

    with pytest.raises(ValueError):
        generator.generate("hello", config)


def test_frequency_penalty():
    logits = torch.tensor(
        [[5.0, 4.0, 3.0, 2.0]]
    )

    result = apply_frequency_presence_penalties(
        logits,
        generated_ids=[1, 1, 2],
        frequency_penalty=1.0,
        presence_penalty=0.0,
    )

    assert result[0, 0].item() == 5.0
    assert result[0, 1].item() == 2.0
    assert result[0, 2].item() == 2.0
    assert result[0, 3].item() == 2.0


def test_presence_penalty():
    logits = torch.tensor(
        [[5.0, 4.0, 3.0, 2.0]]
    )

    result = apply_frequency_presence_penalties(
        logits,
        generated_ids=[1, 1, 2],
        frequency_penalty=0.0,
        presence_penalty=1.0,
    )

    assert result[0, 0].item() == 5.0
    assert result[0, 1].item() == 3.0
    assert result[0, 2].item() == 2.0
    assert result[0, 3].item() == 2.0


def test_frequency_and_presence_penalty():
    logits = torch.tensor(
        [[5.0, 4.0, 3.0, 2.0]]
    )

    result = apply_frequency_presence_penalties(
        logits,
        generated_ids=[1, 1, 2],
        frequency_penalty=0.5,
        presence_penalty=1.0,
    )

    assert result[0, 0].item() == 5.0
    assert result[0, 1].item() == 2.0
    assert result[0, 2].item() == 1.5
    assert result[0, 3].item() == 2.0


def test_frequency_presence_penalties_zero():
    logits = torch.tensor(
        [[5.0, 4.0, 3.0, 2.0]]
    )

    result = apply_frequency_presence_penalties(
        logits,
        generated_ids=[1, 2],
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    assert torch.equal(result, logits)


def test_invalid_frequency_penalty(generator):
    config = GenerationConfig(
        frequency_penalty=-0.1,
    )

    with pytest.raises(ValueError):
        generator.generate("hello", config)


def test_invalid_presence_penalty(generator):
    config = GenerationConfig(
        presence_penalty=-0.1,
    )

    with pytest.raises(ValueError):
        generator.generate("hello", config)

def test_stop_sequence_detection():
    assert TextGenerator._contains_stop_sequence(
        "hello world END",
        ("END",),
    )


def test_stop_sequence_not_found():
    assert not TextGenerator._contains_stop_sequence(
        "hello world",
        ("END",),
    )


def test_multiple_stop_sequences():
    assert TextGenerator._contains_stop_sequence(
        "hello USER: something",
        ("END", "USER:"),
    )


def test_empty_stop_sequences_do_not_stop():
    assert not TextGenerator._contains_stop_sequence(
        "hello world",
        (),
    )


def test_invalid_stop_sequences_type(generator):
    config = GenerationConfig(
        stop_sequences=["END"],
    )

    with pytest.raises(TypeError):
        generator.generate("hello", config)


def test_invalid_stop_sequence_item(generator):
    config = GenerationConfig(
        stop_sequences=("END", 123),
    )

    with pytest.raises(TypeError):
        generator.generate("hello", config)


def test_empty_stop_sequence(generator):
    config = GenerationConfig(
        stop_sequences=("END", ""),
    )

    with pytest.raises(ValueError):
        generator.generate("hello", config)
# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------


def test_deterministic_generation_with_cache(generator):
    config = GenerationConfig(
        max_new_tokens=3,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        do_sample=True,
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
        do_sample=True,
    )

    output = generator.generate(
        "hello",
        config=config,
        use_cache=False,
    )

    assert isinstance(output, str)
    assert "hello" in output
    assert "world" in output


def test_greedy_generation_with_cache(generator):
    config = GenerationConfig(
        max_new_tokens=3,
        do_sample=False,
    )

    output = generator.generate(
        "hello",
        config=config,
        use_cache=True,
    )

    assert output == "hello world hello hello hello"


def test_greedy_generation_without_cache(generator):
    config = GenerationConfig(
        max_new_tokens=3,
        do_sample=False,
    )

    output = generator.generate(
        "hello",
        config=config,
        use_cache=False,
    )

    assert output == "hello world hello hello hello"

def test_special_token_suppression_blocks_pad_and_bos():
    logits = torch.tensor(
        [[10.0, 9.0, 1.0, 2.0, 3.0]],
        dtype=torch.float32,
    )

    config = GenerationConfig(do_sample=False)

    next_token = sample_token(
        logits,
        config,
        forbidden_token_ids=(0, 1),
    )

    assert next_token.item() == 4


def test_special_token_suppression_allows_eos():
    logits = torch.tensor(
        [[1.0, 2.0, 10.0, 3.0]],
        dtype=torch.float32,
    )

    config = GenerationConfig(do_sample=False)

    next_token = sample_token(
        logits,
        config,
        forbidden_token_ids=(0, 1),
    )

    assert next_token.item() == 2


def test_special_token_suppression_allows_unk():
    logits = torch.tensor(
        [[1.0, 2.0, 3.0, 10.0]],
        dtype=torch.float32,
    )

    config = GenerationConfig(do_sample=False)

    next_token = sample_token(
        logits,
        config,
        forbidden_token_ids=(0, 1),
    )

    assert next_token.item() == 3


def test_special_token_suppression_does_not_mutate_logits():
    logits = torch.tensor(
        [[10.0, 9.0, 1.0, 2.0]],
        dtype=torch.float32,
    )

    original = logits.clone()

    config = GenerationConfig(do_sample=False)

    sample_token(
        logits,
        config,
        forbidden_token_ids=(0, 1),
    )

    assert torch.equal(logits, original)


def test_special_token_suppression_handles_invalid_ids():
    logits = torch.tensor(
        [[1.0, 2.0, 10.0]],
        dtype=torch.float32,
    )

    config = GenerationConfig(do_sample=False)

    next_token = sample_token(
        logits,
        config,
        forbidden_token_ids=(-1, 99),
    )

    assert next_token.item() == 2


def test_special_token_suppression_works_with_sampling():
    logits = torch.tensor(
        [[10.0, 9.0, 1.0, 2.0]],
        dtype=torch.float32,
    )

    config = GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )

    torch.manual_seed(42)

    next_token = sample_token(
        logits,
        config,
        forbidden_token_ids=(0, 1),
    )

    assert next_token.item() in {2, 3}
