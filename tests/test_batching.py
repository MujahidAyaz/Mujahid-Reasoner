import pytest
import torch

from src.inference.batching import (
    BatchGenerationInput,
    BatchGenerationOutput,
    tokenize_batch,
)


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    def __init__(self):
        self.mapping = {
            "hello": [10, 11],
            "hello world": [10, 11, 12],
            "test": [20],
        }

    def encode(self, text):
        return FakeEncoding(self.mapping[text])


def test_batch_generation_input_accepts_valid_prompts():
    batch = BatchGenerationInput(
        prompts=("hello", "test"),
    )

    assert batch.prompts == ("hello", "test")


def test_batch_generation_input_rejects_empty_batch():
    with pytest.raises(ValueError, match="prompts must not be empty"):
        BatchGenerationInput(prompts=())


def test_batch_generation_input_rejects_non_string_prompt():
    with pytest.raises(ValueError, match="all prompts must be strings"):
        BatchGenerationInput(prompts=("hello", 123))


def test_batch_generation_input_rejects_empty_prompt():
    with pytest.raises(ValueError, match="all prompts must be non-empty"):
        BatchGenerationInput(prompts=("hello", "   "))


def test_batch_generation_output_accepts_valid_texts():
    output = BatchGenerationOutput(
        texts=("hello", "world"),
    )

    assert output.texts == ("hello", "world")


def test_batch_generation_output_rejects_empty_output():
    with pytest.raises(ValueError, match="texts must not be empty"):
        BatchGenerationOutput(texts=())


def test_tokenize_batch_pads_variable_length_prompts():
    tokenizer = FakeTokenizer()

    result = tokenize_batch(
        tokenizer=tokenizer,
        prompts=("hello", "hello world", "test"),
        pad_token_id=0,
        device=torch.device("cpu"),
    )

    assert result.input_ids.shape == (3, 3)

    assert result.input_ids.tolist() == [
        [10, 11, 0],
        [10, 11, 12],
        [20, 0, 0],
    ]

    assert result.attention_mask.tolist() == [
        [True, True, False],
        [True, True, True],
        [True, False, False],
    ]

    assert result.prompt_lengths == (2, 3, 1)


def test_tokenize_batch_uses_requested_device():
    tokenizer = FakeTokenizer()

    result = tokenize_batch(
        tokenizer=tokenizer,
        prompts=("hello", "test"),
        pad_token_id=0,
        device=torch.device("cpu"),
    )

    assert result.input_ids.device.type == "cpu"
    assert result.attention_mask.device.type == "cpu"


def test_tokenize_batch_rejects_empty_batch():
    tokenizer = FakeTokenizer()

    with pytest.raises(ValueError, match="prompts must not be empty"):
        tokenize_batch(
            tokenizer=tokenizer,
            prompts=(),
            pad_token_id=0,
            device=torch.device("cpu"),
        )


def test_tokenize_batch_rejects_empty_tokenization():
    class EmptyTokenizer:
        def encode(self, text):
            return FakeEncoding([])

    with pytest.raises(
        ValueError,
        match="all prompts must tokenize to at least one token",
    ):
        tokenize_batch(
            tokenizer=EmptyTokenizer(),
            prompts=("hello",),
            pad_token_id=0,
            device=torch.device("cpu"),
        )
