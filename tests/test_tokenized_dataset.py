from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tokenized_dataset import PackedTokenDataset


TOKEN_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tokens"
)

TRAIN_FILE = TOKEN_DIR / "train.bin"
VALIDATION_FILE = TOKEN_DIR / "validation.bin"

SEQUENCE_LENGTH = 128


def create_dataset() -> PackedTokenDataset:
    """Create a dataset using the prepared token stream."""

    return PackedTokenDataset(
        token_file=TRAIN_FILE,
        sequence_length=SEQUENCE_LENGTH,
    )


def test_dataset_builds() -> None:
    """Verify that the packed dataset can be created."""

    dataset = create_dataset()

    assert len(dataset) > 0
    assert dataset.total_tokens > 0


def test_sample_shapes() -> None:
    """Verify input and target sequence shapes."""

    dataset = create_dataset()

    input_ids, target_ids = dataset[0]

    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(target_ids, torch.Tensor)

    assert input_ids.shape == (SEQUENCE_LENGTH,)
    assert target_ids.shape == (SEQUENCE_LENGTH,)


def test_next_token_alignment() -> None:
    """Verify causal next-token alignment."""

    dataset = create_dataset()

    input_ids, target_ids = dataset[0]

    assert torch.equal(
        input_ids[1:],
        target_ids[:-1],
    )


def test_token_dtype() -> None:
    """Verify token IDs use the correct PyTorch dtype."""

    dataset = create_dataset()

    input_ids, target_ids = dataset[0]

    assert input_ids.dtype == torch.long
    assert target_ids.dtype == torch.long


def test_different_sequences() -> None:
    """Verify different dataset indices produce different sequences."""

    dataset = create_dataset()

    first_input, _ = dataset[0]
    second_input, _ = dataset[1]

    assert not torch.equal(
        first_input,
        second_input,
    )


def test_validation_dataset() -> None:
    """Verify that the validation token stream is usable."""

    dataset = PackedTokenDataset(
        token_file=VALIDATION_FILE,
        sequence_length=SEQUENCE_LENGTH,
    )

    assert len(dataset) > 0