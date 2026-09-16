from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import (
    DataLoaderConfig,
    LanguageModelDataModule,
    set_seed,
)


TOKEN_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tokens"
)

TRAIN_FILE = TOKEN_DIR / "train.bin"
VALIDATION_FILE = TOKEN_DIR / "validation.bin"

SEQUENCE_LENGTH = 512
BATCH_SIZE = 2


def create_data_module() -> LanguageModelDataModule:
    """Create the data module for testing."""

    config = DataLoaderConfig(
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    return LanguageModelDataModule(
        train_file=TRAIN_FILE,
        validation_file=VALIDATION_FILE,
        sequence_length=SEQUENCE_LENGTH,
        config=config,
    )


def test_train_dataloader_builds() -> None:
    """Verify the training DataLoader can be created."""

    data_module = create_data_module()

    loader = data_module.train_dataloader()

    assert len(loader) > 0


def test_validation_dataloader_builds() -> None:
    """Verify the validation DataLoader can be created."""

    data_module = create_data_module()

    loader = data_module.validation_dataloader()

    assert len(loader) > 0


def test_batch_shapes() -> None:
    """Verify training batches have the expected shape."""

    data_module = create_data_module()

    loader = data_module.train_dataloader()

    input_ids, target_ids = next(iter(loader))

    assert input_ids.shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
    )

    assert target_ids.shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
    )


def test_batch_dtype() -> None:
    """Verify batches use PyTorch long tensors."""

    data_module = create_data_module()

    loader = data_module.train_dataloader()

    input_ids, target_ids = next(iter(loader))

    assert input_ids.dtype == torch.long
    assert target_ids.dtype == torch.long


def test_batch_alignment() -> None:
    """Verify causal next-token alignment inside a batch."""

    data_module = create_data_module()

    loader = data_module.train_dataloader()

    input_ids, target_ids = next(iter(loader))

    assert torch.equal(
        input_ids[:, 1:],
        target_ids[:, :-1],
    )


def test_reproducibility() -> None:
    """Verify deterministic random-number seeding."""

    set_seed(42)
    first = torch.rand(10)

    set_seed(42)
    second = torch.rand(10)

    assert torch.equal(first, second)