from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.data.dataloader import (
    DataLoaderConfig,
    LanguageModelDataModule,
)


TOKEN_DIR = Path("data/processed/tokens")

TRAIN_FILE = TOKEN_DIR / "train.bin"
VALIDATION_FILE = TOKEN_DIR / "validation.bin"

SEQUENCE_LENGTH = 512


def create_data_module(
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    shuffle: bool = True,
) -> LanguageModelDataModule:
    config = DataLoaderConfig(
        batch_size=2,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    return LanguageModelDataModule(
        train_file=TRAIN_FILE,
        validation_file=VALIDATION_FILE,
        sequence_length=SEQUENCE_LENGTH,
        config=config,
        seed=42,
    )


def test_single_process_training_loader() -> None:
    data_module = create_data_module(
        num_workers=0,
    )

    loader = data_module.train_dataloader()

    assert loader.num_workers == 0
    assert loader.pin_memory is False

    input_ids, target_ids = next(iter(loader))

    assert input_ids.shape == (2, SEQUENCE_LENGTH)
    assert target_ids.shape == (2, SEQUENCE_LENGTH)


def test_validation_loader_is_sequential() -> None:
    data_module = create_data_module(
        shuffle=True,
        num_workers=0,
    )

    loader = data_module.validation_dataloader()

    assert loader.num_workers == 0
    assert loader.sampler is not None
    assert isinstance(
        loader.sampler,
        torch.utils.data.SequentialSampler,
    )


def test_training_loader_uses_resumable_sampler() -> None:
    data_module = create_data_module(
        shuffle=True,
        num_workers=0,
    )

    loader = data_module.train_dataloader()

    assert data_module.train_sampler is not None
    assert loader.sampler is data_module.train_sampler


def test_shuffle_false_does_not_create_resumable_sampler() -> None:
    data_module = create_data_module(
        shuffle=False,
        num_workers=0,
    )

    loader = data_module.train_dataloader()

    assert data_module.train_sampler is None
    assert isinstance(
        loader.sampler,
        torch.utils.data.SequentialSampler,
    )


def test_pin_memory_configuration_is_applied() -> None:
    data_module = create_data_module(
        num_workers=0,
        pin_memory=True,
    )

    loader = data_module.validation_dataloader()

    assert loader.pin_memory is True


def test_prefetch_configuration_requires_workers() -> None:
    data_module = create_data_module(
        num_workers=0,
        prefetch_factor=8,
    )

    loader = data_module.validation_dataloader()

    assert loader.num_workers == 0
    assert loader.pin_memory is False


def test_persistent_workers_requires_worker_processes() -> None:
    with pytest.raises(
        ValueError,
        match="persistent_workers requires num_workers > 0",
    ):
        create_data_module(
            num_workers=0,
            persistent_workers=True,
        )


def test_exact_resume_path_rejects_worker_processes() -> None:
    data_module = create_data_module(
        num_workers=1,
    )

    with pytest.raises(
        ValueError,
        match="Exact resumable training currently requires",
    ):
        data_module.train_dataloader()


def test_validation_supports_worker_configuration() -> None:
    data_module = create_data_module(
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    loader = data_module.validation_dataloader()

    try:
        assert loader.num_workers == 1
        assert loader.pin_memory is True
        assert loader.persistent_workers is True
        assert loader.prefetch_factor == 2

        input_ids, target_ids = next(iter(loader))

        assert input_ids.shape == (2, SEQUENCE_LENGTH)
        assert target_ids.shape == (2, SEQUENCE_LENGTH)

    finally:
        del loader