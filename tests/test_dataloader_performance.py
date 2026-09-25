from __future__ import annotations

import time
from pathlib import Path

import torch

from src.data.dataloader import (
    DataLoaderConfig,
    LanguageModelDataModule,
)


TOKEN_DIR = Path("data/processed/tokens")

TRAIN_FILE = TOKEN_DIR / "train.bin"
VALIDATION_FILE = TOKEN_DIR / "validation.bin"

SEQUENCE_LENGTH = 512
BATCH_SIZE = 2


def create_data_module(
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> LanguageModelDataModule:
    config = DataLoaderConfig(
        batch_size=BATCH_SIZE,
        shuffle=True,
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


def benchmark_loader(
    loader,
    *,
    batches: int = 20,
) -> tuple[float, int]:
    iterator = iter(loader)

    start = time.perf_counter()

    samples = 0

    for _ in range(batches):
        input_ids, target_ids = next(iterator)

        assert input_ids.shape == (
            BATCH_SIZE,
            SEQUENCE_LENGTH,
        )

        assert target_ids.shape == (
            BATCH_SIZE,
            SEQUENCE_LENGTH,
        )

        samples += input_ids.size(0)

    elapsed = time.perf_counter() - start

    return elapsed, samples


def test_single_process_loader_baseline() -> None:
    data_module = create_data_module(
        num_workers=0,
        pin_memory=False,
    )

    loader = data_module.train_dataloader()

    elapsed, samples = benchmark_loader(
        loader,
        batches=20,
    )

    assert elapsed > 0.0
    assert samples == BATCH_SIZE * 20


def test_single_process_pinned_memory_loader() -> None:
    data_module = create_data_module(
        num_workers=0,
        pin_memory=True,
    )

    loader = data_module.train_dataloader()

    elapsed, samples = benchmark_loader(
        loader,
        batches=20,
    )

    assert elapsed > 0.0
    assert samples == BATCH_SIZE * 20


def test_validation_worker_loader() -> None:
    data_module = create_data_module(
        num_workers=1,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
    )

    loader = data_module.validation_dataloader()

    try:
        elapsed, samples = benchmark_loader(
            loader,
            batches=20,
        )

        assert elapsed > 0.0
        assert samples == BATCH_SIZE * 20

    finally:
        del loader


def test_data_loader_output_is_contiguous() -> None:
    data_module = create_data_module(
        num_workers=0,
    )

    loader = data_module.train_dataloader()

    input_ids, target_ids = next(iter(loader))

    assert input_ids.is_contiguous()
    assert target_ids.is_contiguous()


def test_data_loader_output_dtype_is_optimal() -> None:
    data_module = create_data_module(
        num_workers=0,
    )

    loader = data_module.train_dataloader()

    input_ids, target_ids = next(iter(loader))

    assert input_ids.dtype == torch.long
    assert target_ids.dtype == torch.long