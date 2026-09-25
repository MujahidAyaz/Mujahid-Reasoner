from __future__ import annotations

import pytest

from src.data.dataloader import DataLoaderConfig


def test_defaults() -> None:
    config = DataLoaderConfig()

    assert config.batch_size == 2
    assert config.shuffle is True
    assert config.num_workers == 0
    assert config.pin_memory is False
    assert config.drop_last is False
    assert config.persistent_workers is False
    assert config.prefetch_factor == 2


def test_custom_worker_configuration() -> None:
    config = DataLoaderConfig(
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    assert config.batch_size == 4
    assert config.num_workers == 2
    assert config.pin_memory is True
    assert config.drop_last is True
    assert config.persistent_workers is True
    assert config.prefetch_factor == 4


def test_dataloader_kwargs_without_workers() -> None:
    config = DataLoaderConfig(
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=4,
    )

    kwargs = config.dataloader_kwargs()

    assert kwargs == {
        "batch_size": 2,
        "num_workers": 0,
        "pin_memory": False,
        "drop_last": False,
    }

    assert "prefetch_factor" not in kwargs
    assert "persistent_workers" not in kwargs


def test_dataloader_kwargs_with_workers() -> None:
    config = DataLoaderConfig(
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    kwargs = config.dataloader_kwargs()

    assert kwargs == {
        "batch_size": 2,
        "num_workers": 2,
        "pin_memory": True,
        "drop_last": False,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": -1},
        {"num_workers": -1},
        {"prefetch_factor": 0},
        {"prefetch_factor": -1},
    ],
)
def test_invalid_values_are_rejected(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        DataLoaderConfig(**kwargs)


def test_persistent_workers_requires_workers() -> None:
    with pytest.raises(
        ValueError,
        match="persistent_workers requires num_workers > 0",
    ):
        DataLoaderConfig(
            num_workers=0,
            persistent_workers=True,
        )


def test_prefetch_factor_is_preserved_for_workers() -> None:
    config = DataLoaderConfig(
        num_workers=1,
        prefetch_factor=8,
    )

    kwargs = config.dataloader_kwargs()

    assert kwargs["prefetch_factor"] == 8


def test_configuration_is_reusable() -> None:
    config = DataLoaderConfig(
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=3,
    )

    first = config.dataloader_kwargs()
    second = config.dataloader_kwargs()

    assert first == second