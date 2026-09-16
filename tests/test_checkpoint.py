from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.checkpoint import CheckpointManager


class CheckpointTestModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.linear = nn.Linear(4, 2)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(x)


def _create_training_objects() -> tuple[
    CheckpointTestModel,
    AdamW,
    LambdaLR,
]:
    model = CheckpointTestModel()

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
    )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0,
    )

    return model, optimizer, scheduler


def _train_one_step(
    model: nn.Module,
    optimizer: AdamW,
) -> None:
    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 2)

    outputs = model(inputs)

    loss = nn.functional.mse_loss(
        outputs,
        targets,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_creates_checkpoint_directory(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "checkpoints"

    manager = CheckpointManager(directory)

    assert directory.exists()


def test_latest_and_best_paths(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(tmp_path)

    assert manager.latest_path == (
        tmp_path / "latest.pt"
    )

    assert manager.best_path == (
        tmp_path / "best.pt"
    )


def test_saves_checkpoint(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = (
        _create_training_objects()
    )

    manager = CheckpointManager(tmp_path)

    path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=2,
        global_step=100,
        best_validation_loss=1.25,
        metrics={"train_loss": 1.5},
    )

    assert path.exists()
    assert path == tmp_path / "latest.pt"


def test_saves_best_checkpoint(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = (
        _create_training_objects()
    )

    manager = CheckpointManager(tmp_path)

    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=2,
        global_step=100,
        best_validation_loss=1.25,
        is_best=True,
    )

    assert manager.latest_path.exists()
    assert manager.best_path.exists()


def test_custom_filename(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = (
        _create_training_objects()
    )

    manager = CheckpointManager(tmp_path)

    path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=10,
        best_validation_loss=2.0,
        filename="epoch_001.pt",
    )

    assert path == tmp_path / "epoch_001.pt"
    assert path.exists()


def test_load_restores_model_state(
    tmp_path: Path,
) -> None:
    torch.manual_seed(42)

    model, optimizer, scheduler = (
        _create_training_objects()
    )

    _train_one_step(
        model,
        optimizer,
    )

    original_state = {
        key: value.clone()
        for key, value in model.state_dict().items()
    }

    manager = CheckpointManager(tmp_path)

    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        global_step=150,
        best_validation_loss=0.75,
    )

    restored_model, restored_optimizer, restored_scheduler = (
        _create_training_objects()
    )

    manager.load(
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )

    for key, value in original_state.items():
        assert torch.equal(
            value,
            restored_model.state_dict()[key],
        )


def test_load_returns_training_metadata(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = (
        _create_training_objects()
    )

    manager = CheckpointManager(tmp_path)

    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=5,
        global_step=250,
        best_validation_loss=0.42,
        metrics={
            "train_loss": 0.50,
            "val_loss": 0.42,
        },
    )

    restored_model, restored_optimizer, restored_scheduler = (
        _create_training_objects()
    )

    metadata = manager.load(
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )

    assert metadata["epoch"] == 5
    assert metadata["global_step"] == 250
    assert metadata["best_validation_loss"] == pytest.approx(
        0.42
    )

    assert metadata["metrics"] == {
        "train_loss": 0.50,
        "val_loss": 0.42,
    }


def test_optimizer_state_is_restored(
    tmp_path: Path,
) -> None:
    torch.manual_seed(42)

    model, optimizer, scheduler = (
        _create_training_objects()
    )

    _train_one_step(
        model,
        optimizer,
    )

    manager = CheckpointManager(tmp_path)

    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=1,
        best_validation_loss=1.0,
    )

    restored_model, restored_optimizer, restored_scheduler = (
        _create_training_objects()
    )

    manager.load(
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )

    assert len(restored_optimizer.state) == (
        len(optimizer.state)
    )


def test_scheduler_state_is_restored(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = (
        _create_training_objects()
    )

    optimizer.step()
    scheduler.step()

    original_state = scheduler.state_dict()

    manager = CheckpointManager(tmp_path)

    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=1,
        best_validation_loss=1.0,
    )

    restored_model, restored_optimizer, restored_scheduler = (
        _create_training_objects()
    )

    manager.load(
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )

    assert (
        restored_scheduler.state_dict()
        == original_state
    )


def test_rng_state_is_restored(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = (
        _create_training_objects()
    )

    manager = CheckpointManager(tmp_path)

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=1,
        best_validation_loss=1.0,
    )

    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(1)

    manager.load(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    assert random.random() == pytest.approx(
        expected_python
    )

    assert np.random.random() == pytest.approx(
        expected_numpy
    )

    assert torch.equal(
        torch.rand(1),
        expected_torch,
    )


def test_missing_checkpoint_raises_error(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = (
        _create_training_objects()
    )

    manager = CheckpointManager(tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match="Checkpoint not found",
    ):
        manager.load(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )


def test_invalid_epoch_raises_error(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = (
        _create_training_objects()
    )

    manager = CheckpointManager(tmp_path)

    with pytest.raises(
        ValueError,
        match="epoch",
    ):
        manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=-1,
            global_step=0,
            best_validation_loss=1.0,
        )


def test_invalid_global_step_raises_error(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = (
        _create_training_objects()
    )

    manager = CheckpointManager(tmp_path)

    with pytest.raises(
        ValueError,
        match="global_step",
    ):
        manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=0,
            global_step=-1,
            best_validation_loss=1.0,
        )


def test_invalid_validation_loss_raises_error(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = (
        _create_training_objects()
    )

    manager = CheckpointManager(tmp_path)

    with pytest.raises(
        ValueError,
        match="best_validation_loss",
    ):
        manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=0,
            global_step=0,
            best_validation_loss=-1.0,
        )


def test_checkpoint_without_scheduler(
    tmp_path: Path,
) -> None:
    model, optimizer, _ = (
        _create_training_objects()
    )

    manager = CheckpointManager(tmp_path)

    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=1,
        global_step=10,
        best_validation_loss=1.0,
    )

    restored_model, restored_optimizer, _ = (
        _create_training_objects()
    )

    metadata = manager.load(
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=None,
    )

    assert metadata["epoch"] == 1