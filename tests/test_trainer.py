from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import (
    Trainer,
    TrainerConfig,
)


class TrainerTestModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 16,
        hidden_size: int = 16,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            hidden_size,
        )

        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.embedding(input_ids)

        return self.lm_head(hidden)


def _create_loaders() -> tuple[
    DataLoader,
    DataLoader,
]:
    torch.manual_seed(42)

    inputs = torch.randint(
        0,
        16,
        (8, 8),
    )

    targets = torch.randint(
        0,
        16,
        (8, 8),
    )

    dataset = TensorDataset(
        inputs,
        targets,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    validation_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    return train_loader, validation_loader


def _create_trainer(
    tmp_path: Path,
    *,
    max_steps: int = 2,
    gradient_accumulation_steps: int = 1,
) -> Trainer:
    model = TrainerTestModel()

    train_loader, validation_loader = (
        _create_loaders()
    )

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
    )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0,
    )

    config = TrainerConfig(
        device="cpu",
        max_epochs=5,
        max_steps=max_steps,
        gradient_accumulation_steps=(
            gradient_accumulation_steps
        ),
        log_every_steps=1000,
        eval_every_steps=1000,
        checkpoint_every_steps=1000,
        output_dir=str(tmp_path / "runs"),
    )

    return Trainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )


def test_config_defaults() -> None:
    config = TrainerConfig()

    assert config.device == "cpu"
    assert config.max_epochs == 1
    assert config.gradient_accumulation_steps == 1
    assert config.max_grad_norm == 1.0


def test_invalid_config_values() -> None:
    with pytest.raises(ValueError):
        TrainerConfig(max_epochs=0)

    with pytest.raises(ValueError):
        TrainerConfig(max_steps=0)

    with pytest.raises(ValueError):
        TrainerConfig(
            gradient_accumulation_steps=0
        )

    with pytest.raises(ValueError):
        TrainerConfig(max_grad_norm=0)

    with pytest.raises(ValueError):
        TrainerConfig(log_every_steps=0)

    with pytest.raises(ValueError):
        TrainerConfig(eval_every_steps=0)

    with pytest.raises(ValueError):
        TrainerConfig(
            checkpoint_every_steps=0
        )


def test_trainer_initial_state(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path
    )

    assert trainer.state.epoch == 0
    assert trainer.state.global_step == 0
    assert trainer.state.best_validation_loss == (
        float("inf")
    )


def test_prepare_batch_moves_to_device(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path
    )

    inputs = torch.ones(
        2,
        8,
        dtype=torch.long,
    )

    targets = torch.ones(
        2,
        8,
        dtype=torch.long,
    )

    prepared_inputs, prepared_targets = (
        trainer._prepare_batch(
            (inputs, targets)
        )
    )

    assert prepared_inputs.device.type == "cpu"
    assert prepared_targets.device.type == "cpu"
    assert prepared_inputs.dtype == torch.long
    assert prepared_targets.dtype == torch.long


def test_evaluate_returns_finite_loss(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path
    )

    loss = trainer.evaluate()

    assert isinstance(loss, float)
    assert torch.isfinite(
        torch.tensor(loss)
    )


def test_evaluate_restores_train_mode(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path
    )

    trainer.model.train()

    trainer.evaluate()

    assert trainer.model.training


def test_training_updates_parameters(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path,
        max_steps=1,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in (
            trainer.model.named_parameters()
        )
    }

    state = trainer.train()

    assert state.global_step == 1

    changed = False

    for name, parameter in (
        trainer.model.named_parameters()
    ):
        if not torch.equal(
            before[name],
            parameter,
        ):
            changed = True
            break

    assert changed


def test_max_steps_stops_training(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path,
        max_steps=2,
    )

    state = trainer.train()

    assert state.global_step == 2


def test_gradient_accumulation(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path,
        max_steps=2,
        gradient_accumulation_steps=2,
    )

    state = trainer.train()

    assert state.global_step == 2


def test_scheduler_steps_after_optimizer(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path,
        max_steps=2,
    )

    initial_scheduler_step = (
        trainer.scheduler.last_epoch
    )

    trainer.train()

    assert (
        trainer.scheduler.last_epoch
        > initial_scheduler_step
    )


def test_training_loss_is_finite(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path,
        max_steps=2,
    )

    state = trainer.train()

    assert torch.isfinite(
        torch.tensor(state.train_loss)
    )


def test_validation_loss_is_finite(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path,
        max_steps=1,
    )

    trainer.config

    validation_loss = trainer.evaluate()

    assert torch.isfinite(
        torch.tensor(validation_loss)
    )


def test_checkpoint_is_created(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path,
        max_steps=1,
    )

    trainer.config

    trainer._save_checkpoint(
        is_best=True
    )

    assert trainer.checkpoint_manager.latest_path.exists()
    assert trainer.checkpoint_manager.best_path.exists()


def test_resume_restores_state(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path,
        max_steps=1,
    )

    trainer.train()

    trainer._save_checkpoint(
        is_best=True,
    )

    original_step = trainer.state.global_step

    resumed = _create_trainer(
        tmp_path,
        max_steps=2,
    )

    resumed.resume()

    assert resumed.state.global_step == (
        original_step
    )


def test_should_stop_without_max_steps(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path,
        max_steps=1,
    )

    trainer.state.global_step = 0

    assert not trainer._should_stop()

    trainer.state.global_step = 1

    assert trainer._should_stop()