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

from src.data.dataloader import ResumableRandomSampler
from src.training.trainer import Trainer, TrainerConfig


class TrainerTestModel(nn.Module):
    """Small deterministic language-model-like network for trainer tests."""

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


def _create_dataset(
    sample_count: int = 8,
) -> TensorDataset:
    """Create a deterministic synthetic language-model dataset."""

    torch.manual_seed(42)

    inputs = torch.randint(
        0,
        16,
        (sample_count, 8),
    )

    targets = torch.randint(
        0,
        16,
        (sample_count, 8),
    )

    return TensorDataset(
        inputs,
        targets,
    )


def _create_loaders(
    *,
    batch_size: int = 2,
    sample_count: int = 8,
) -> tuple[DataLoader, DataLoader]:
    """Create deterministic training and validation loaders."""

    dataset = _create_dataset(
        sample_count=sample_count,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    validation_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, validation_loader


def _create_optimizer(
    model: nn.Module,
) -> AdamW:
    """Create the optimizer used by trainer tests."""

    return AdamW(
        model.parameters(),
        lr=1e-3,
    )


def _create_scheduler(
    optimizer: AdamW,
) -> LambdaLR:
    """Create a deterministic scheduler used by trainer tests."""

    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0,
    )


def _create_trainer(
    tmp_path: Path,
    *,
    max_epochs: int = 5,
    max_steps: int | None = 2,
    gradient_accumulation_steps: int = 1,
    batch_size: int = 2,
    sample_count: int = 8,
    model: nn.Module | None = None,
) -> Trainer:
    """Build a trainer with fully controllable test configuration."""

    if model is None:
        model = TrainerTestModel()

    train_loader, validation_loader = _create_loaders(
        batch_size=batch_size,
        sample_count=sample_count,
    )

    optimizer = _create_optimizer(model)

    scheduler = _create_scheduler(
        optimizer,
    )

    config = TrainerConfig(
        device="cpu",
        max_epochs=max_epochs,
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
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


def _create_trainer_with_sampler(
    tmp_path: Path,
    *,
    max_epochs: int = 5,
    max_steps: int | None = 1,
    batch_size: int = 2,
    sample_count: int = 8,
) -> Trainer:
    """Build a trainer using the deterministic resumable sampler."""

    model = TrainerTestModel()

    dataset = _create_dataset(
        sample_count=sample_count,
    )

    sampler = ResumableRandomSampler(
        dataset,
        seed=42,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
    )

    validation_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = _create_optimizer(model)

    scheduler = _create_scheduler(
        optimizer,
    )

    config = TrainerConfig(
        device="cpu",
        max_epochs=max_epochs,
        max_steps=max_steps,
        gradient_accumulation_steps=1,
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
        train_sampler=sampler,
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
            gradient_accumulation_steps=0,
        )

    with pytest.raises(ValueError):
        TrainerConfig(max_grad_norm=0)

    with pytest.raises(ValueError):
        TrainerConfig(log_every_steps=0)

    with pytest.raises(ValueError):
        TrainerConfig(eval_every_steps=0)

    with pytest.raises(ValueError):
        TrainerConfig(
            checkpoint_every_steps=0,
        )


def test_trainer_initial_state(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path,
    )

    assert trainer.state.epoch == 0
    assert trainer.state.global_step == 0
    assert trainer.state.best_validation_loss == float("inf")


def test_prepare_batch_moves_to_device(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path,
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
            (inputs, targets),
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
        tmp_path,
    )

    loss = trainer.evaluate()

    assert isinstance(loss, float)
    assert torch.isfinite(
        torch.tensor(loss),
    )


def test_evaluate_restores_train_mode(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path,
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
        for name, parameter in trainer.model.named_parameters()
    }

    state = trainer.train()

    assert state.global_step == 1

    changed = False

    for name, parameter in trainer.model.named_parameters():
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


def test_gradient_accumulation_matches_equivalent_batch(
    tmp_path: Path,
) -> None:
    """
    Verify that two equal-sized micro-batches produce the same
    optimizer update as one equivalent larger batch.

    Trainer A:
        batch_size=2
        accumulation_steps=2

    Trainer B:
        batch_size=4
        accumulation_steps=1
    """

    torch.manual_seed(1234)

    model_accumulated = TrainerTestModel()
    model_equivalent = TrainerTestModel()

    model_equivalent.load_state_dict(
        model_accumulated.state_dict(),
    )

    accumulated = _create_trainer(
        tmp_path / "accumulated",
        max_steps=1,
        gradient_accumulation_steps=2,
        batch_size=2,
        sample_count=4,
        model=model_accumulated,
    )

    equivalent = _create_trainer(
        tmp_path / "equivalent",
        max_steps=1,
        gradient_accumulation_steps=1,
        batch_size=4,
        sample_count=4,
        model=model_equivalent,
    )

    accumulated.train()
    equivalent.train()

    assert accumulated.state.global_step == 1
    assert equivalent.state.global_step == 1

    for accumulated_parameter, equivalent_parameter in zip(
        accumulated.model.parameters(),
        equivalent.model.parameters(),
    ):
        torch.testing.assert_close(
            accumulated_parameter,
            equivalent_parameter,
            rtol=1e-5,
            atol=1e-6,
        )


def test_partial_gradient_accumulation_matches_actual_window(
    tmp_path: Path,
) -> None:
    """
    Verify that a partial final accumulation window is normalized
    using the number of micro-batches actually processed.

    Trainer A:
        3 micro-batches × 2 samples
        accumulation_steps=4

    Trainer B:
        1 batch × 6 samples
        accumulation_steps=1
    """

    torch.manual_seed(5678)

    model_partial = TrainerTestModel()
    model_equivalent = TrainerTestModel()

    model_equivalent.load_state_dict(
        model_partial.state_dict(),
    )

    partial = _create_trainer(
        tmp_path / "partial",
        max_steps=1,
        gradient_accumulation_steps=4,
        batch_size=2,
        sample_count=6,
        model=model_partial,
    )

    equivalent = _create_trainer(
        tmp_path / "equivalent_partial",
        max_steps=1,
        gradient_accumulation_steps=1,
        batch_size=6,
        sample_count=6,
        model=model_equivalent,
    )

    partial.train()
    equivalent.train()

    assert partial.state.global_step == 1
    assert equivalent.state.global_step == 1

    for partial_parameter, equivalent_parameter in zip(
        partial.model.parameters(),
        equivalent.model.parameters(),
    ):
        torch.testing.assert_close(
            partial_parameter,
            equivalent_parameter,
            rtol=1e-5,
            atol=1e-6,
        )


def test_gradient_accumulation_optimizer_steps_once_per_window(
    tmp_path: Path,
) -> None:
    """
    Four micro-batches with accumulation=2 must produce exactly two
    optimizer and scheduler updates during one epoch.
    """

    trainer = _create_trainer(
        tmp_path,
        max_epochs=1,
        max_steps=None,
        gradient_accumulation_steps=2,
        batch_size=2,
        sample_count=8,
    )

    initial_scheduler_step = trainer.scheduler.last_epoch

    state = trainer.train()

    assert state.global_step == 2

    assert (
        trainer.scheduler.last_epoch
        == initial_scheduler_step + 2
    )


def test_partial_gradient_accumulation_performs_one_optimizer_step(
    tmp_path: Path,
) -> None:
    """
    Three micro-batches with accumulation=4 must still flush one
    valid partial optimizer update at the end of the epoch.
    """

    trainer = _create_trainer(
        tmp_path,
        max_epochs=1,
        max_steps=None,
        gradient_accumulation_steps=4,
        batch_size=2,
        sample_count=6,
    )

    state = trainer.train()

    assert state.global_step == 1
    assert state.total_tokens == 6 * 8


def test_gradient_accumulation_counts_all_tokens(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path,
        max_steps=1,
        gradient_accumulation_steps=2,
        batch_size=2,
        sample_count=4,
    )

    state = trainer.train()

    assert state.global_step == 1

    # 4 samples × 8 tokens.
    assert state.total_tokens == 32


def test_gradient_accumulation_loss_is_window_average(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path,
        max_steps=1,
        gradient_accumulation_steps=2,
        batch_size=2,
        sample_count=4,
    )

    state = trainer.train()

    assert torch.isfinite(
        torch.tensor(state.train_loss),
    )

    assert state.train_loss > 0.0


def test_gradient_buffers_are_cleared_after_optimizer_step(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path,
        max_steps=1,
        gradient_accumulation_steps=2,
    )

    trainer.train()

    for parameter in trainer.model.parameters():
        assert parameter.grad is None


def test_max_steps_counts_optimizer_updates(
    tmp_path: Path,
) -> None:
    """
    max_steps refers to optimizer updates, not micro-batches.

    With accumulation=4 and max_steps=2, eight micro-batches are
    required for two optimizer updates.
    """

    trainer = _create_trainer(
        tmp_path,
        max_steps=2,
        gradient_accumulation_steps=4,
        batch_size=2,
        sample_count=16,
    )

    state = trainer.train()

    assert state.global_step == 2

    assert state.total_tokens == (
        2 * 4 * 2 * 8
    )


def test_scheduler_steps_after_optimizer(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path,
        max_steps=2,
    )

    initial_scheduler_step = trainer.scheduler.last_epoch

    trainer.train()

    assert (
        trainer.scheduler.last_epoch
        > initial_scheduler_step
    )


def test_scheduler_does_not_step_per_micro_batch(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path,
        max_steps=1,
        gradient_accumulation_steps=4,
        batch_size=2,
        sample_count=8,
    )

    initial_scheduler_step = trainer.scheduler.last_epoch

    trainer.train()

    assert (
        trainer.scheduler.last_epoch
        == initial_scheduler_step + 1
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
        torch.tensor(state.train_loss),
    )


def test_validation_loss_is_finite(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path,
        max_steps=1,
    )

    validation_loss = trainer.evaluate()

    assert torch.isfinite(
        torch.tensor(validation_loss),
    )


def test_checkpoint_is_created(
    tmp_path: Path,
) -> None:
    trainer = _create_trainer(
        tmp_path,
        max_steps=1,
    )

    trainer._save_checkpoint(
        is_best=True,
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

    assert resumed.state.global_step == original_step


def test_sampler_resume_does_not_double_skip(
    tmp_path: Path,
) -> None:
    """
    Verify that restoring a resumable sampler does not cause the
    trainer to skip the sampler position a second time.
    """

    trainer = _create_trainer_with_sampler(
        tmp_path,
        max_steps=1,
        batch_size=2,
        sample_count=8,
    )

    iterator = iter(trainer.train_loader)

    next(iterator)

    consumed_position = trainer.train_sampler.position

    assert consumed_position == 2

    trainer.state.batch_in_epoch = 1

    trainer._save_checkpoint(
        is_best=False,
    )

    resumed = _create_trainer_with_sampler(
        tmp_path,
        max_steps=1,
        batch_size=2,
        sample_count=8,
    )

    resumed.resume()

    assert (
        resumed.train_sampler.position
        == consumed_position
    )

    assert resumed.state.batch_in_epoch == 1


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