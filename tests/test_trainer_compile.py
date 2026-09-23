from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import Trainer, TrainerConfig


class CompileTestModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 16,
        hidden_size: int = 8,
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
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embedding(input_ids)
        return self.lm_head(hidden_states)


class ForwardWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        *args,
        **kwargs,
    ):
        return self.model(
            *args,
            **kwargs,
        )


class TensorLoss:
    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )


def create_dataloaders() -> tuple[
    DataLoader,
    DataLoader,
]:
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
        ],
        dtype=torch.long,
    )

    labels = torch.tensor(
        [
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [5, 6, 7, 8],
        ],
        dtype=torch.long,
    )

    dataset = TensorDataset(
        input_ids,
        labels,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    return loader, loader


def create_trainer(
    *,
    compile: bool = False,
) -> Trainer:
    model = CompileTestModel()

    train_loader, validation_loader = (
        create_dataloaders()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    config = TrainerConfig(
        device="cpu",
        precision="fp32",
        compile=compile,
        max_epochs=1,
        max_steps=1,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        log_every_steps=1,
        eval_every_steps=1,
        checkpoint_every_steps=1,
        max_eval_batches=1,
        output_dir="experiments/test-compile",
        seed=42,
    )

    return Trainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        scheduler=None,
        loss_fn=TensorLoss(),
        config=config,
    )


def test_compile_defaults_to_disabled() -> None:
    config = TrainerConfig()

    assert config.compile is False


def test_compile_disabled_uses_canonical_model() -> None:
    trainer = create_trainer(
        compile=False,
    )

    assert trainer.model is trainer._forward_model


def test_compile_enabled_calls_torch_compile() -> None:
    trainer_model = CompileTestModel()

    train_loader, validation_loader = (
        create_dataloaders()
    )

    optimizer = torch.optim.AdamW(
        trainer_model.parameters(),
        lr=1e-3,
    )

    config = TrainerConfig(
        device="cpu",
        precision="fp32",
        compile=True,
        max_epochs=1,
        max_steps=1,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        log_every_steps=1,
        eval_every_steps=1,
        checkpoint_every_steps=1,
        max_eval_batches=1,
        output_dir="experiments/test-compile",
        seed=42,
    )

    compiled_model = ForwardWrapper(
        trainer_model,
    )

    with patch(
        "src.training.trainer.torch.compile",
        return_value=compiled_model,
    ) as compile_mock:

        trainer = Trainer(
            model=trainer_model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            optimizer=optimizer,
            scheduler=None,
            loss_fn=TensorLoss(),
            config=config,
        )

    compile_mock.assert_called_once_with(
        trainer_model,
    )

    assert trainer.model is trainer_model
    assert trainer._forward_model is compiled_model
    assert trainer._forward_model is not trainer.model


def test_compile_preserves_canonical_model() -> None:
    trainer_model = CompileTestModel()

    original_state = {
        name: parameter.detach().clone()
        for name, parameter in trainer_model.named_parameters()
    }

    train_loader, validation_loader = (
        create_dataloaders()
    )

    optimizer = torch.optim.AdamW(
        trainer_model.parameters(),
        lr=1e-3,
    )

    config = TrainerConfig(
        device="cpu",
        precision="fp32",
        compile=True,
        max_epochs=1,
        max_steps=1,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        log_every_steps=1,
        eval_every_steps=1,
        checkpoint_every_steps=1,
        max_eval_batches=1,
        output_dir="experiments/test-compile",
        seed=42,
    )

    compiled_model = ForwardWrapper(
        trainer_model,
    )

    with patch(
        "src.training.trainer.torch.compile",
        return_value=compiled_model,
    ):
        trainer = Trainer(
            model=trainer_model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            optimizer=optimizer,
            scheduler=None,
            loss_fn=TensorLoss(),
            config=config,
        )

    for name, parameter in trainer.model.named_parameters():
        assert torch.equal(
            parameter.detach(),
            original_state[name],
        )


def test_compile_forward_model_is_used_for_evaluation() -> None:
    trainer_model = CompileTestModel()

    train_loader, validation_loader = (
        create_dataloaders()
    )

    optimizer = torch.optim.AdamW(
        trainer_model.parameters(),
        lr=1e-3,
    )

    config = TrainerConfig(
        device="cpu",
        precision="fp32",
        compile=True,
        max_epochs=1,
        max_steps=1,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        log_every_steps=1,
        eval_every_steps=1,
        checkpoint_every_steps=1,
        max_eval_batches=1,
        output_dir="experiments/test-compile",
        seed=42,
    )

    compiled_model = ForwardWrapper(
        trainer_model,
    )

    with patch(
        "src.training.trainer.torch.compile",
        return_value=compiled_model,
    ):
        trainer = Trainer(
            model=trainer_model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            optimizer=optimizer,
            scheduler=None,
            loss_fn=TensorLoss(),
            config=config,
        )

    with patch.object(
        compiled_model,
        "forward",
        wraps=compiled_model.forward,
    ) as forward_mock:

        trainer.evaluate()

    assert forward_mock.call_count > 0


def test_compile_failure_is_reported() -> None:
    trainer_model = CompileTestModel()

    train_loader, validation_loader = (
        create_dataloaders()
    )

    optimizer = torch.optim.AdamW(
        trainer_model.parameters(),
        lr=1e-3,
    )

    config = TrainerConfig(
        device="cpu",
        precision="fp32",
        compile=True,
        max_epochs=1,
        max_steps=1,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        log_every_steps=1,
        eval_every_steps=1,
        checkpoint_every_steps=1,
        max_eval_batches=1,
        output_dir="experiments/test-compile",
        seed=42,
    )

    with patch(
        "src.training.trainer.torch.compile",
        side_effect=RuntimeError(
            "synthetic compile failure"
        ),
    ):

        with pytest.raises(
            RuntimeError,
            match="Failed to initialize torch\\.compile",
        ):

            Trainer(
                model=trainer_model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                optimizer=optimizer,
                scheduler=None,
                loss_fn=TensorLoss(),
                config=config,
            )


def test_compile_rejects_non_module_result() -> None:
    trainer_model = CompileTestModel()

    train_loader, validation_loader = (
        create_dataloaders()
    )

    optimizer = torch.optim.AdamW(
        trainer_model.parameters(),
        lr=1e-3,
    )

    config = TrainerConfig(
        device="cpu",
        precision="fp32",
        compile=True,
        max_epochs=1,
        max_steps=1,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        log_every_steps=1,
        eval_every_steps=1,
        checkpoint_every_steps=1,
        max_eval_batches=1,
        output_dir="experiments/test-compile",
        seed=42,
    )

    with patch(
        "src.training.trainer.torch.compile",
        return_value=object(),
    ):

        with pytest.raises(
            TypeError,
            match="torch\\.compile\\(\\) returned",
        ):

            Trainer(
                model=trainer_model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                optimizer=optimizer,
                scheduler=None,
                loss_fn=TensorLoss(),
                config=config,
            )