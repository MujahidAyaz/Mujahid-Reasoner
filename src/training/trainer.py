from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.training.checkpoint import CheckpointManager
from src.training.gradient_clipping import clip_gradients
from src.training.loss import CausalLanguageModelLoss
from src.training.metrics import TrainingMetrics, TrainingTimer


@dataclass(frozen=True)
class TrainerConfig:
    device: str = "cpu"
    max_epochs: int = 1
    max_steps: int | None = None
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    log_every_steps: int = 10
    eval_every_steps: int = 100
    checkpoint_every_steps: int = 100
    max_eval_batches: int | None = None
    output_dir: str = "experiments/runs"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.max_epochs <= 0:
            raise ValueError(
                "max_epochs must be greater than 0."
            )

        if (
            self.max_steps is not None
            and self.max_steps <= 0
        ):
            raise ValueError(
                "max_steps must be greater than 0."
            )

        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be greater than 0."
            )

        if self.max_grad_norm <= 0:
            raise ValueError(
                "max_grad_norm must be greater than 0."
            )

        if self.log_every_steps <= 0:
            raise ValueError(
                "log_every_steps must be greater than 0."
            )

        if self.eval_every_steps <= 0:
            raise ValueError(
                "eval_every_steps must be greater than 0."
            )

        if self.checkpoint_every_steps <= 0:
            raise ValueError(
                "checkpoint_every_steps must be greater than 0."
            )

        if (
            self.max_eval_batches is not None
            and self.max_eval_batches <= 0
        ):
            raise ValueError(
                "max_eval_batches must be greater than 0."
            )

        if self.seed < 0:
            raise ValueError(
                "seed must be non-negative."
            )


@dataclass
class TrainingState:
    epoch: int = 0
    global_step: int = 0

    best_validation_loss: float = math.inf

    train_loss: float = 0.0
    validation_loss: float = math.inf

    total_tokens: int = 0
    elapsed_seconds: float = 0.0

    last_gradient_norm: float = 0.0
    last_learning_rate: float = 0.0


class Trainer:
    """Production-oriented training loop for causal language models."""

    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader: Any,
        validation_loader: Any,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        config: TrainerConfig,
        loss_fn: nn.Module | None = None,
        checkpoint_manager: CheckpointManager | None = None,
    ) -> None:
        self.config = config

        self.device = torch.device(
            config.device
        )

        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.loss_fn = (
            loss_fn
            if loss_fn is not None
            else CausalLanguageModelLoss()
        )

        self.checkpoint_manager = (
            checkpoint_manager
            if checkpoint_manager is not None
            else CheckpointManager(
                Path(config.output_dir)
                / "checkpoints"
            )
        )

        self.state = TrainingState()

        self.metrics = TrainingMetrics()

        self.timer = TrainingTimer()

        self._micro_steps_since_update = 0
        self._accumulated_loss = 0.0
        self._accumulated_tokens = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> TrainingState:
        """Run the training loop."""

        self.timer.start()

        for epoch in range(
            self.state.epoch,
            self.config.max_epochs,
        ):
            self.state.epoch = epoch

            self._train_epoch()

            if self._should_stop():
                break

            self.state.epoch = epoch + 1

        self.state.elapsed_seconds = (
            self.timer.elapsed()
        )

        # Always perform a final validation pass.
        final_validation_loss = self.evaluate()

        self.state.validation_loss = (
            final_validation_loss
        )

        if (
            final_validation_loss
            < self.state.best_validation_loss
        ):
            self.state.best_validation_loss = (
                final_validation_loss
            )
            is_best = True
        else:
            is_best = False

        self.state.elapsed_seconds = (
            self.timer.elapsed()
        )

        self._sync_metrics()

        # Always save a final checkpoint.
        self._save_checkpoint(
            is_best=is_best
        )

        return self.state

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_epoch(self) -> None:
        self.model.train()

        self.optimizer.zero_grad(
            set_to_none=True
        )

        for batch in self.train_loader:
            inputs, targets = self._prepare_batch(
                batch
            )

            model_output = self.model(inputs)
            logits = (
                model_output[0]
                if isinstance(model_output, tuple)
                else model_output
            )

            loss = self.loss_fn(
                logits,
                targets,
            )

            scaled_loss = (
                loss
                / self.config.gradient_accumulation_steps
            )

            scaled_loss.backward()

            self._accumulated_loss += (
                float(loss.detach().item())
            )

            self._accumulated_tokens += (
                targets.numel()
            )

            self._micro_steps_since_update += 1

            if (
                self._micro_steps_since_update
                >= self.config.gradient_accumulation_steps
            ):
                self._optimizer_step()

                if self._should_stop():
                    break

    def _optimizer_step(self) -> None:
        gradient_norm = clip_gradients(
            self.model,
            max_norm=self.config.max_grad_norm,
        )

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.optimizer.zero_grad(
            set_to_none=True
        )

        self.state.global_step += 1

        self.state.train_loss = (
            self._accumulated_loss
            / self._micro_steps_since_update
        )

        self.state.total_tokens += (
            self._accumulated_tokens
        )

        self.state.last_gradient_norm = (
            gradient_norm
        )

        self.state.last_learning_rate = (
            self.optimizer.param_groups[0]["lr"]
        )

        self._accumulated_loss = 0.0
        self._accumulated_tokens = 0
        self._micro_steps_since_update = 0

        self.state.elapsed_seconds = (
            self.timer.elapsed()
        )

        self._sync_metrics()

        if (
            self.state.global_step
            % self.config.log_every_steps
            == 0
        ):
            self._log_training_step()

        if (
            self.state.global_step
            % self.config.eval_every_steps
            == 0
        ):
            validation_loss = self.evaluate()

            self.state.validation_loss = (
                validation_loss
            )

            is_best = (
                validation_loss
                < self.state.best_validation_loss
            )

            if is_best:
                self.state.best_validation_loss = (
                    validation_loss
                )

            self._sync_metrics()

            self._save_checkpoint(
                is_best=is_best
            )

        elif (
            self.state.global_step
            % self.config.checkpoint_every_steps
            == 0
        ):
            self._save_checkpoint(
                is_best=False
            )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate the model on the validation set."""

        self.model.eval()

        total_loss = 0.0
        batch_count = 0

        for batch in self.validation_loader:
            inputs, targets = self._prepare_batch(
                batch
            )

            model_output = self.model(inputs)
            logits = (
                model_output[0]
                if isinstance(model_output, tuple)
                else model_output
            )

            loss = self.loss_fn(
                logits,
                targets,
            )

            total_loss += float(
                loss.item()
            )

            batch_count += 1

            if (
                self.config.max_eval_batches
                is not None
                and batch_count
                >= self.config.max_eval_batches
            ):
                break

        self.model.train()

        if batch_count == 0:
            return math.inf

        return total_loss / batch_count

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        *,
        is_best: bool,
    ) -> None:
        self.state.elapsed_seconds = (
            self.timer.elapsed()
        )

        self._sync_metrics()

        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.state.epoch,
            global_step=self.state.global_step,
            best_validation_loss=(
                self.state.best_validation_loss
            ),
            metrics=self.metrics.to_dict(),
            is_best=is_best,
        )

    def resume(
        self,
        *,
        filename: str = "latest.pt",
    ) -> TrainingState:
        """Resume training from a checkpoint."""

        metadata = self.checkpoint_manager.load(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            filename=filename,
            map_location=self.device,
        )

        self.state.epoch = metadata[
            "epoch"
        ]

        self.state.global_step = metadata[
            "global_step"
        ]

        self.state.best_validation_loss = (
            metadata["best_validation_loss"]
        )

        checkpoint_metrics = metadata.get(
            "metrics",
            {},
        )

        self.state.train_loss = (
            checkpoint_metrics.get(
                "train_loss",
                0.0,
            )
        )

        self.state.validation_loss = (
            checkpoint_metrics.get(
                "validation_loss",
                math.inf,
            )
        )

        self.state.total_tokens = int(
            checkpoint_metrics.get(
                "total_tokens",
                0,
            )
        )

        self.state.elapsed_seconds = float(
            checkpoint_metrics.get(
                "elapsed_seconds",
                0.0,
            )
        )

        self.state.last_learning_rate = float(
            checkpoint_metrics.get(
                "learning_rate",
                self.optimizer.param_groups[0][
                    "lr"
                ],
            )
        )

        self.state.last_gradient_norm = float(
            checkpoint_metrics.get(
                "gradient_norm",
                0.0,
            )
        )

        self._sync_metrics()

        return self.state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_batch(
        self,
        batch: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        inputs, targets = batch

        if not isinstance(inputs, Tensor):
            inputs = torch.as_tensor(inputs)

        if not isinstance(targets, Tensor):
            targets = torch.as_tensor(targets)

        return (
            inputs.to(
                self.device,
                non_blocking=True,
            ),
            targets.to(
                self.device,
                non_blocking=True,
            ),
        )

    def _should_stop(self) -> bool:
        if self.config.max_steps is None:
            return False

        return (
            self.state.global_step
            >= self.config.max_steps
        )

    def _sync_metrics(self) -> None:
        self.metrics.train_loss = (
            self.state.train_loss
        )

        self.metrics.validation_loss = (
            self.state.validation_loss
        )

        self.metrics.learning_rate = (
            self.state.last_learning_rate
        )

        self.metrics.gradient_norm = (
            self.state.last_gradient_norm
        )

        self.metrics.total_tokens = (
            self.state.total_tokens
        )

        self.metrics.global_step = (
            self.state.global_step
        )

        self.metrics.elapsed_seconds = (
            self.state.elapsed_seconds
        )

    def _log_training_step(self) -> None:
        print(
            f"step={self.state.global_step} "
            f"loss={self.metrics.train_loss:.4f} "
            f"ppl={self.metrics.train_perplexity:.2f} "
            f"lr={self.metrics.learning_rate:.6e} "
            f"grad_norm={self.metrics.gradient_norm:.4f} "
            f"tokens={self.metrics.total_tokens:,} "
            f"tokens/sec={self.metrics.tokens_per_second:.2f}"
        )