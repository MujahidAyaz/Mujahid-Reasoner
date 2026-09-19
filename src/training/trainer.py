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
from src.training.runtime import TrainingRuntime


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for the training loop."""

    device: str = "cpu"
    precision: str = "fp32"

    allow_tf32: bool = True
    cudnn_benchmark: bool = True

    gradient_checkpointing: bool = False

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
                "max_steps must be greater than 0 when provided."
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
                "max_eval_batches must be greater than 0 "
                "when provided."
            )

        if self.seed < 0:
            raise ValueError(
                "seed must be non-negative."
            )


@dataclass
class TrainingState:
    """Mutable state of the training process."""

    epoch: int = 0
    global_step: int = 0
    batch_in_epoch: int = 0

    best_validation_loss: float = math.inf

    train_loss: float = 0.0
    validation_loss: float = math.inf

    total_tokens: int = 0
    elapsed_seconds: float = 0.0

    last_gradient_norm: float = 0.0
    last_learning_rate: float = 0.0


class Trainer:
    """
    Production-oriented language-model trainer.

    Features:
        - automatic device/runtime configuration
        - FP32/FP16/BF16 support
        - automatic mixed precision
        - CUDA GradScaler for FP16
        - optional gradient checkpointing
        - true micro-batch gradient accumulation
        - correct partial accumulation handling
        - gradient clipping after normalization
        - optimizer stepping
        - scheduler stepping
        - validation
        - checkpointing
        - deterministic sampler-aware resume
        - metric tracking
        - cumulative elapsed training time

    Gradient accumulation semantics:

        N micro-batches
            ↓
        forward/backward
            ↓
        average accumulated gradients
            ↓
        gradient clipping
            ↓
        optimizer.step()
            ↓
        scheduler.step()

    The accumulated gradient is always the arithmetic mean of the
    micro-batch gradients. This remains correct when an epoch ends
    with a partial accumulation window.

    Gradient checkpointing:

        When enabled in TrainerConfig, the trainer activates the
        model's gradient-checkpointing implementation when available.

        Checkpointing is a training-time activation-memory optimization.
        It is automatically inactive during evaluation because the model
        is placed in evaluation mode.

        KV caching remains independent from checkpointing and is not
        used during training.

    Resume semantics:

        If a resumable sampler is provided, its internal position is
        authoritative. The trainer does not independently skip those
        already-consumed samples a second time.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader,
        validation_loader,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        config: TrainerConfig,
        loss_fn: CausalLanguageModelLoss | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        train_sampler: Any | None = None,
    ) -> None:
        self.config = config

        # --------------------------------------------------------------
        # Runtime
        # --------------------------------------------------------------

        self.runtime = TrainingRuntime(
            device=config.device,
            precision=config.precision,
            allow_tf32=config.allow_tf32,
            cudnn_benchmark=config.cudnn_benchmark,
        )

        self.device = self.runtime.device

        self.model = model.to(
            self.device
        )

        # --------------------------------------------------------------
        # Gradient checkpointing
        # --------------------------------------------------------------

        self._configure_gradient_checkpointing()

        self.scaler = (
            self.runtime.create_grad_scaler()
        )

        # --------------------------------------------------------------
        # Data
        # --------------------------------------------------------------

        self.train_loader = train_loader
        self.validation_loader = validation_loader

        # --------------------------------------------------------------
        # Optimization
        # --------------------------------------------------------------

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.loss_fn = (
            loss_fn
            if loss_fn is not None
            else CausalLanguageModelLoss()
        )

        # --------------------------------------------------------------
        # Checkpointing
        # --------------------------------------------------------------

        self.checkpoint_manager = (
            checkpoint_manager
            if checkpoint_manager is not None
            else CheckpointManager(
                Path(config.output_dir)
                / "checkpoints"
            )
        )

        self.train_sampler = train_sampler

        # --------------------------------------------------------------
        # State / metrics / timing
        # --------------------------------------------------------------

        self.state = TrainingState()

        self.metrics = TrainingMetrics()

        self.timer = TrainingTimer()

        self._timer_offset_seconds = 0.0

        self._micro_steps_since_update = 0

        self._accumulated_loss = 0.0
        self._accumulated_tokens = 0

        self._last_evaluated_step = -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> TrainingState:
        """Run the training process."""

        self.timer.start()

        print(
            f"Training runtime: {self.runtime.summary()}"
        )

        for epoch in range(
            self.state.epoch,
            self.config.max_epochs,
        ):
            self.state.epoch = epoch

            self._prepare_epoch(
                epoch
            )

            self._train_epoch()

            if self._should_stop():
                break

            self.state.epoch = epoch + 1
            self.state.batch_in_epoch = 0

            if self.train_sampler is not None:
                self.train_sampler.set_epoch(
                    epoch + 1
                )

        self.state.elapsed_seconds = (
            self._elapsed_training_seconds()
        )

        # Avoid evaluating twice when the final optimizer step
        # already triggered evaluation.
        if (
            self._last_evaluated_step
            != self.state.global_step
        ):
            final_validation_loss = (
                self.evaluate()
            )

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
        else:
            is_best = False

        self.state.elapsed_seconds = (
            self._elapsed_training_seconds()
        )

        self._sync_metrics()

        self._save_checkpoint(
            is_best=is_best
        )

        return self.state

    def resume(
        self,
        filename: str = "latest.pt",
    ) -> TrainingState:
        """
        Restore the complete training state.

        Restores:
            - model
            - optimizer
            - scheduler
            - RNG state
            - metrics
            - epoch
            - batch position
            - sampler position
        """

        metadata = self.checkpoint_manager.load(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            filename=filename,
        )

        self.state.epoch = int(
            metadata["epoch"]
        )

        self.state.global_step = int(
            metadata["global_step"]
        )

        self.state.batch_in_epoch = int(
            metadata.get(
                "batch_in_epoch",
                0,
            )
        )

        self.state.best_validation_loss = float(
            metadata["best_validation_loss"]
        )

        metrics = metadata.get(
            "metrics",
            {},
        )

        if isinstance(
            metrics,
            dict,
        ):
            self._restore_metrics(
                metrics
            )

        self._timer_offset_seconds = (
            self.state.elapsed_seconds
        )

        dataloader_state = metadata.get(
            "dataloader_state"
        )

        if dataloader_state is not None:
            self._restore_dataloader_state(
                dataloader_state
            )

        elif self.train_sampler is not None:
            # Compatibility with checkpoints that contain
            # batch_in_epoch but no sampler state.
            self._restore_sampler_position_fallback()

        self._sync_metrics()

        return self.state

    def evaluate(self) -> float:
        """Evaluate the model on the validation DataLoader."""

        was_training = self.model.training

        self.model.eval()

        total_loss = 0.0
        batch_count = 0

        try:
            with torch.no_grad():
                for batch in self.validation_loader:
                    inputs, targets = (
                        self._prepare_batch(
                            batch
                        )
                    )

                    with self.runtime.autocast_context():
                        model_output = self.model(
                            inputs
                        )

                        logits = (
                            model_output[0]
                            if isinstance(
                                model_output,
                                tuple,
                            )
                            else model_output
                        )

                        loss = self.loss_fn(
                            logits,
                            targets,
                        )

                    total_loss += float(
                        loss.detach().item()
                    )

                    batch_count += 1

                    if (
                        self.config.max_eval_batches
                        is not None
                        and batch_count
                        >= self.config.max_eval_batches
                    ):
                        break

        finally:
            if was_training:
                self.model.train()

        if batch_count == 0:
            return math.inf

        validation_loss = (
            total_loss / batch_count
        )

        self._last_evaluated_step = (
            self.state.global_step
        )

        return validation_loss

    # ------------------------------------------------------------------
    # Runtime configuration
    # ------------------------------------------------------------------

    def _configure_gradient_checkpointing(
        self,
    ) -> None:
        """
        Configure model-level gradient checkpointing.

        MujahidReasonerModel exposes explicit enable/disable methods.
        The trainer uses duck typing so lightweight test models and
        alternative model implementations remain compatible.

        If checkpointing is requested but the model does not expose
        the required API, fail immediately rather than silently
        running a different experiment than requested.
        """

        enable = getattr(
            self.model,
            "enable_gradient_checkpointing",
            None,
        )

        disable = getattr(
            self.model,
            "disable_gradient_checkpointing",
            None,
        )

        if self.config.gradient_checkpointing:
            if not callable(enable):
                raise TypeError(
                    "gradient_checkpointing=True requires the model "
                    "to implement enable_gradient_checkpointing()."
                )

            enable()

            return

        if callable(disable):
            disable()

    # ------------------------------------------------------------------
    # Epoch handling
    # ------------------------------------------------------------------

    def _prepare_epoch(
        self,
        epoch: int,
    ) -> None:
        """Prepare the sampler for the requested epoch."""

        if self.train_sampler is None:
            return

        sampler_epoch = getattr(
            self.train_sampler,
            "epoch",
            None,
        )

        if sampler_epoch != epoch:
            self.train_sampler.set_epoch(
                epoch
            )

    def _train_epoch(self) -> None:
        """Train over the remaining batches of the current epoch."""

        self.model.train()

        self.optimizer.zero_grad(
            set_to_none=True
        )

        self._reset_accumulation()

        # The resumable sampler owns the consumed-sample position.
        #
        # Therefore, when a sampler is present, do NOT skip
        # state.batch_in_epoch again. The DataLoader will already
        # start from the restored sampler position.
        #
        # For non-resumable loaders, batch_in_epoch remains the
        # fallback resume mechanism.
        use_sampler_position = (
            self.train_sampler is not None
            and hasattr(
                self.train_sampler,
                "state_dict",
            )
        )

        batches_to_skip = (
            0
            if use_sampler_position
            else self.state.batch_in_epoch
        )

        for batch_index, batch in enumerate(
            self.train_loader
        ):
            if batch_index < batches_to_skip:
                continue

            inputs, targets = (
                self._prepare_batch(
                    batch
                )
            )

            with self.runtime.autocast_context():
                model_output = self.model(
                    inputs
                )

                logits = (
                    model_output[0]
                    if isinstance(
                        model_output,
                        tuple,
                    )
                    else model_output
                )

                loss = self.loss_fn(
                    logits,
                    targets,
                )

            # Accumulate the raw micro-batch loss.
            #
            # Normalization is deliberately deferred until the
            # optimizer step because the final accumulation window
            # may contain fewer micro-batches than configured.
            self._backward(
                loss
            )

            self._accumulated_loss += float(
                loss.detach().item()
            )

            self._accumulated_tokens += (
                targets.numel()
            )

            self._micro_steps_since_update += 1

            self.state.batch_in_epoch = (
                self._next_batch_position(
                    batch_index
                )
            )

            if (
                self._micro_steps_since_update
                >= self.config.gradient_accumulation_steps
            ):
                self._optimizer_step()

                if self._should_stop():
                    return

        # If the epoch ended with a partial accumulation,
        # average those gradients over the actual number of
        # micro-batches rather than the configured maximum.
        if not self._should_stop():
            self._flush_remaining_gradients()

    def _next_batch_position(
        self,
        batch_index: int,
    ) -> int:
        """
        Return the number of batches consumed in the current epoch.

        For a resumable sampler, enumerate() starts from the remaining
        portion of the DataLoader after restoration, so batch_index is
        relative to the resumed iterator. We therefore derive the
        absolute position from the sampler whenever possible.
        """

        if self.train_sampler is not None:
            position = getattr(
                self.train_sampler,
                "position",
                None,
            )

            if position is not None:
                return self._samples_to_batches(
                    int(position)
                )

        return batch_index + 1

    # ------------------------------------------------------------------
    # Gradient accumulation
    # ------------------------------------------------------------------

    def _reset_accumulation(self) -> None:
        """Reset the current micro-batch accumulation window."""

        self._micro_steps_since_update = 0
        self._accumulated_loss = 0.0
        self._accumulated_tokens = 0

    def _backward(
        self,
        loss: Tensor,
    ) -> None:
        """
        Run backward for one micro-batch.

        The loss is intentionally not divided here. Gradient
        normalization is performed exactly once immediately before
        clipping and the optimizer update.
        """

        if self.scaler is not None:
            self.scaler.scale(
                loss
            ).backward()

            return

        loss.backward()

    def _normalize_accumulated_gradients(
        self,
        accumulation_count: int,
    ) -> None:
        """
        Convert accumulated gradients into their arithmetic mean.

        This is done after GradScaler unscaling so the operation is
        numerically correct for FP16 training as well.
        """

        if accumulation_count <= 0:
            raise ValueError(
                "accumulation_count must be greater than 0."
            )

        for parameter in self.model.parameters():
            gradient = parameter.grad

            if gradient is None:
                continue

            gradient.div_(
                accumulation_count
            )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def _optimizer_step(self) -> None:
        """Perform one optimizer update from accumulated gradients."""

        accumulation_count = (
            self._micro_steps_since_update
        )

        if accumulation_count <= 0:
            return

        # GradScaler gradients must be unscaled before:
        #   1. normalization
        #   2. gradient clipping
        if self.scaler is not None:
            self.scaler.unscale_(
                self.optimizer
            )

        self._normalize_accumulated_gradients(
            accumulation_count
        )

        gradient_norm = clip_gradients(
            self.model,
            max_norm=self.config.max_grad_norm,
        )

        optimizer_step_succeeded = True

        if self.scaler is not None:
            old_scale = self.scaler.get_scale()

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            new_scale = self.scaler.get_scale()

            # A reduced scale indicates that the optimizer step
            # was skipped because non-finite gradients were detected.
            optimizer_step_succeeded = (
                new_scale >= old_scale
            )

        else:
            self.optimizer.step()

        if optimizer_step_succeeded:
            if self.scheduler is not None:
                self.scheduler.step()

            self.state.global_step += 1

        self.optimizer.zero_grad(
            set_to_none=True
        )

        self.state.train_loss = (
            self._accumulated_loss
            / accumulation_count
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

        self._reset_accumulation()

        self.state.elapsed_seconds = (
            self._elapsed_training_seconds()
        )

        self._sync_metrics()

        # Do not trigger evaluation/checkpointing for an optimizer
        # update that was skipped because of numerical overflow.
        if not optimizer_step_succeeded:
            return

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
            validation_loss = (
                self.evaluate()
            )

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

    def _flush_remaining_gradients(self) -> None:
        """Apply a final partial gradient accumulation."""

        if (
            self._micro_steps_since_update
            == 0
        ):
            return

        self._optimizer_step()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        *,
        is_best: bool,
    ) -> None:
        """Save complete training state."""

        self.state.elapsed_seconds = (
            self._elapsed_training_seconds()
        )

        self._sync_metrics()

        dataloader_state = (
            self._build_dataloader_state()
        )

        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.state.epoch,
            batch_in_epoch=(
                self.state.batch_in_epoch
            ),
            global_step=self.state.global_step,
            best_validation_loss=(
                self.state.best_validation_loss
            ),
            metrics=self.metrics.to_dict(),
            is_best=is_best,
            dataloader_state=dataloader_state,
        )

    def _build_dataloader_state(
        self,
    ) -> dict[str, Any] | None:
        """Build serializable DataLoader state."""

        if self.train_sampler is None:
            return None

        if not hasattr(
            self.train_sampler,
            "state_dict",
        ):
            return None

        return {
            "train_sampler": (
                self.train_sampler.state_dict()
            )
        }

    def _restore_dataloader_state(
        self,
        dataloader_state: Any,
    ) -> None:
        """Restore sampler state from checkpoint."""

        if not isinstance(
            dataloader_state,
            dict,
        ):
            raise ValueError(
                "dataloader_state must be a dictionary."
            )

        if not dataloader_state:
            return

        sampler_state = dataloader_state.get(
            "train_sampler"
        )

        if sampler_state is None:
            raise ValueError(
                "dataloader_state is missing "
                "'train_sampler'."
            )

        if self.train_sampler is None:
            raise RuntimeError(
                "Checkpoint contains sampler state, "
                "but the current Trainer has no train sampler."
            )

        if not hasattr(
            self.train_sampler,
            "load_state_dict",
        ):
            raise RuntimeError(
                "Current training sampler does not support "
                "state restoration."
            )

        if not isinstance(
            sampler_state,
            dict,
        ):
            raise ValueError(
                "train_sampler state must be a dictionary."
            )

        self.train_sampler.load_state_dict(
            sampler_state
        )

        self.state.epoch = int(
            sampler_state["epoch"]
        )

        self.state.batch_in_epoch = (
            self._samples_to_batches(
                int(
                    sampler_state[
                        "position"
                    ]
                )
            )
        )

    def _restore_sampler_position_fallback(
        self,
    ) -> None:
        """Restore sampler position for older checkpoints."""

        if self.train_sampler is None:
            return

        if not hasattr(
            self.train_sampler,
            "set_epoch",
        ):
            return

        self.train_sampler.set_epoch(
            self.state.epoch
        )

        if hasattr(
            self.train_sampler,
            "position",
        ):
            self.train_sampler.position = (
                self._batches_to_samples(
                    self.state.batch_in_epoch
                )
            )

    def _batches_to_samples(
        self,
        batch_count: int,
    ) -> int:
        """Convert processed batches into processed samples."""

        batch_size = getattr(
            self.train_loader,
            "batch_size",
            None,
        )

        if batch_size is None:
            raise RuntimeError(
                "Cannot determine batch size from train_loader."
            )

        return (
            batch_count
            * int(batch_size)
        )

    def _samples_to_batches(
        self,
        sample_count: int,
    ) -> int:
        """Convert processed samples into processed batches."""

        batch_size = getattr(
            self.train_loader,
            "batch_size",
            None,
        )

        if batch_size is None:
            raise RuntimeError(
                "Cannot determine batch size from train_loader."
            )

        batch_size = int(
            batch_size
        )

        if sample_count == 0:
            return 0

        return sample_count // batch_size

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def _prepare_batch(
        self,
        batch: Any,
    ) -> tuple[Tensor, Tensor]:
        """Move a batch to the resolved runtime device."""

        if not isinstance(
            batch,
            (tuple, list),
        ):
            raise TypeError(
                "Expected batch to be a tuple or list "
                "containing inputs and targets."
            )

        if len(batch) != 2:
            raise ValueError(
                "Expected batch to contain exactly "
                "inputs and targets."
            )

        inputs, targets = batch

        if not isinstance(
            inputs,
            Tensor,
        ):
            inputs = torch.as_tensor(
                inputs,
                dtype=torch.long,
            )

        if not isinstance(
            targets,
            Tensor,
        ):
            targets = torch.as_tensor(
                targets,
                dtype=torch.long,
            )

        inputs = inputs.to(
            self.device,
            non_blocking=True,
        )

        targets = targets.to(
            self.device,
            non_blocking=True,
        )

        return inputs, targets

    # ------------------------------------------------------------------
    # Stopping
    # ------------------------------------------------------------------

    def _should_stop(self) -> bool:
        """Return whether maximum training steps were reached."""

        if self.config.max_steps is None:
            return False

        return (
            self.state.global_step
            >= self.config.max_steps
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _sync_metrics(self) -> None:
        """Synchronize TrainingMetrics with TrainingState."""

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

    def _restore_metrics(
        self,
        metrics: dict[str, Any],
    ) -> None:
        """Restore persisted metric values."""

        self.state.train_loss = float(
            metrics.get(
                "train_loss",
                self.state.train_loss,
            )
        )

        self.state.validation_loss = float(
            metrics.get(
                "validation_loss",
                self.state.validation_loss,
            )
        )

        self.state.last_learning_rate = float(
            metrics.get(
                "learning_rate",
                self.state.last_learning_rate,
            )
        )

        self.state.last_gradient_norm = float(
            metrics.get(
                "gradient_norm",
                self.state.last_gradient_norm,
            )
        )

        self.state.total_tokens = int(
            metrics.get(
                "total_tokens",
                metrics.get(
                    "tokens",
                    self.state.total_tokens,
                ),
            )
        )

        self.state.global_step = int(
            metrics.get(
                "global_step",
                self.state.global_step,
            )
        )

        self.state.elapsed_seconds = float(
            metrics.get(
                "elapsed_seconds",
                self.state.elapsed_seconds,
            )
        )

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def _elapsed_training_seconds(self) -> float:
        """Return cumulative elapsed time across resumed runs."""

        return (
            self._timer_offset_seconds
            + self.timer.elapsed()
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_training_step(self) -> None:
        """Print concise training progress."""

        print(
            f"step={self.state.global_step} "
            f"epoch={self.state.epoch} "
            f"batch={self.state.batch_in_epoch} "
            f"train_loss={self.state.train_loss:.4f} "
            f"val_loss={self.state.validation_loss:.4f} "
            f"lr={self.state.last_learning_rate:.6g} "
            f"tokens={self.state.total_tokens:,}"
        )