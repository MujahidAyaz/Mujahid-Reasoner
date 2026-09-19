from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


CHECKPOINT_VERSION = 2


class CheckpointManager:
    """
    Save and restore complete training state.

    The checkpoint contains:
        - model state
        - optimizer state
        - scheduler state
        - current epoch
        - current batch position
        - current global step
        - best validation loss
        - training metrics
        - Python RNG state
        - NumPy RNG state
        - PyTorch CPU RNG state
        - PyTorch CUDA RNG states
        - checkpoint format version

    `batch_in_epoch` represents the next batch that should be processed
    after restoring the checkpoint.

    Checkpoint writes are atomic: data is first written to a temporary
    file and then moved into place.
    """

    def __init__(
        self,
        directory: str | Path,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    @property
    def latest_path(self) -> Path:
        return self.directory / "latest.pt"

    @property
    def best_path(self) -> Path:
        return self.directory / "best.pt"

    def save(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        epoch: int,
        global_step: int,
        best_validation_loss: float,
        metrics: Mapping[str, float] | None = None,
        is_best: bool = False,
        filename: str = "latest.pt",
        batch_in_epoch: int = 0,
        dataloader_state: Mapping[str, Any] | None = None,
    ) -> Path:
        """
        Save a complete training checkpoint.

        Args:
            model:
                Model whose parameters should be saved.

            optimizer:
                Optimizer state to save.

            scheduler:
                Learning-rate scheduler state, if used.

            epoch:
                Zero-based epoch containing the next batch to process.

            global_step:
                Number of optimizer updates completed.

            best_validation_loss:
                Lowest validation loss observed so far.

            metrics:
                Additional scalar training metrics.

            is_best:
                Also update `best.pt` when True.

            filename:
                Name of the primary checkpoint file.

            batch_in_epoch:
                Zero-based index of the next batch to process within
                `epoch`.

            dataloader_state:
                Optional serializable state required by a custom
                dataloader/sampler implementation.
        """

        self._validate_save_arguments(
            epoch=epoch,
            global_step=global_step,
            best_validation_loss=best_validation_loss,
            batch_in_epoch=batch_in_epoch,
            filename=filename,
        )

        checkpoint: dict[str, Any] = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "global_step": global_step,
            "best_validation_loss": float(
                best_validation_loss
            ),
            "metrics": dict(metrics or {}),
            "rng_state": self._capture_rng_state(),
            "dataloader_state": dict(
                dataloader_state or {}
            ),
        }

        path = self.directory / filename
        self._atomic_save(checkpoint, path)

        if is_best and path != self.best_path:
            self._atomic_save(
                checkpoint,
                self.best_path,
            )

        return path

    def load(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        filename: str = "latest.pt",
        map_location: str | torch.device = "cpu",
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        """
        Load a complete training checkpoint.

        Returns:
            Dictionary containing training metadata and resume state.
        """

        path = self.directory / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

        self._validate_checkpoint_structure(
            checkpoint,
            path,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scheduler_state = checkpoint[
            "scheduler_state_dict"
        ]

        if scheduler is not None:
            if scheduler_state is None:
                raise ValueError(
                    "Checkpoint does not contain scheduler "
                    "state, but a scheduler was provided."
                )

            scheduler.load_state_dict(
                scheduler_state
            )

        elif scheduler_state is not None:
            raise ValueError(
                "Checkpoint contains scheduler state, "
                "but no scheduler was provided."
            )

        if restore_rng:
            self._restore_rng_state(
                checkpoint["rng_state"]
            )

        return {
            "checkpoint_version": checkpoint[
                "checkpoint_version"
            ],
            "epoch": checkpoint["epoch"],
            "batch_in_epoch": checkpoint[
                "batch_in_epoch"
            ],
            "global_step": checkpoint[
                "global_step"
            ],
            "best_validation_loss": checkpoint[
                "best_validation_loss"
            ],
            "metrics": dict(
                checkpoint["metrics"]
            ),
            "dataloader_state": dict(
                checkpoint.get(
                    "dataloader_state",
                    {},
                )
            ),
        }

    @staticmethod
    def _validate_save_arguments(
        *,
        epoch: int,
        global_step: int,
        best_validation_loss: float,
        batch_in_epoch: int,
        filename: str,
    ) -> None:
        if epoch < 0:
            raise ValueError(
                "epoch must be >= 0."
            )

        if global_step < 0:
            raise ValueError(
                "global_step must be >= 0."
            )

        if batch_in_epoch < 0:
            raise ValueError(
                "batch_in_epoch must be >= 0."
            )

        if best_validation_loss < 0:
            raise ValueError(
                "best_validation_loss must be >= 0."
            )

        if not filename.strip():
            raise ValueError(
                "filename must not be empty."
            )

        if Path(filename).name != filename:
            raise ValueError(
                "filename must contain only a file name, "
                "not a directory path."
            )

    @staticmethod
    def _validate_checkpoint_structure(
        checkpoint: Any,
        path: Path,
    ) -> None:
        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"Invalid checkpoint format: {path}"
            )

        required_keys = {
            "checkpoint_version",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "epoch",
            "batch_in_epoch",
            "global_step",
            "best_validation_loss",
            "metrics",
            "rng_state",
            "dataloader_state",
        }

        missing_keys = required_keys.difference(
            checkpoint.keys()
        )

        if missing_keys:
            raise ValueError(
                "Checkpoint is missing required keys: "
                f"{sorted(missing_keys)}"
            )

        version = checkpoint[
            "checkpoint_version"
        ]

        if version != CHECKPOINT_VERSION:
            raise ValueError(
                "Unsupported checkpoint version: "
                f"{version}. Expected "
                f"{CHECKPOINT_VERSION}."
            )

        if not isinstance(
            checkpoint["epoch"],
            int,
        ):
            raise ValueError(
                "Checkpoint field 'epoch' must be an int."
            )

        if not isinstance(
            checkpoint["batch_in_epoch"],
            int,
        ):
            raise ValueError(
                "Checkpoint field 'batch_in_epoch' "
                "must be an int."
            )

        if not isinstance(
            checkpoint["global_step"],
            int,
        ):
            raise ValueError(
                "Checkpoint field 'global_step' "
                "must be an int."
            )

        if checkpoint["epoch"] < 0:
            raise ValueError(
                "Checkpoint field 'epoch' must be >= 0."
            )

        if checkpoint["batch_in_epoch"] < 0:
            raise ValueError(
                "Checkpoint field 'batch_in_epoch' "
                "must be >= 0."
            )

        if checkpoint["global_step"] < 0:
            raise ValueError(
                "Checkpoint field 'global_step' "
                "must be >= 0."
            )

        if not isinstance(
            checkpoint["metrics"],
            dict,
        ):
            raise ValueError(
                "Checkpoint field 'metrics' must be a dict."
            )

        if not isinstance(
            checkpoint["rng_state"],
            dict,
        ):
            raise ValueError(
                "Checkpoint field 'rng_state' "
                "must be a dict."
            )

        if not isinstance(
            checkpoint["dataloader_state"],
            dict,
        ):
            raise ValueError(
                "Checkpoint field 'dataloader_state' "
                "must be a dict."
            )

    @staticmethod
    def _atomic_save(
        checkpoint: Mapping[str, Any],
        path: Path,
    ) -> None:
        temporary_path = path.with_suffix(
            path.suffix + ".tmp"
        )

        try:
            torch.save(
                checkpoint,
                temporary_path,
            )
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        """
        Capture Python, NumPy, and PyTorch RNG states.
        """

        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }

        if torch.cuda.is_available():
            state["cuda"] = (
                torch.cuda.get_rng_state_all()
            )

        return state

    @staticmethod
    def _restore_rng_state(
        state: Mapping[str, Any],
    ) -> None:
        """
        Restore Python, NumPy, and PyTorch RNG states.
        """

        required_keys = {
            "python",
            "numpy",
            "torch",
        }

        missing_keys = required_keys.difference(
            state.keys()
        )

        if missing_keys:
            raise ValueError(
                "RNG state is missing required keys: "
                f"{sorted(missing_keys)}"
            )

        random.setstate(
            state["python"]
        )

        np.random.set_state(
            state["numpy"]
        )

        torch.set_rng_state(
            state["torch"]
        )

        if (
            "cuda" in state
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(
                state["cuda"]
            )