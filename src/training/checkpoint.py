from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CheckpointManager:
    """
    Save and restore complete training state.

    Stores:
        - model state
        - optimizer state
        - scheduler state
        - current epoch
        - current global step
        - best validation loss
        - training metrics
        - random-number-generator states
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
        metrics: dict[str, float] | None = None,
        is_best: bool = False,
        filename: str = "latest.pt",
    ) -> Path:
        """
        Save a complete training checkpoint.
        """

        if epoch < 0:
            raise ValueError("epoch must be >= 0.")

        if global_step < 0:
            raise ValueError("global_step must be >= 0.")

        if best_validation_loss < 0:
            raise ValueError(
                "best_validation_loss must be >= 0."
            )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
            "epoch": epoch,
            "global_step": global_step,
            "best_validation_loss": best_validation_loss,
            "metrics": metrics or {},
            "rng_state": self._capture_rng_state(),
        }

        path = self.directory / filename

        temporary_path = path.with_suffix(
            path.suffix + ".tmp"
        )

        torch.save(
            checkpoint,
            temporary_path,
        )

        temporary_path.replace(path)

        if is_best:
            best_path = self.best_path

            if path != best_path:
                torch.save(
                    checkpoint,
                    best_path,
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
            Training metadata such as epoch, global step,
            best validation loss, and metrics.
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

        required_keys = {
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "epoch",
            "global_step",
            "best_validation_loss",
            "metrics",
            "rng_state",
        }

        missing_keys = required_keys.difference(
            checkpoint.keys()
        )

        if missing_keys:
            raise ValueError(
                "Checkpoint is missing required keys: "
                f"{sorted(missing_keys)}"
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
            "epoch": checkpoint["epoch"],
            "global_step": checkpoint["global_step"],
            "best_validation_loss": checkpoint[
                "best_validation_loss"
            ],
            "metrics": checkpoint["metrics"],
        }

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
            state["cuda"] = torch.cuda.get_rng_state_all()

        return state

    @staticmethod
    def _restore_rng_state(
        state: dict[str, Any],
    ) -> None:
        """
        Restore Python, NumPy, and PyTorch RNG states.
        """

        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])

        if (
            "cuda" in state
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(
                state["cuda"]
            )