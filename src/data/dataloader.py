from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from src.data.tokenized_dataset import PackedTokenDataset


@dataclass
class SamplerState:
    """
    Serializable state for deterministic resumable sampling.

    Attributes:
        epoch:
            Current dataset epoch.

        position:
            Number of samples already consumed in the current epoch.

        seed:
            Base seed used to deterministically generate the epoch
            permutation.
    """

    epoch: int = 0
    position: int = 0
    seed: int = 42


class ResumableRandomSampler(Sampler[int]):
    """
    Deterministic random sampler that supports exact mid-epoch resume.

    Unlike PyTorch's default RandomSampler, this sampler does not depend
    on the global RNG state to recreate an epoch permutation.

    The permutation is deterministically generated from:

        seed + epoch

    Only the current position needs to be checkpointed. This keeps
    checkpoint size independent of dataset size.

    Exact resume assumes the sampler is consumed from a single-process
    DataLoader (num_workers=0), which avoids worker prefetch advancing
    the sampler ahead of the processed training batch.
    """

    def __init__(
        self,
        data_source,
        *,
        seed: int = 42,
        epoch: int = 0,
        position: int = 0,
    ) -> None:
        if seed < 0:
            raise ValueError(
                "seed must be non-negative."
            )

        if epoch < 0:
            raise ValueError(
                "epoch must be non-negative."
            )

        if position < 0:
            raise ValueError(
                "position must be non-negative."
            )

        self.data_source = data_source
        self.seed = seed
        self.epoch = epoch
        self.position = position

        self._validate_position()

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        """
        Yield the remaining indices for the current epoch.

        The permutation is generated deterministically from the sampler
        seed and epoch. The current position is advanced as indices are
        yielded.
        """

        permutation = self._generate_permutation()

        if self.position > len(permutation):
            raise RuntimeError(
                "Sampler position exceeds dataset length."
            )

        while self.position < len(permutation):
            index = int(
                permutation[self.position].item()
            )

            self.position += 1

            yield index

    def set_epoch(
        self,
        epoch: int,
    ) -> None:
        """
        Start a new epoch.

        The sample position is reset because every epoch receives a
        fresh deterministic permutation.
        """

        if epoch < 0:
            raise ValueError(
                "epoch must be non-negative."
            )

        self.epoch = epoch
        self.position = 0

    def state_dict(self) -> dict[str, int]:
        """Return serializable sampler state."""

        return {
            "epoch": self.epoch,
            "position": self.position,
            "seed": self.seed,
        }

    def load_state_dict(
        self,
        state: dict[str, int],
    ) -> None:
        """Restore sampler state."""

        required_keys = {
            "epoch",
            "position",
            "seed",
        }

        missing_keys = required_keys.difference(
            state.keys()
        )

        if missing_keys:
            raise ValueError(
                "Sampler state is missing required keys: "
                f"{sorted(missing_keys)}"
            )

        epoch = int(state["epoch"])
        position = int(state["position"])
        seed = int(state["seed"])

        if epoch < 0:
            raise ValueError(
                "Sampler epoch must be non-negative."
            )

        if position < 0:
            raise ValueError(
                "Sampler position must be non-negative."
            )

        if seed < 0:
            raise ValueError(
                "Sampler seed must be non-negative."
            )

        if seed != self.seed:
            raise ValueError(
                "Sampler seed mismatch. "
                f"Checkpoint seed={seed}, "
                f"current seed={self.seed}."
            )

        self.epoch = epoch
        self.position = position

        self._validate_position()

    def _generate_permutation(self) -> torch.Tensor:
        """
        Generate the deterministic permutation for the current epoch.

        A private generator is used so sampler generation does not modify
        the global PyTorch RNG state used by model training.
        """

        generator = torch.Generator()

        generator.manual_seed(
            self._epoch_seed()
        )

        return torch.randperm(
            len(self.data_source),
            generator=generator,
        )

    def _epoch_seed(self) -> int:
        """
        Derive a deterministic seed for the current epoch.

        The multiplier provides separation between neighboring epoch
        seeds while remaining deterministic and inexpensive.
        """

        return (
            self.seed
            + (self.epoch * 1_000_003)
        ) % (
            2**63 - 1
        )

    def _validate_position(self) -> None:
        """Validate sampler position against dataset size."""

        dataset_length = len(self.data_source)

        if self.position > dataset_length:
            raise ValueError(
                "Sampler position cannot exceed dataset length. "
                f"position={self.position}, "
                f"dataset_length={dataset_length}."
            )


class DataLoaderConfig:
    """Configuration for training and validation DataLoaders."""

    def __init__(
        self,
        *,
        batch_size: int = 2,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError(
                "batch_size must be at least 1."
            )

        if num_workers < 0:
            raise ValueError(
                "num_workers cannot be negative."
            )

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last


class LanguageModelDataModule:
    """
    Build PyTorch DataLoaders for Mujahid-Reasoner.

    Training uses a deterministic resumable sampler whenever shuffling
    is enabled. Validation remains sequential and deterministic.
    """

    def __init__(
        self,
        *,
        train_file: Path,
        validation_file: Path,
        sequence_length: int,
        config: DataLoaderConfig,
        seed: int = 42,
    ) -> None:
        if seed < 0:
            raise ValueError(
                "seed must be non-negative."
            )

        self.train_file = train_file
        self.validation_file = validation_file
        self.sequence_length = sequence_length
        self.config = config
        self.seed = seed

        self._train_dataset: PackedTokenDataset | None = None
        self._validation_dataset: PackedTokenDataset | None = None
        self._train_sampler: ResumableRandomSampler | None = None

    @property
    def train_dataset(self) -> PackedTokenDataset:
        """Return the training dataset."""

        if self._train_dataset is None:
            self._train_dataset = PackedTokenDataset(
                token_file=self.train_file,
                sequence_length=self.sequence_length,
            )

        return self._train_dataset

    @property
    def validation_dataset(self) -> PackedTokenDataset:
        """Return the validation dataset."""

        if self._validation_dataset is None:
            self._validation_dataset = PackedTokenDataset(
                token_file=self.validation_file,
                sequence_length=self.sequence_length,
            )

        return self._validation_dataset

    @property
    def train_sampler(
        self,
    ) -> ResumableRandomSampler | None:
        """
        Return the training sampler.

        Returns None when shuffle is disabled.
        """

        return self._train_sampler

    def train_dataloader(self) -> DataLoader:
        """Create the training DataLoader."""

        if self.config.shuffle:
            if self.config.num_workers != 0:
                raise ValueError(
                    "Exact resumable training currently requires "
                    "num_workers=0."
                )

            self._train_sampler = ResumableRandomSampler(
                self.train_dataset,
                seed=self.seed,
            )

            return DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                sampler=self._train_sampler,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=self.config.drop_last,
            )

        self._train_sampler = None

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
        )

    def validation_dataloader(self) -> DataLoader:
        """Create the validation DataLoader."""

        return DataLoader(
            self.validation_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
        )

    def state_dict(self) -> dict[str, object]:
        """
        Return serializable DataLoader state.

        The state is intentionally small. The sampler stores only its
        deterministic seed, epoch and current position rather than the
        entire shuffled permutation.
        """

        state: dict[str, object] = {
            "seed": self.seed,
        }

        if self._train_sampler is not None:
            state["train_sampler"] = (
                self._train_sampler.state_dict()
            )

        return state

    def load_state_dict(
        self,
        state: dict[str, object],
    ) -> None:
        """Restore DataLoader and sampler state."""

        if "seed" not in state:
            raise ValueError(
                "DataLoader state is missing 'seed'."
            )

        checkpoint_seed = int(
            state["seed"]
        )

        if checkpoint_seed != self.seed:
            raise ValueError(
                "DataLoader seed mismatch. "
                f"Checkpoint seed={checkpoint_seed}, "
                f"current seed={self.seed}."
            )

        sampler_state = state.get(
            "train_sampler"
        )

        if sampler_state is None:
            return

        if self._train_sampler is None:
            raise ValueError(
                "Checkpoint contains training sampler state, "
                "but the current DataLoader has no resumable sampler."
            )

        if not isinstance(
            sampler_state,
            dict,
        ):
            raise ValueError(
                "train_sampler state must be a dictionary."
            )

        self._train_sampler.load_state_dict(
            sampler_state
        )


def set_seed(
    seed: int,
) -> None:
    """Set random seeds for reproducible experiments."""

    if seed < 0:
        raise ValueError(
            "seed must be non-negative."
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)