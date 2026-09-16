from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.tokenized_dataset import PackedTokenDataset


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

    This class keeps dataset construction and DataLoader configuration
    in one place so the training loop remains simple.
    """

    def __init__(
        self,
        *,
        train_file: Path,
        validation_file: Path,
        sequence_length: int,
        config: DataLoaderConfig,
    ) -> None:
        self.train_file = train_file
        self.validation_file = validation_file
        self.sequence_length = sequence_length
        self.config = config

        self._train_dataset: PackedTokenDataset | None = None
        self._validation_dataset: PackedTokenDataset | None = None

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

    def train_dataloader(self) -> DataLoader:
        """Create the training DataLoader."""

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
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


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)