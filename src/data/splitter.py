from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for train/validation splitting."""

    validation_ratio: float = 0.01
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_ratio < 1.0:
            raise ValueError("validation_ratio must be between 0 and 1.")


class DatasetSplitter:
    """Create deterministic train/validation splits."""

    def __init__(self, config: SplitConfig) -> None:
        self.config = config

    def split(
        self,
        documents: Iterable[T],
    ) -> tuple[list[T], list[T]]:
        """
        Split documents into training and validation sets.

        The same seed produces the same split.
        """

        items = list(documents)

        rng = random.Random(self.config.seed)
        rng.shuffle(items)

        validation_size = max(
            1,
            int(len(items) * self.config.validation_ratio),
        )

        validation = items[:validation_size]
        train = items[validation_size:]

        return train, validation