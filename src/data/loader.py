from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration required to load a Hugging Face dataset."""

    name: str
    config: str
    split: str = "train"
    streaming: bool = True


class DatasetLoader:
    """
    Load datasets for the Mujahid-Reasoner data pipeline.

    The loader is intentionally responsible only for dataset access.
    Cleaning, filtering, deduplication, and splitting belong to later stages.
    """

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self._dataset: Any | None = None

    def load(self) -> Any:
        """Load and cache the configured dataset."""

        if self._dataset is None:
            self._dataset = load_dataset(
                path=self.config.name,
                name=self.config.config,
                split=self.config.split,
                streaming=self.config.streaming,
            )

        return self._dataset

    def stream(self) -> Iterator[dict[str, Any]]:
        """
        Stream documents one at a time.

        Yields:
            Dataset examples as dictionaries.
        """

        dataset = self.load()

        for example in dataset:
            yield example