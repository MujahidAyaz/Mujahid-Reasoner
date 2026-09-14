from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for document-level quality filters."""

    min_characters: int = 200
    max_characters: int = 100_000


class DocumentFilter:
    """
    Filter documents using simple, deterministic quality constraints.

    More advanced quality filtering can be added later without changing
    the pipeline interface.
    """

    def __init__(self, config: FilterConfig) -> None:
        self.config = config

    def is_valid(self, text: str) -> bool:
        """Return True when a document passes all configured filters."""

        if not text:
            return False

        length = len(text)

        if length < self.config.min_characters:
            return False

        if length > self.config.max_characters:
            return False

        return True