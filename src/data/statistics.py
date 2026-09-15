from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class DatasetStatistics:
    """Statistics collected during corpus processing."""

    documents_seen: int = 0
    documents_kept: int = 0
    documents_rejected: int = 0
    duplicate_documents: int = 0

    total_characters: int = 0

    min_characters: int | None = None
    max_characters: int = 0

    rejection_reasons: Counter[str] = field(
        default_factory=Counter
    )

    @property
    def average_characters(self) -> float:
        """Average characters per retained document."""

        if self.documents_kept == 0:
            return 0.0

        return self.total_characters / self.documents_kept

    @property
    def rejection_rate(self) -> float:
        """Percentage of documents rejected."""

        if self.documents_seen == 0:
            return 0.0

        return (
            self.documents_rejected
            / self.documents_seen
            * 100
        )

    def record_seen(self) -> None:
        """Record a document entering the pipeline."""

        self.documents_seen += 1

    def record_rejected(self, reason: str = "unknown") -> None:
        """Record a rejected document and its reason."""

        self.documents_rejected += 1
        self.rejection_reasons[reason] += 1

    def record_duplicate(self) -> None:
        """Record an exact duplicate document."""

        self.duplicate_documents += 1
        self.rejection_reasons["duplicate"] += 1

    def record_kept(self, text: str) -> None:
        """Record an accepted document."""

        character_count = len(text)

        self.documents_kept += 1
        self.total_characters += character_count

        if self.min_characters is None:
            self.min_characters = character_count
        else:
            self.min_characters = min(
                self.min_characters,
                character_count,
            )

        self.max_characters = max(
            self.max_characters,
            character_count,
        )

    def summary(self) -> dict:
        """Return all statistics as a serializable dictionary."""

        return {
            "documents_seen": self.documents_seen,
            "documents_kept": self.documents_kept,
            "documents_rejected": self.documents_rejected,
            "duplicate_documents": self.duplicate_documents,
            "rejection_rate_percent": round(
                self.rejection_rate,
                2,
            ),
            "total_characters": self.total_characters,
            "min_characters": self.min_characters or 0,
            "max_characters": self.max_characters,
            "average_characters": round(
                self.average_characters,
                2,
            ),
            "rejection_reasons": dict(
                self.rejection_reasons
            ),
        }