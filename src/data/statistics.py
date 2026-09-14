from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetStatistics:
    """Statistics collected while processing a text corpus."""

    documents_seen: int = 0
    documents_kept: int = 0
    documents_rejected: int = 0
    duplicate_documents: int = 0
    total_characters: int = 0

    @property
    def average_characters(self) -> float:
        """Average number of characters per retained document."""

        if self.documents_kept == 0:
            return 0.0

        return self.total_characters / self.documents_kept

    def record_seen(self) -> None:
        self.documents_seen += 1

    def record_rejected(self) -> None:
        self.documents_rejected += 1

    def record_duplicate(self) -> None:
        self.duplicate_documents += 1

    def record_kept(self, text: str) -> None:
        self.documents_kept += 1
        self.total_characters += len(text)

    def summary(self) -> dict[str, int | float]:
        """Return statistics as a serializable dictionary."""

        return {
            "documents_seen": self.documents_seen,
            "documents_kept": self.documents_kept,
            "documents_rejected": self.documents_rejected,
            "duplicate_documents": self.duplicate_documents,
            "total_characters": self.total_characters,
            "average_characters": round(self.average_characters, 2),
        }