from __future__ import annotations

import hashlib


class ExactDeduplicator:
    """
    Detect exact duplicate documents using SHA-256 fingerprints.

    Only the fingerprint is stored, not the document itself.
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()

    @staticmethod
    def fingerprint(text: str) -> str:
        """Generate a deterministic SHA-256 fingerprint."""

        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        """
        Return True if the document has already been seen.

        Otherwise register it and return False.
        """

        fingerprint = self.fingerprint(text)

        if fingerprint in self._seen:
            return True

        self._seen.add(fingerprint)
        return False

    @property
    def unique_documents(self) -> int:
        """Return the number of unique documents seen."""

        return len(self._seen)