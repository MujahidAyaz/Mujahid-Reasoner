from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


class ExactDeduplicator:
    """
    Detect exact duplicate documents using SHA-256 fingerprints.

    Fingerprints are stored in a SQLite database instead of an in-memory
    Python set so deduplication remains scalable for large corpora.
    """

    COMMIT_INTERVAL = 1_000

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._connection = sqlite3.connect(
            self.database_path,
        )

        self._connection.execute(
            "PRAGMA journal_mode=WAL"
        )

        self._connection.execute(
            "PRAGMA synchronous=NORMAL"
        )

        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS fingerprints (
                fingerprint BLOB PRIMARY KEY
            )
            """
        )

        self._connection.commit()

        self._pending_operations = 0

    @staticmethod
    def fingerprint(text: str) -> bytes:
        """Generate a deterministic SHA-256 fingerprint."""

        return hashlib.sha256(
            text.encode("utf-8")
        ).digest()

    def is_duplicate(self, text: str) -> bool:
        """
        Return True if the document has already been seen.

        Otherwise register its fingerprint and return False.
        """

        fingerprint = self.fingerprint(text)

        cursor = self._connection.execute(
            """
            INSERT OR IGNORE INTO fingerprints (fingerprint)
            VALUES (?)
            """,
            (fingerprint,),
        )

        self._pending_operations += 1

        if (
            self._pending_operations
            >= self.COMMIT_INTERVAL
        ):
            self._connection.commit()
            self._pending_operations = 0

        return cursor.rowcount == 0

    @property
    def unique_documents(self) -> int:
        """Return the number of unique documents stored."""

        self._connection.commit()
        self._pending_operations = 0

        cursor = self._connection.execute(
            "SELECT COUNT(*) FROM fingerprints"
        )

        result = cursor.fetchone()

        return int(result[0]) if result is not None else 0

    def close(self) -> None:
        """Commit pending operations and close the database."""

        if self._pending_operations > 0:
            self._connection.commit()
            self._pending_operations = 0

        self._connection.close()

    def __enter__(self) -> "ExactDeduplicator":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()