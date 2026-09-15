from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for document-level quality filtering."""

    min_characters: int = 200
    max_characters: int = 100_000

    max_url_ratio: float = 0.02
    max_email_count: int = 3
    max_repeated_line_ratio: float = 0.30
    max_non_printable_ratio: float = 0.02

    min_alpha_ratio: float = 0.20
    min_word_count: int = 40


@dataclass(frozen=True)
class FilterResult:
    """Result of evaluating a document."""

    is_valid: bool
    reason: str


class DocumentFilter:
    """
    Deterministic document-quality filter.

    The filter intentionally uses lightweight signals that are cheap to
    evaluate during large-scale corpus preprocessing.
    """

    URL_PATTERN = re.compile(
        r"(https?://|www\.)\S+",
        re.IGNORECASE,
    )

    EMAIL_PATTERN = re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        re.IGNORECASE,
    )

    WORD_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)

    def __init__(self, config: FilterConfig) -> None:
        self.config = config

    def evaluate(self, text: str) -> FilterResult:
        """Evaluate a document and return a decision with a reason."""

        if not text:
            return FilterResult(False, "empty")

        length = len(text)

        if length < self.config.min_characters:
            return FilterResult(False, "too_short")

        if length > self.config.max_characters:
            return FilterResult(False, "too_long")

        if self._non_printable_ratio(text) > self.config.max_non_printable_ratio:
            return FilterResult(False, "too_many_non_printable")

        if self._alpha_ratio(text) < self.config.min_alpha_ratio:
            return FilterResult(False, "low_alpha_ratio")

        words = self.WORD_PATTERN.findall(text)

        if len(words) < self.config.min_word_count:
            return FilterResult(False, "too_few_words")

        if self._url_ratio(text) > self.config.max_url_ratio:
            return FilterResult(False, "too_many_urls")

        if len(self.EMAIL_PATTERN.findall(text)) > self.config.max_email_count:
            return FilterResult(False, "too_many_emails")

        if self._repeated_line_ratio(text) > self.config.max_repeated_line_ratio:
            return FilterResult(False, "repeated_lines")

        return FilterResult(True, "accepted")

    def is_valid(self, text: str) -> bool:
        """Return True when a document passes all filters."""

        return self.evaluate(text).is_valid

    @staticmethod
    def _alpha_ratio(text: str) -> float:
        """Calculate the ratio of alphabetic characters."""

        if not text:
            return 0.0

        alpha_count = sum(character.isalpha() for character in text)

        return alpha_count / len(text)

    @staticmethod
    def _non_printable_ratio(text: str) -> float:
        """Calculate the ratio of non-printable characters."""

        if not text:
            return 0.0

        non_printable = 0

        for character in text:
            if character in "\n\t\r":
                continue

            if not character.isprintable():
                non_printable += 1

        return non_printable / len(text)

    def _url_ratio(self, text: str) -> float:
        """Calculate the proportion of text occupied by URLs."""

        if not text:
            return 0.0

        urls = self.URL_PATTERN.findall(text)

        if not urls:
            return 0.0

        url_characters = sum(len(url) for url in urls)

        return url_characters / len(text)

    @staticmethod
    def _repeated_line_ratio(text: str) -> float:
        """Calculate the ratio of duplicate non-empty lines."""

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if len(lines) < 2:
            return 0.0

        unique_lines = set(lines)

        repeated_count = len(lines) - len(unique_lines)

        return repeated_count / len(lines)