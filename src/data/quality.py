from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class QualityConfig:
    """Configuration for advanced document-quality checks."""

    max_replacement_character_ratio: float = 0.005
    max_suspicious_encoding_ratio: float = 0.01
    max_repeated_character_ratio: float = 0.05
    
    max_symbol_ratio: float = 0.30


@dataclass(frozen=True)
class QualityResult:
    """Result of advanced text-quality evaluation."""

    is_valid: bool
    reason: str


class TextQualityAnalyzer:
    """
    Detect common forms of corrupted, malformed, or low-quality text.

    This analyzer is intentionally deterministic and lightweight so it can
    operate efficiently during large-scale corpus preprocessing.
    """

    WORD_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)

    SUSPICIOUS_SEQUENCES = (
        "â€",
        "â€™",
        "â€œ",
        "â€“",
        "â€”",
        "Ã©",
        "Ã¨",
        "Ã¡",
        "Ã³",
        "Â ",
        "ï¿½",
    )

    def __init__(self, config: QualityConfig) -> None:
        self.config = config

    def evaluate(self, text: str) -> QualityResult:
        """Evaluate a document and return the first detected issue."""

        if not text:
            return QualityResult(False, "empty")

        if self._replacement_character_ratio(text) > (
            self.config.max_replacement_character_ratio
        ):
            return QualityResult(False, "encoding_corruption")

        if self._suspicious_encoding_ratio(text) > (
            self.config.max_suspicious_encoding_ratio
        ):
            return QualityResult(False, "suspicious_encoding")

        if self._repeated_character_ratio(text) > (
            self.config.max_repeated_character_ratio
        ):
            return QualityResult(False, "repeated_characters")

        

        if self._symbol_ratio(text) > self.config.max_symbol_ratio:
            return QualityResult(False, "too_many_symbols")

        return QualityResult(True, "accepted")

    def is_valid(self, text: str) -> bool:
        """Return whether the document passes quality checks."""

        return self.evaluate(text).is_valid

    @staticmethod
    def _replacement_character_ratio(text: str) -> float:
        """Measure Unicode replacement characters such as �."""

        if not text:
            return 0.0

        count = text.count("\ufffd")
        return count / len(text)

    def _suspicious_encoding_ratio(self, text: str) -> float:
        """
        Estimate whether text contains common UTF-8/Windows-1252 corruption.

        Examples:
            â€™
            â€œ
            â€“
            Ã©
        """

        if not text:
            return 0.0

        suspicious_count = sum(
            text.count(sequence)
            for sequence in self.SUSPICIOUS_SEQUENCES
        )

        return suspicious_count / len(text)

    @staticmethod
    def _repeated_character_ratio(text: str) -> float:
        """
        Measure excessive runs of the same character.

        Example:
            "!!!!!!!!!!!!!!!!!!!!"
            "aaaaaaaaaaaaaaaaaaaa"
        """

        if not text:
            return 0.0

        repeated_characters = 0

        for index in range(1, len(text)):
            if text[index] == text[index - 1]:
                repeated_characters += 1

        return repeated_characters / len(text)

    def _repeated_word_ratio(self, text: str) -> float:
        """
        Detect pathological repetition of the same word.

        Normal language reuses common words frequently, so this check only
        measures excessive repetition of a single word.
        """

        words = [
            word.lower()
            for word in self.WORD_PATTERN.findall(text)
        ]

        if len(words) < 20:
            return 0.0

        word_counts: dict[str, int] = {}

        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        most_common_count = max(word_counts.values())

        return most_common_count / len(words)

    @staticmethod
    def _symbol_ratio(text: str) -> float:
        """Measure the proportion of punctuation/symbol characters."""

        if not text:
            return 0.0

        symbol_count = 0

        for character in text:
            if character.isspace():
                continue

            category = unicodedata.category(character)

            if category.startswith(("P", "S")):
                symbol_count += 1

        non_whitespace = sum(
            1 for character in text if not character.isspace()
        )

        if non_whitespace == 0:
            return 0.0

        return symbol_count / non_whitespace