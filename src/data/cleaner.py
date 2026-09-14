from __future__ import annotations

import re


class TextCleaner:
    """Normalize raw text before quality filtering."""

    def __init__(
        self,
        *,
        remove_null_bytes: bool = True,
        normalize_whitespace: bool = True,
    ) -> None:
        self.remove_null_bytes = remove_null_bytes
        self.normalize_whitespace = normalize_whitespace

    def clean(self, text: str) -> str:
        """Clean a single document and return normalized text."""

        if not isinstance(text, str):
            return ""

        if self.remove_null_bytes:
            text = text.replace("\x00", "")

        if self.normalize_whitespace:
            lines = [line.strip() for line in text.splitlines()]
            text = "\n".join(lines)
            text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()