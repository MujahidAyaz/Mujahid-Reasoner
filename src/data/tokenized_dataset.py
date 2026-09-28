from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset[tuple[Tensor, Tensor]]):
    """
    Memory-mapped dataset for packed causal-language-model training.

    Tokens are stored as uint16 because the tokenizer vocabulary fits
    within the uint16 range. The memory-mapped representation keeps the
    complete corpus out of RAM.
    """

    TOKEN_DTYPE = np.uint16

    def __init__(
        self,
        token_file: Path,
        sequence_length: int,
    ) -> None:
        if sequence_length < 2:
            raise ValueError(
                "sequence_length must be at least 2."
            )

        if not token_file.exists():
            raise FileNotFoundError(
                f"Token file not found: {token_file}"
            )

        self.token_file = token_file
        self.sequence_length = sequence_length

        file_size = token_file.stat().st_size
        dtype_size = np.dtype(
            self.TOKEN_DTYPE
        ).itemsize

        if file_size == 0:
            raise ValueError(
                f"Token file is empty: {token_file}"
            )

        if file_size % dtype_size != 0:
            raise ValueError(
                "Token file size is not aligned to "
                f"{self.TOKEN_DTYPE}."
            )

        self.total_tokens = (
            file_size // dtype_size
        )

        self.sequence_count = (
            self.total_tokens
            // self.sequence_length
        )

        if self.sequence_count == 0:
            raise ValueError(
                "Token file does not contain enough "
                "tokens for one sequence."
            )

        self.tokens = np.memmap(
            token_file,
            dtype=self.TOKEN_DTYPE,
            mode="r",
        )

    def __len__(self) -> int:
        """Return the number of complete training sequences."""

        return self.sequence_count

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor]:
        """Return one causal-LM input/target pair."""

        if index < 0:
            index += self.sequence_count

        if not 0 <= index < self.sequence_count:
            raise IndexError(
                f"Dataset index out of range: {index}"
            )

        start = (
            index * self.sequence_length
        )

        end = start + self.sequence_length + 1

        chunk = self.tokens[start:end]

        input_ids = torch.from_numpy(
            chunk[:-1].astype(
                np.int64,
                copy=False,
            )
        )

        target_ids = torch.from_numpy(
            chunk[1:].astype(
                np.int64,
                copy=False,
            )
        )

        return input_ids, target_ids

    def close(self) -> None:
        """Release the memory-mapped token file."""

        mmap = getattr(
            self.tokens,
            "_mmap",
            None,
        )

        if mmap is not None:
            mmap.close()

    def __del__(self) -> None:
        """Best-effort cleanup of the memory map."""

        try:
            self.close()
        except Exception:
            pass


def load_token_manifest(
    manifest_path: Path,
) -> dict:
    """Load metadata generated during token preparation."""

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Token manifest not found: {manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    if not isinstance(manifest, dict):
        raise ValueError(
            "Token manifest must contain a JSON object."
        )

    return manifest