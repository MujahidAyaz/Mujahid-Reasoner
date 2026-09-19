from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class BatchGenerationInput:
    prompts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.prompts:
            raise ValueError("prompts must not be empty.")

        if not all(isinstance(prompt, str) for prompt in self.prompts):
            raise ValueError("all prompts must be strings.")

        if not all(prompt.strip() for prompt in self.prompts):
            raise ValueError("all prompts must be non-empty.")


@dataclass(frozen=True)
class BatchGenerationOutput:
    texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.texts:
            raise ValueError("texts must not be empty.")


@dataclass(frozen=True)
class TokenizedBatch:
    input_ids: Tensor
    attention_mask: Tensor
    prompt_lengths: tuple[int, ...]


def tokenize_batch(
    tokenizer,
    prompts: tuple[str, ...],
    pad_token_id: int,
    device: torch.device,
) -> TokenizedBatch:
    if not prompts:
        raise ValueError("prompts must not be empty.")

    encoded = [
        tokenizer.encode(prompt).ids
        for prompt in prompts
    ]

    if any(not ids for ids in encoded):
        raise ValueError(
            "all prompts must tokenize to at least one token."
        )

    max_length = max(len(ids) for ids in encoded)

    input_ids = torch.tensor(
        [
            ids + [pad_token_id] * (max_length - len(ids))
            for ids in encoded
        ],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.tensor(
        [
            [True] * len(ids) + [False] * (max_length - len(ids))
            for ids in encoded
        ],
        dtype=torch.bool,
        device=device,
    )

    prompt_lengths = tuple(len(ids) for ids in encoded)

    return TokenizedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lengths=prompt_lengths,
    )

def group_prompt_indices_by_length(
    tokenized_prompts: list[list[int]],
) -> dict[int, list[int]]:
    """
    Group prompt indices by tokenized sequence length.

    Prompts with equal lengths can safely share the current
    batched KV-cache decoding path.
    """
    groups: dict[int, list[int]] = {}

    for index, token_ids in enumerate(tokenized_prompts):
        length = len(token_ids)

        if length == 0:
            raise ValueError(
                "all prompts must tokenize to at least one token"
            )

        groups.setdefault(length, []).append(index)

    return groups