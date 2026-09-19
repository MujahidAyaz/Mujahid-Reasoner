from __future__ import annotations

import torch
from torch import Tensor

from src.inference.config import GenerationConfig


def apply_repetition_penalty(
    logits: Tensor,
    generated_ids: list[int] | list[list[int]],
    penalty: float,
) -> Tensor:
    if penalty == 1.0:
        return logits

    logits = logits.clone()

    if not generated_ids:
        return logits

    if isinstance(generated_ids[0], int):
        histories = [generated_ids]
    else:
        histories = generated_ids

    if logits.size(0) != len(histories):
        raise ValueError(
            "Number of histories must match batch size."
        )

    for batch_index, history in enumerate(histories):
        for token_id in set(history):
            if not 0 <= token_id < logits.size(-1):
                continue

            if logits[batch_index, token_id] > 0:
                logits[batch_index, token_id] /= penalty
            else:
                logits[batch_index, token_id] *= penalty

    return logits


def apply_frequency_presence_penalties(
    logits: Tensor,
    generated_ids: list[int] | list[list[int]],
    frequency_penalty: float,
    presence_penalty: float,
) -> Tensor:
    if not generated_ids:
        return logits

    if (
        frequency_penalty == 0.0
        and presence_penalty == 0.0
    ):
        return logits

    if isinstance(generated_ids[0], int):
        histories = [generated_ids]
    else:
        histories = generated_ids

    if logits.size(0) != len(histories):
        raise ValueError(
            "Number of histories must match batch size."
        )

    logits = logits.clone()

    for batch_index, history in enumerate(histories):
        if not history:
            continue

        counts = torch.bincount(
            torch.tensor(
                history,
                dtype=torch.long,
                device=logits.device,
            ),
            minlength=logits.size(-1),
        ).to(logits.dtype)

        penalty = (
            frequency_penalty * counts
            + presence_penalty * (counts > 0).to(logits.dtype)
        )

        logits[batch_index] -= penalty

    return logits


def apply_top_k(
    logits: Tensor,
    top_k: int,
) -> Tensor:
    if top_k <= 0:
        return logits

    k = min(top_k, logits.size(-1))

    values, _ = torch.topk(
        logits,
        k=k,
        dim=-1,
    )

    threshold = values[:, -1].unsqueeze(-1)

    return torch.where(
        logits < threshold,
        torch.full_like(logits, float("-inf")),
        logits,
    )


def apply_top_p(
    logits: Tensor,
    top_p: float,
) -> Tensor:
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
        dim=-1,
    )

    cumulative_probs = torch.softmax(
        sorted_logits,
        dim=-1,
    ).cumsum(dim=-1)

    sorted_mask = cumulative_probs > top_p

    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
    sorted_mask[:, 0] = False

    sorted_logits = sorted_logits.masked_fill(
        sorted_mask,
        float("-inf"),
    )

    filtered_logits = torch.full_like(
        logits,
        float("-inf"),
    )

    filtered_logits.scatter_(
        1,
        sorted_indices,
        sorted_logits,
    )

    return filtered_logits


def sample_token(
    logits: Tensor,
    config: GenerationConfig,
    generated_ids: list[int] | list[list[int]] | None = None,
    forbidden_token_ids: tuple[int, ...] = (),
) -> Tensor:
    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape [batch, vocab_size]."
        )

    logits = logits.clone()

    if forbidden_token_ids:
        for token_id in forbidden_token_ids:
            if 0 <= token_id < logits.size(-1):
                logits[:, token_id] = float("-inf")

    if generated_ids:
        logits = apply_repetition_penalty(
            logits,
            generated_ids,
            config.repetition_penalty,
        )

        logits = apply_frequency_presence_penalties(
            logits,
            generated_ids,
            config.frequency_penalty,
            config.presence_penalty,
        )

    if not config.do_sample:
        return torch.argmax(
            logits,
            dim=-1,
        ).unsqueeze(-1)

    logits = logits / config.temperature

    logits = apply_top_k(
        logits,
        config.top_k,
    )

    logits = apply_top_p(
        logits,
        config.top_p,
    )

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    return torch.multinomial(
        probabilities,
        num_samples=1,
    )