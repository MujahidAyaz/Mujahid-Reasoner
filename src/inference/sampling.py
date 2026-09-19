from __future__ import annotations

import torch
from torch import Tensor

from src.inference.config import GenerationConfig


def apply_repetition_penalty(
    logits: Tensor,
    generated_ids: list[int],
    penalty: float,
) -> Tensor:
    if penalty == 1.0 or not generated_ids:
        return logits

    logits = logits.clone()

    for token_id in set(generated_ids):
        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty

    return logits


def apply_frequency_presence_penalties(
    logits: Tensor,
    generated_ids: list[int],
    frequency_penalty: float,
    presence_penalty: float,
) -> Tensor:
    if not generated_ids:
        return logits

    if frequency_penalty == 0.0 and presence_penalty == 0.0:
        return logits

    logits = logits.clone()

    counts = torch.bincount(
        torch.tensor(
            generated_ids,
            dtype=torch.long,
            device=logits.device,
        ),
        minlength=logits.size(-1),
    ).to(logits.dtype)

    penalty = (
        frequency_penalty * counts
        + presence_penalty * (counts > 0).to(logits.dtype)
    )

    logits -= penalty.unsqueeze(0)

    return logits


def sample_token(
    logits: Tensor,
    config: GenerationConfig,
    generated_ids: list[int] | None = None,
    forbidden_token_ids: tuple[int, ...] = (),
) -> Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocab_size].")

    if forbidden_token_ids:
        logits = logits.clone()

        for token_id in forbidden_token_ids:
            if 0 <= token_id < logits.size(-1):
                logits[:, token_id] = float("-inf")

    if not config.do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / config.temperature

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

    if config.top_k > 0:
        k = min(config.top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        threshold = values[:, -1].unsqueeze(-1)
        logits = torch.where(
            logits < threshold,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    if config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
            dim=-1,
        )
        cumulative_probs = torch.softmax(
            sorted_logits,
            dim=-1,
        ).cumsum(dim=-1)

        sorted_mask = cumulative_probs > config.top_p
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = False

        sorted_logits = sorted_logits.masked_fill(
            sorted_mask,
            float("-inf"),
        )

        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(1, sorted_indices, sorted_logits)

    probabilities = torch.softmax(logits, dim=-1)

    return torch.multinomial(probabilities, num_samples=1)
