from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenizers import Tokenizer
from torch import Tensor

from src.model.model import MujahidReasonerModel


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9


class TextGenerator:
    """Autoregressive text generator for Mujahid-Reasoner."""

    def __init__(
        self,
        model: MujahidReasonerModel,
        tokenizer: Tokenizer,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = torch.device(device)

        self.model.eval()

        self.bos_token_id = tokenizer.token_to_id("<bos>")
        self.eos_token_id = tokenizer.token_to_id("<eos>")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> str:
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        config = config or GenerationConfig()

        self._validate_config(config)

        encoded = self.tokenizer.encode(prompt)

        input_ids = torch.tensor(
            [encoded.ids],
            dtype=torch.long,
            device=self.device,
        )

        generated_ids = input_ids.clone()

        for _ in range(config.max_new_tokens):
            context = generated_ids[
                :, -self.model.config.max_sequence_length :
            ]

            logits = self.model(context)

            next_token_logits = logits[:, -1, :]

            next_token_logits = (
                next_token_logits
                / config.temperature
            )

            next_token_logits = self._apply_top_k(
                next_token_logits,
                config.top_k,
            )

            next_token_logits = self._apply_top_p(
                next_token_logits,
                config.top_p,
            )

            probabilities = torch.softmax(
                next_token_logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            generated_ids = torch.cat(
                [generated_ids, next_token],
                dim=1,
            )

            if (
                self.eos_token_id is not None
                and next_token.item()
                == self.eos_token_id
            ):
                break

        return self.tokenizer.decode(
            generated_ids[0].tolist(),
            skip_special_tokens=True,
        )

    @staticmethod
    def _apply_top_k(
        logits: Tensor,
        top_k: int,
    ) -> Tensor:
        if top_k <= 0:
            return logits

        top_k = min(top_k, logits.shape[-1])

        values, _ = torch.topk(
            logits,
            top_k,
            dim=-1,
        )

        threshold = values[:, -1].unsqueeze(-1)

        return logits.masked_fill(
            logits < threshold,
            float("-inf"),
        )

    @staticmethod
    def _apply_top_p(
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

        sorted_probabilities = torch.softmax(
            sorted_logits,
            dim=-1,
        )

        cumulative_probabilities = torch.cumsum(
            sorted_probabilities,
            dim=-1,
        )

        remove_mask = (
            cumulative_probabilities > top_p
        )

        remove_mask[:, 1:] = remove_mask[:, :-1].clone()
        remove_mask[:, 0] = False

        sorted_logits = sorted_logits.masked_fill(
            remove_mask,
            float("-inf"),
        )

        filtered_logits = torch.full_like(
            logits,
            float("-inf"),
        )

        filtered_logits.scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_logits,
        )

        return filtered_logits

    @staticmethod
    def _validate_config(
        config: GenerationConfig,
    ) -> None:
        if config.max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be positive."
            )

        if config.temperature <= 0:
            raise ValueError(
                "temperature must be positive."
            )

        if config.top_k < 0:
            raise ValueError(
                "top_k must be non-negative."
            )

        if not 0 < config.top_p <= 1:
            raise ValueError(
                "top_p must be in the range (0, 1]."
            )