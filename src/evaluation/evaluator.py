from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.model.model import MujahidReasonerModel


@dataclass(frozen=True)
class EvaluationResult:
    """Results produced by model evaluation."""

    loss: float
    perplexity: float
    tokens: int
    batches: int


class ModelEvaluator:
    """Evaluate a causal language model on a validation DataLoader."""

    def __init__(
        self,
        model: MujahidReasonerModel,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)

        self.model.eval()

    @torch.inference_mode()
    def evaluate(
        self,
        dataloader: DataLoader[Any],
    ) -> EvaluationResult:
        total_loss = 0.0
        total_tokens = 0
        total_batches = 0

        for batch in dataloader:
            input_ids, target_ids = self._unpack_batch(
                batch
            )

            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            model_output = self.model(input_ids)
            logits = (
                model_output[0]
                if isinstance(model_output, tuple)
                else model_output
            )

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )

            token_count = target_ids.numel()

            total_loss += loss.item() * token_count
            total_tokens += token_count
            total_batches += 1

        if total_tokens == 0:
            raise RuntimeError(
                "No evaluation tokens were processed."
            )

        average_loss = total_loss / total_tokens

        perplexity = exp(average_loss)

        return EvaluationResult(
            loss=average_loss,
            perplexity=perplexity,
            tokens=total_tokens,
            batches=total_batches,
        )

    @staticmethod
    def _unpack_batch(
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract input and target tensors from a batch."""

        if isinstance(batch, dict):
            if "input_ids" not in batch:
                raise KeyError(
                    "Batch dictionary must contain "
                    "'input_ids'."
                )

            if "target_ids" in batch:
                return (
                    batch["input_ids"],
                    batch["target_ids"],
                )

            if "labels" in batch:
                return (
                    batch["input_ids"],
                    batch["labels"],
                )

            raise KeyError(
                "Batch dictionary must contain "
                "'target_ids' or 'labels'."
            )

        if isinstance(batch, (tuple, list)):
            if len(batch) < 2:
                raise ValueError(
                    "Batch must contain input and target tensors."
                )

            return batch[0], batch[1]

        raise TypeError(
            "Unsupported batch format. Expected a dictionary "
            "or tuple/list containing input and target tensors."
        )