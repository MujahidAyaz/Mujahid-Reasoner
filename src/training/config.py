from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    train_file: str
    validation_file: str


@dataclass(frozen=True)
class TokenizerConfig:
    path: str


@dataclass(frozen=True)
class SequenceConfig:
    length: int


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int
    shuffle: bool
    num_workers: int
    pin_memory: bool
    drop_last: bool


@dataclass(frozen=True)
class TrainingConfig:
    device: str
    max_epochs: int
    max_steps: int | None

    gradient_accumulation_steps: int
    max_grad_norm: float

    learning_rate: float
    weight_decay: float

    beta1: float
    beta2: float
    optimizer_eps: float

    warmup_steps: int
    min_lr_ratio: float

    log_every_steps: int
    eval_every_steps: int
    checkpoint_every_steps: int

    max_eval_batches: int | None

    seed: int


@dataclass(frozen=True)
class OutputConfig:
    directory: str


@dataclass(frozen=True)
class CheckpointConfig:
    resume: bool
    filename: str


@dataclass(frozen=True)
class FullTrainingConfig:
    data: DataConfig
    tokenizer: TokenizerConfig
    sequence: SequenceConfig
    dataloader: DataLoaderConfig
    training: TrainingConfig
    output: OutputConfig
    checkpoint: CheckpointConfig


def _require(
    mapping: dict[str, Any],
    key: str,
    section: str,
) -> Any:
    if key not in mapping:
        raise ValueError(
            f"Missing required key '{key}' "
            f"in section '{section}'."
        )

    return mapping[key]


def load_training_config(
    config_path: str | Path,
) -> FullTrainingConfig:
    """
    Load and validate the complete training configuration.
    """

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Training configuration not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError(
            "Training configuration must contain a YAML mapping."
        )

    data = raw.get("data", {})
    tokenizer = raw.get("tokenizer", {})
    sequence = raw.get("sequence", {})
    dataloader = raw.get("dataloader", {})
    training = raw.get("training", {})
    output = raw.get("output", {})
    checkpoint = raw.get("checkpoint", {})

    if not isinstance(data, dict):
        raise ValueError("'data' must be a mapping.")

    if not isinstance(tokenizer, dict):
        raise ValueError(
            "'tokenizer' must be a mapping."
        )

    if not isinstance(sequence, dict):
        raise ValueError(
            "'sequence' must be a mapping."
        )

    if not isinstance(dataloader, dict):
        raise ValueError(
            "'dataloader' must be a mapping."
        )

    if not isinstance(training, dict):
        raise ValueError(
            "'training' must be a mapping."
        )

    if not isinstance(output, dict):
        raise ValueError(
            "'output' must be a mapping."
        )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "'checkpoint' must be a mapping."
        )

    config = FullTrainingConfig(
        data=DataConfig(
            train_file=_require(
                data,
                "train_file",
                "data",
            ),
            validation_file=_require(
                data,
                "validation_file",
                "data",
            ),
        ),
        tokenizer=TokenizerConfig(
            path=_require(
                tokenizer,
                "path",
                "tokenizer",
            ),
        ),
        sequence=SequenceConfig(
            length=_require(
                sequence,
                "length",
                "sequence",
            ),
        ),
        dataloader=DataLoaderConfig(
            batch_size=_require(
                dataloader,
                "batch_size",
                "dataloader",
            ),
            shuffle=_require(
                dataloader,
                "shuffle",
                "dataloader",
            ),
            num_workers=_require(
                dataloader,
                "num_workers",
                "dataloader",
            ),
            pin_memory=_require(
                dataloader,
                "pin_memory",
                "dataloader",
            ),
            drop_last=_require(
                dataloader,
                "drop_last",
                "dataloader",
            ),
        ),
        training=TrainingConfig(
            device=_require(
                training,
                "device",
                "training",
            ),
            max_epochs=_require(
                training,
                "max_epochs",
                "training",
            ),
            max_steps=training.get(
                "max_steps"
            ),
            gradient_accumulation_steps=_require(
                training,
                "gradient_accumulation_steps",
                "training",
            ),
            max_grad_norm=_require(
                training,
                "max_grad_norm",
                "training",
            ),
            learning_rate=_require(
                training,
                "learning_rate",
                "training",
            ),
            weight_decay=_require(
                training,
                "weight_decay",
                "training",
            ),
            beta1=_require(
                training,
                "beta1",
                "training",
            ),
            beta2=_require(
                training,
                "beta2",
                "training",
            ),
            optimizer_eps=_require(
                training,
                "optimizer_eps",
                "training",
            ),
            warmup_steps=_require(
                training,
                "warmup_steps",
                "training",
            ),
            min_lr_ratio=_require(
                training,
                "min_lr_ratio",
                "training",
            ),
            log_every_steps=_require(
                training,
                "log_every_steps",
                "training",
            ),
            eval_every_steps=_require(
                training,
                "eval_every_steps",
                "training",
            ),
            checkpoint_every_steps=_require(
                training,
                "checkpoint_every_steps",
                "training",
            ),
            max_eval_batches=training.get(
                "max_eval_batches"
            ),
            seed=_require(
                training,
                "seed",
                "training",
            ),
        ),
        output=OutputConfig(
            directory=_require(
                output,
                "directory",
                "output",
            ),
        ),
        checkpoint=CheckpointConfig(
            resume=_require(
                checkpoint,
                "resume",
                "checkpoint",
            ),
            filename=_require(
                checkpoint,
                "filename",
                "checkpoint",
            ),
        ),
    )

    _validate_config(config)

    return config


def _validate_config(
    config: FullTrainingConfig,
) -> None:
    if config.sequence.length <= 0:
        raise ValueError(
            "sequence.length must be > 0."
        )

    loader = config.dataloader

    if loader.batch_size <= 0:
        raise ValueError(
            "dataloader.batch_size must be > 0."
        )

    if loader.num_workers < 0:
        raise ValueError(
            "dataloader.num_workers must be >= 0."
        )

    training = config.training

    if training.max_epochs <= 0:
        raise ValueError(
            "training.max_epochs must be > 0."
        )

    if (
        training.max_steps is not None
        and training.max_steps <= 0
    ):
        raise ValueError(
            "training.max_steps must be > 0."
        )

    if training.learning_rate <= 0:
        raise ValueError(
            "training.learning_rate must be > 0."
        )

    if training.weight_decay < 0:
        raise ValueError(
            "training.weight_decay must be >= 0."
        )

    if not 0 <= training.beta1 < 1:
        raise ValueError(
            "training.beta1 must be in [0, 1)."
        )

    if not 0 <= training.beta2 < 1:
        raise ValueError(
            "training.beta2 must be in [0, 1)."
        )

    if training.optimizer_eps <= 0:
        raise ValueError(
            "training.optimizer_eps must be > 0."
        )

    if training.warmup_steps < 0:
        raise ValueError(
            "training.warmup_steps must be >= 0."
        )

    if not 0 <= training.min_lr_ratio <= 1:
        raise ValueError(
            "training.min_lr_ratio must be in [0, 1]."
        )

    if training.seed < 0:
        raise ValueError(
            "training.seed must be >= 0."
        )