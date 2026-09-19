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
    persistent_workers: bool = False
    prefetch_factor: int = 2


@dataclass(frozen=True)
class TrainingConfig:
    device: str

    # Numerical precision.
    precision: str

    # Optional PyTorch graph compilation.
    compile: bool

    # Memory/performance optimizations.
    gradient_checkpointing: bool
    allow_tf32: bool
    cudnn_benchmark: bool

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


def _optional_bool(
    mapping: dict[str, Any],
    key: str,
    default: bool,
    section: str,
) -> bool:
    value = mapping.get(
        key,
        default,
    )

    if not isinstance(value, bool):
        raise ValueError(
            f"'{section}.{key}' must be a boolean."
        )

    return value


def _optional_int(
    mapping: dict[str, Any],
    key: str,
    default: int,
    section: str,
) -> int:
    value = mapping.get(
        key,
        default,
    )

    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise ValueError(
            f"'{section}.{key}' must be an integer."
        )

    return value


def load_training_config(
    config_path: str | Path,
) -> FullTrainingConfig:
    """
    Load and validate the complete training configuration.

    The loader remains backward compatible with the original
    training.yaml while supporting the Training Engine 2.0
    performance and precision settings.
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

    data = raw.get(
        "data",
        {},
    )

    tokenizer = raw.get(
        "tokenizer",
        {},
    )

    sequence = raw.get(
        "sequence",
        {},
    )

    dataloader = raw.get(
        "dataloader",
        {},
    )

    training = raw.get(
        "training",
        {},
    )

    output = raw.get(
        "output",
        {},
    )

    checkpoint = raw.get(
        "checkpoint",
        {},
    )

    sections = {
        "data": data,
        "tokenizer": tokenizer,
        "sequence": sequence,
        "dataloader": dataloader,
        "training": training,
        "output": output,
        "checkpoint": checkpoint,
    }

    for name, value in sections.items():
        if not isinstance(value, dict):
            raise ValueError(
                f"'{name}' must be a mapping."
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
            persistent_workers=_optional_bool(
                dataloader,
                "persistent_workers",
                False,
                "dataloader",
            ),
            prefetch_factor=_optional_int(
                dataloader,
                "prefetch_factor",
                2,
                "dataloader",
            ),
        ),
        training=TrainingConfig(
            device=_require(
                training,
                "device",
                "training",
            ),
            precision=training.get(
                "precision",
                "auto",
            ),
            compile=_optional_bool(
                training,
                "compile",
                False,
                "training",
            ),
            gradient_checkpointing=_optional_bool(
                training,
                "gradient_checkpointing",
                False,
                "training",
            ),
            allow_tf32=_optional_bool(
                training,
                "allow_tf32",
                True,
                "training",
            ),
            cudnn_benchmark=_optional_bool(
                training,
                "cudnn_benchmark",
                True,
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
    # --------------------------------------------------------------
    # Sequence
    # --------------------------------------------------------------

    if config.sequence.length <= 0:
        raise ValueError(
            "sequence.length must be > 0."
        )

    # --------------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------------

    loader = config.dataloader

    if loader.batch_size <= 0:
        raise ValueError(
            "dataloader.batch_size must be > 0."
        )

    if loader.num_workers < 0:
        raise ValueError(
            "dataloader.num_workers must be >= 0."
        )

    if loader.prefetch_factor <= 0:
        raise ValueError(
            "dataloader.prefetch_factor must be > 0."
        )

    if (
        loader.persistent_workers
        and loader.num_workers == 0
    ):
        raise ValueError(
            "dataloader.persistent_workers requires "
            "num_workers > 0."
        )

    if (
        loader.prefetch_factor != 2
        and loader.num_workers == 0
    ):
        raise ValueError(
            "dataloader.prefetch_factor requires "
            "num_workers > 0."
        )

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------

    training = config.training

    valid_devices = {
        "auto",
        "cpu",
        "cuda",
        "mps",
    }

    if training.device not in valid_devices:
        raise ValueError(
            "training.device must be one of: "
            "auto, cpu, cuda, mps."
        )

    valid_precisions = {
        "auto",
        "fp32",
        "fp16",
        "bf16",
    }

    if training.precision not in valid_precisions:
        raise ValueError(
            "training.precision must be one of: "
            "auto, fp32, fp16, bf16."
        )

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

    if training.gradient_accumulation_steps <= 0:
        raise ValueError(
            "training.gradient_accumulation_steps "
            "must be > 0."
        )

    if training.max_grad_norm <= 0:
        raise ValueError(
            "training.max_grad_norm must be > 0."
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

    if training.log_every_steps <= 0:
        raise ValueError(
            "training.log_every_steps must be > 0."
        )

    if training.eval_every_steps <= 0:
        raise ValueError(
            "training.eval_every_steps must be > 0."
        )

    if training.checkpoint_every_steps <= 0:
        raise ValueError(
            "training.checkpoint_every_steps must be > 0."
        )

    if (
        training.max_eval_batches is not None
        and training.max_eval_batches <= 0
    ):
        raise ValueError(
            "training.max_eval_batches must be > 0 "
            "when provided."
        )

    if training.seed < 0:
        raise ValueError(
            "training.seed must be >= 0."
        )

    # --------------------------------------------------------------
    # Precision/device compatibility
    # --------------------------------------------------------------

    if (
        training.device == "cpu"
        and training.precision == "fp16"
    ):
        raise ValueError(
            "FP16 training is not supported by the "
            "current CPU training path. Use fp32, bf16 "
            "where supported, or auto."
        )

    if (
        training.device == "cpu"
        and training.compile
    ):
        # torch.compile can technically work on CPU, but
        # enabling it automatically in a small development
        # environment can introduce significant compilation
        # overhead. Keep it explicitly opt-in.
        pass

    if (
        training.device == "mps"
        and training.precision == "fp16"
    ):
        # FP16 is supported by many MPS configurations.
        # No hard rejection is needed here.
        pass

    # TF32 is meaningful only on CUDA hardware.
    # We do not reject it because the runtime layer will
    # simply ignore it on unsupported devices.