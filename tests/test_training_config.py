from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.config import load_training_config


CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "training.yaml"
)


def test_load_training_config() -> None:
    config = load_training_config(
        CONFIG_PATH
    )

    assert config.data.train_file.endswith(
        "train.bin"
    )

    assert config.data.validation_file.endswith(
        "validation.bin"
    )

    assert config.tokenizer.path.endswith(
        "tokenizer.json"
    )

    assert config.sequence.length == 512

    assert config.dataloader.batch_size == 2

    assert config.training.learning_rate == pytest.approx(
        3e-4
    )

    assert config.training.weight_decay == pytest.approx(
        0.1
    )

    assert config.training.beta1 == pytest.approx(
        0.9
    )

    assert config.training.beta2 == pytest.approx(
        0.95
    )

    assert config.training.warmup_steps == 100

    assert config.training.min_lr_ratio == pytest.approx(
        0.1
    )

    assert config.checkpoint.resume is False


def test_missing_config_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Training configuration not found",
    ):
        load_training_config(
            tmp_path / "missing.yaml"
        )


def test_invalid_yaml_root(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.yaml"

    path.write_text(
        "- invalid\n- yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must contain a YAML mapping",
    ):
        load_training_config(path)


def test_missing_required_key(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"

    path.write_text(
        """
data:
  train_file: "train.bin"
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Missing required key",
    ):
        load_training_config(path)


def test_invalid_batch_size(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"

    path.write_text(
        """
data:
  train_file: "train.bin"
  validation_file: "validation.bin"

tokenizer:
  path: "tokenizer.json"

sequence:
  length: 512

dataloader:
  batch_size: 0
  shuffle: true
  num_workers: 0
  pin_memory: false
  drop_last: false

training:
  device: "cpu"
  max_epochs: 1
  max_steps: 10
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  learning_rate: 0.001
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  optimizer_eps: 0.00000001
  warmup_steps: 1
  min_lr_ratio: 0.1
  log_every_steps: 1
  eval_every_steps: 1
  checkpoint_every_steps: 1
  max_eval_batches: 1
  seed: 42

output:
  directory: "runs"

checkpoint:
  resume: false
  filename: "latest.pt"
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="batch_size",
    ):
        load_training_config(path)


def test_invalid_learning_rate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"

    path.write_text(
        """
data:
  train_file: "train.bin"
  validation_file: "validation.bin"

tokenizer:
  path: "tokenizer.json"

sequence:
  length: 512

dataloader:
  batch_size: 2
  shuffle: true
  num_workers: 0
  pin_memory: false
  drop_last: false

training:
  device: "cpu"
  max_epochs: 1
  max_steps: 10
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  learning_rate: 0
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  optimizer_eps: 0.00000001
  warmup_steps: 1
  min_lr_ratio: 0.1
  log_every_steps: 1
  eval_every_steps: 1
  checkpoint_every_steps: 1
  max_eval_batches: 1
  seed: 42

output:
  directory: "runs"

checkpoint:
  resume: false
  filename: "latest.pt"
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="learning_rate",
    ):
        load_training_config(path)


def test_invalid_beta(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"

    path.write_text(
        """
data:
  train_file: "train.bin"
  validation_file: "validation.bin"

tokenizer:
  path: "tokenizer.json"

sequence:
  length: 512

dataloader:
  batch_size: 2
  shuffle: true
  num_workers: 0
  pin_memory: false
  drop_last: false

training:
  device: "cpu"
  max_epochs: 1
  max_steps: 10
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  learning_rate: 0.001
  weight_decay: 0.1
  beta1: 1.0
  beta2: 0.95
  optimizer_eps: 0.00000001
  warmup_steps: 1
  min_lr_ratio: 0.1
  log_every_steps: 1
  eval_every_steps: 1
  checkpoint_every_steps: 1
  max_eval_batches: 1
  seed: 42

output:
  directory: "runs"

checkpoint:
  resume: false
  filename: "latest.pt"
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="beta1",
    ):
        load_training_config(path)