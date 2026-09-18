from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import (
    DataLoaderConfig,
    LanguageModelDataModule,
)
from src.evaluation.evaluator import ModelEvaluator
from src.model.config import load_model_config
from src.model.model import MujahidReasonerModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Mujahid-Reasoner checkpoint."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "experiments/runs/mujahid-reasoner/"
            "checkpoints/best.pt"
        ),
    )

    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model.yaml"),
    )

    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/training.yaml"),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration must be a YAML mapping: {path}"
        )

    return config


def build_data_module(
    training_config_path: Path,
) -> LanguageModelDataModule:
    config = load_yaml(training_config_path)

    data_config = config["data"]
    sequence_config = config["sequence"]
    dataloader_config = config["dataloader"]

    loader_config = DataLoaderConfig(
        batch_size=int(
            dataloader_config["batch_size"]
        ),
        shuffle=False,
        num_workers=int(
            dataloader_config["num_workers"]
        ),
        pin_memory=bool(
            dataloader_config["pin_memory"]
        ),
        drop_last=False,
    )

    return LanguageModelDataModule(
        train_file=PROJECT_ROOT
        / data_config["train_file"],
        validation_file=PROJECT_ROOT
        / data_config["validation_file"],
        sequence_length=int(
            sequence_config["length"]
        ),
        config=loader_config,
    )


def main() -> None:
    args = parse_args()

    checkpoint_path = (
        PROJECT_ROOT / args.checkpoint
    )
    model_config_path = (
        PROJECT_ROOT / args.model_config
    )
    training_config_path = (
        PROJECT_ROOT / args.training_config
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if not model_config_path.exists():
        raise FileNotFoundError(
            f"Model config not found: {model_config_path}"
        )

    if not training_config_path.exists():
        raise FileNotFoundError(
            f"Training config not found: {training_config_path}"
        )

    device = torch.device(args.device)

    print("=" * 70)
    print("MUJAHID-REASONER EVALUATION")
    print("=" * 70)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Device     : {device}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    model_config = load_model_config(
        model_config_path
    )

    model = MujahidReasonerModel(
        model_config
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Validation data
    # ------------------------------------------------------------------

    data_module = build_data_module(
        training_config_path
    )

    validation_loader = (
        data_module.validation_dataloader()
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    evaluator = ModelEvaluator(
        model=model,
        device=str(device),
    )

    print(
        f"Parameters : "
        f"{model.parameter_count():,}"
    )

    print(
        f"Validation batches : "
        f"{len(validation_loader):,}"
    )

    print("-" * 70)

    start_time = time.perf_counter()

    result = evaluator.evaluate(
        validation_loader
    )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    tokens_per_second = (
        result.tokens / elapsed_seconds
        if elapsed_seconds > 0
        else 0.0
    )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    metrics = {
        "checkpoint": str(
            checkpoint_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "parameters": model.parameter_count(),
        "validation_loss": result.loss,
        "validation_perplexity": result.perplexity,
        "tokens": result.tokens,
        "batches": result.batches,
        "evaluation_seconds": elapsed_seconds,
        "tokens_per_second": tokens_per_second,
        "device": str(device),
    }

    print(
        "Validation Loss       : "
        f"{result.loss:.6f}"
    )

    print(
        "Validation Perplexity : "
        f"{result.perplexity:.4f}"
    )

    print(
        "Tokens Evaluated      : "
        f"{result.tokens:,}"
    )

    print(
        "Batches Evaluated     : "
        f"{result.batches:,}"
    )

    print(
        "Evaluation Time       : "
        f"{elapsed_seconds:.2f}s"
    )

    print(
        "Evaluation Throughput : "
        f"{tokens_per_second:.2f} tokens/sec"
    )

    print("-" * 70)

    output_path = (
        PROJECT_ROOT
        / "experiments"
        / "runs"
        / "mujahid-reasoner"
        / "evaluation.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print(
        f"Saved evaluation: {output_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()