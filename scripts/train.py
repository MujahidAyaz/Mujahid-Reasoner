from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import (
    DataLoaderConfig,
    LanguageModelDataModule,
    set_seed,
)
from src.model.config import load_model_config
from src.model.model import MujahidReasonerModel
from src.training.config import load_training_config
from src.training.loss import CausalLanguageModelLoss
from src.training.optimizer import create_adamw_optimizer
from src.training.scheduler import create_warmup_cosine_scheduler
from src.training.trainer import Trainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Mujahid-Reasoner."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "training.yaml",
        help="Path to training configuration.",
    )

    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "model.yaml",
        help="Path to model configuration.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override maximum training steps.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override training device.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the configured checkpoint.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_training_config(args.config)
    model_config = load_model_config(args.model_config)

    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else config.training.max_steps
    )

    device = (
        args.device
        if args.device is not None
        else config.training.device
    )

    resume = args.resume or config.checkpoint.resume

    set_seed(config.training.seed)

    print("=" * 60)
    print("Mujahid-Reasoner Training")
    print("=" * 60)

    print(f"Device          : {device}")
    print(f"Precision       : {config.training.precision}")
    print(f"Compile         : {config.training.compile}")
    print(
        f"Gradient Checkpointing : "
        f"{config.training.gradient_checkpointing}"
    )
    print(f"Sequence length : {config.sequence.length}")
    print(f"Batch size      : {config.dataloader.batch_size}")
    print(
        f"Gradient accumulation : "
        f"{config.training.gradient_accumulation_steps}"
    )
    print(f"Max steps       : {max_steps}")
    print()

    data_loader_config = DataLoaderConfig(
        batch_size=config.dataloader.batch_size,
        shuffle=config.dataloader.shuffle,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        drop_last=config.dataloader.drop_last,
    )

    data_module = LanguageModelDataModule(
        train_file=PROJECT_ROOT / config.data.train_file,
        validation_file=PROJECT_ROOT / config.data.validation_file,
        sequence_length=config.sequence.length,
        config=data_loader_config,
        seed=config.training.seed,
    )

    train_loader = data_module.train_dataloader()
    validation_loader = data_module.validation_dataloader()

    model = MujahidReasonerModel(model_config)

    parameter_count = model.parameter_count()

    print(f"Parameters      : {parameter_count:,}")

    optimizer = create_adamw_optimizer(
        model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(
            config.training.beta1,
            config.training.beta2,
        ),
        eps=config.training.optimizer_eps,
    )

    total_steps = config.training.max_steps

    if total_steps is None:
        steps_per_epoch = (
            len(train_loader)
            // config.training.gradient_accumulation_steps
        )

        total_steps = steps_per_epoch * config.training.max_epochs

    scheduler = create_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=config.training.warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=config.training.min_lr_ratio,
    )

    trainer_config = TrainerConfig(
        device=device,
        precision=config.training.precision,
        allow_tf32=config.training.allow_tf32,
        cudnn_benchmark=config.training.cudnn_benchmark,
        compile=config.training.compile,
        gradient_checkpointing=config.training.gradient_checkpointing,
        max_epochs=config.training.max_epochs,
        max_steps=max_steps,
        gradient_accumulation_steps=(
            config.training.gradient_accumulation_steps
        ),
        max_grad_norm=config.training.max_grad_norm,
        log_every_steps=config.training.log_every_steps,
        eval_every_steps=config.training.eval_every_steps,
        checkpoint_every_steps=(
            config.training.checkpoint_every_steps
        ),
        max_eval_batches=config.training.max_eval_batches,
        output_dir=(
            PROJECT_ROOT / config.output.directory
        ).as_posix(),
        seed=config.training.seed,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=CausalLanguageModelLoss(),
        config=trainer_config,
        train_sampler=data_module.train_sampler,
    )

    if resume:
        print(
            f"Resuming from "
            f"{config.checkpoint.filename}"
        )

        trainer.resume(
            filename=config.checkpoint.filename
        )

    print()
    print("Starting training...")
    print()

    state = trainer.train()

    print()
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)

    print(f"Global step     : {state.global_step}")
    print(f"Train loss      : {state.train_loss:.4f}")

    if state.validation_loss != float("inf"):
        print(
            f"Validation loss : "
            f"{state.validation_loss:.4f}"
        )


if __name__ == "__main__":
    main()