from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import yaml
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "tokenizer.yaml"


class TokenizerTrainer:
    """Train and save a custom Byte-Level BPE tokenizer."""

    def __init__(self, config_path: Path) -> None:
        self.config = self._load_config(config_path)

        tokenizer_config = self.config["tokenizer"]
        data_config = self.config["data"]
        output_config = self.config["output"]
        training_config = self.config["training"]

        self.vocab_size = int(tokenizer_config["vocab_size"])
        self.min_frequency = int(tokenizer_config["min_frequency"])

        self.special_tokens = list(
            self.config["special_tokens"]
        )

        self.train_file = (
            PROJECT_ROOT / data_config["train_file"]
        )

        self.output_directory = (
            PROJECT_ROOT / output_config["directory"]
        )

        self.tokenizer_file = (
            self.output_directory
            / output_config["tokenizer_file"]
        )

        self.limit_documents = training_config.get(
            "limit_documents"
        )

    @staticmethod
    def _load_config(config_path: Path) -> dict:
        """Load YAML configuration."""

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError(
                "Tokenizer configuration must be a YAML mapping."
            )

        return config

    def _document_iterator(self) -> Iterable[str]:
        """Yield training documents from JSONL."""

        if not self.train_file.exists():
            raise FileNotFoundError(
                f"Training data not found: {self.train_file}"
            )

        with self.train_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            for index, line in enumerate(file):
                if (
                    self.limit_documents is not None
                    and index >= self.limit_documents
                ):
                    break

                line = line.strip()

                if not line:
                    continue

                document = json.loads(line)

                if isinstance(document, str) and document.strip():
                    yield document

    def train(self) -> Tokenizer:
        """Train the Byte-Level BPE tokenizer."""

        tokenizer = Tokenizer(
            BPE(
                unk_token="<unk>",
            )
        )

        tokenizer.pre_tokenizer = ByteLevel(
            add_prefix_space=False
        )

        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True,
        )

        tokenizer.train_from_iterator(
            self._document_iterator(),
            trainer=trainer,
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        tokenizer.save(str(self.tokenizer_file))

        return tokenizer


def main() -> None:
    """Train the Mujahid-Reasoner tokenizer."""

    print("=" * 70)
    print("Mujahid-Reasoner Tokenizer Training")
    print("=" * 70)

    trainer = TokenizerTrainer(CONFIG_PATH)

    print(f"Training data : {trainer.train_file}")
    print(f"Vocabulary    : {trainer.vocab_size:,}")
    print(f"Min frequency : {trainer.min_frequency}")
    print(f"Output        : {trainer.tokenizer_file}")
    print()

    tokenizer = trainer.train()

    print()
    print("Tokenizer training complete.")
    print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")
    print(f"Saved to: {trainer.tokenizer_file}")


if __name__ == "__main__":
    main()