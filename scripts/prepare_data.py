from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.data.cleaner import TextCleaner
from src.data.deduplicator import ExactDeduplicator
from src.data.filters import DocumentFilter, FilterConfig
from src.data.loader import DatasetConfig, DatasetLoader
from src.data.splitter import DatasetSplitter, SplitConfig
from src.data.statistics import DatasetStatistics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"


def load_config() -> dict:
    """Load the data pipeline configuration."""

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_documents(documents: list[str], path: Path) -> None:
    """Save documents as JSON Lines."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for document in documents:
            json.dump(document, file, ensure_ascii=False)
            file.write("\n")


def main() -> None:
    config = load_config()

    dataset_config = DatasetConfig(
        name=config["dataset"]["name"],
        config=config["dataset"]["config"],
        split=config["dataset"]["split"],
        streaming=config["dataset"]["streaming"],
    )

    filter_config = FilterConfig(
        min_characters=config["processing"]["min_characters"],
        max_characters=config["processing"]["max_characters"],
    )

    split_config = SplitConfig(
        validation_ratio=config["processing"]["validation_ratio"],
        seed=config["processing"]["seed"],
    )

    loader = DatasetLoader(dataset_config)
    cleaner = TextCleaner(
        remove_null_bytes=config["quality"]["remove_null_bytes"],
        normalize_whitespace=config["quality"]["normalize_whitespace"],
    )
    document_filter = DocumentFilter(filter_config)
    deduplicator = ExactDeduplicator()
    splitter = DatasetSplitter(split_config)
    statistics = DatasetStatistics()

    max_documents = config["processing"]["max_documents"]

    documents: list[str] = []

    print("Starting data preparation...")
    print(f"Dataset: {dataset_config.name}")
    print(f"Maximum documents: {max_documents:,}")

    for example in loader.stream():
        statistics.record_seen()

        text = cleaner.clean(example.get("text", ""))

        if not document_filter.is_valid(text):
            statistics.record_rejected()
            continue

        if deduplicator.is_duplicate(text):
            statistics.record_duplicate()
            continue

        statistics.record_kept(text)
        documents.append(text)

        if len(documents) >= max_documents:
            break

        if len(documents) % 1_000 == 0:
            print(f"Collected: {len(documents):,} documents")

    train_documents, validation_documents = splitter.split(documents)

    processed_dir = PROJECT_ROOT / config["output"]["processed_dir"]

    save_documents(
        train_documents,
        processed_dir / "train.jsonl",
    )

    save_documents(
        validation_documents,
        processed_dir / "validation.jsonl",
    )

    print("\nData preparation complete.")
    print("-" * 50)

    for key, value in statistics.summary().items():
        print(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")

    print(f"Train documents: {len(train_documents):,}")
    print(f"Validation documents: {len(validation_documents):,}")


if __name__ == "__main__":
    main()