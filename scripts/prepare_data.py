
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.data.cleaner import TextCleaner
from src.data.deduplicator import ExactDeduplicator
from src.data.filters import DocumentFilter, FilterConfig
from src.data.loader import DatasetConfig, DatasetLoader
from src.data.quality import QualityConfig, TextQualityAnalyzer
from src.data.splitter import DatasetSplitter, SplitConfig
from src.data.statistics import DatasetStatistics


CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"


def load_config() -> dict:
    """Load the data pipeline configuration."""

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def append_document(
    file,
    document: str,
) -> None:
    """Append one document to a JSONL file."""

    json.dump(
        document,
        file,
        ensure_ascii=False,
    )
    file.write("\n")


def main() -> None:
    """Prepare a large corpus using streaming and disk-backed deduplication."""

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

    quality_config = QualityConfig(
        max_replacement_character_ratio=0.005,
        max_suspicious_encoding_ratio=0.01,
        max_repeated_character_ratio=0.05,
        max_symbol_ratio=0.30,
    )

    loader = DatasetLoader(dataset_config)

    cleaner = TextCleaner(
        remove_null_bytes=config["quality"]["remove_null_bytes"],
        normalize_whitespace=config["quality"]["normalize_whitespace"],
    )

    document_filter = DocumentFilter(filter_config)
    quality_analyzer = TextQualityAnalyzer(quality_config)

    processed_dir = (
        PROJECT_ROOT
        / config["output"]["processed_dir"]
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = processed_dir / "train.jsonl"
    validation_path = processed_dir / "validation.jsonl"

    dedup_dir = processed_dir / "dedup"
    dedup_database = dedup_dir / "fingerprints.sqlite"

    max_documents = int(
        config["processing"]["max_documents"]
    )

    validation_ratio = float(
        config["processing"]["validation_ratio"]
    )

    seed = int(
        config["processing"]["seed"]
    )

    random_generator = random.Random(seed)

    statistics = DatasetStatistics()

    print("=" * 70)
    print("Mujahid-Reasoner Data Preparation")
    print("=" * 70)
    print(f"Dataset            : {dataset_config.name}")
    print(f"Configuration      : {dataset_config.config}")
    print(f"Maximum documents  : {max_documents:,}")
    print(f"Validation ratio   : {validation_ratio:.4f}")
    print(f"Output directory   : {processed_dir}")
    print(f"Dedup database     : {dedup_database}")
    print()

    train_count = 0
    validation_count = 0
    accepted_count = 0

    with (
        train_path.open("w", encoding="utf-8") as train_file,
        validation_path.open("w", encoding="utf-8") as validation_file,
        ExactDeduplicator(dedup_database) as deduplicator,
    ):
        print("Starting data preparation...")
        print()

        for example in loader.stream():
            statistics.record_seen()

            text = cleaner.clean(
                example.get("text", "")
            )

            filter_result = document_filter.evaluate(text)

            if not filter_result.is_valid:
                statistics.record_rejected(
                    filter_result.reason
                )
                continue

            quality_result = quality_analyzer.evaluate(text)

            if not quality_result.is_valid:
                statistics.record_rejected(
                    quality_result.reason
                )
                continue

            if deduplicator.is_duplicate(text):
                statistics.record_duplicate()
                continue

            statistics.record_kept(text)

            accepted_count += 1

            if (
                random_generator.random()
                < validation_ratio
            ):
                append_document(
                    validation_file,
                    text,
                )
                validation_count += 1
            else:
                append_document(
                    train_file,
                    text,
                )
                train_count += 1

            if accepted_count % 1_000 == 0:
                print(
                    f"Collected: {accepted_count:,} | "
                    f"Train: {train_count:,} | "
                    f"Validation: {validation_count:,}"
                )

            if accepted_count >= max_documents:
                break

    manifest = {
        "dataset": {
            "name": dataset_config.name,
            "config": dataset_config.config,
            "split": dataset_config.split,
            "streaming": dataset_config.streaming,
        },
        "processing": {
            "max_documents": max_documents,
            "validation_ratio": validation_ratio,
            "seed": seed,
            "storage_mode": "streaming_jsonl",
            "deduplication": "sqlite_sha256",
        },
        "statistics": statistics.summary(),
        "output": {
            "train_documents": train_count,
            "validation_documents": validation_count,
            "total_documents": (
                train_count + validation_count
            ),
        },
    }

    manifest_path = (
        processed_dir / "dataset_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("Data preparation complete.")
    print("-" * 70)

    for key, value in statistics.summary().items():
        if isinstance(value, int):
            print(f"{key}: {value:,}")
        else:
            print(f"{key}: {value}")

    print(f"Train documents: {train_count:,}")
    print(f"Validation documents: {validation_count:,}")
    print(f"Total documents: {train_count + validation_count:,}")
    print(f"Manifest: {manifest_path}")
    
if __name__ == "__main__":
    main()