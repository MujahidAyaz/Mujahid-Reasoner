from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.jsonl"


def load_documents(path: Path) -> list[str]:
    """Load documents from a JSONL file."""

    documents: list[str] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            document = json.loads(line)

            if isinstance(document, str):
                documents.append(document)

    return documents


def main() -> None:
    """Inspect a processed corpus."""

    documents = load_documents(DATA_PATH)

    print(f"Dataset: {DATA_PATH}")
    print(f"Documents: {len(documents):,}")
    print()

    if not documents:
        print("No documents found.")
        return

    lengths = [len(document) for document in documents]

    print("Character statistics")
    print("-" * 40)
    print(f"Minimum: {min(lengths):,}")
    print(f"Maximum: {max(lengths):,}")
    print(f"Average: {sum(lengths) / len(lengths):,.2f}")
    print(f"Total: {sum(lengths):,}")
    print()

    print("Sample documents")
    print("=" * 80)

    sample_indices = [0, len(documents) // 2, len(documents) - 1]

    for index in sample_indices:
        document = documents[index]

        print(f"\n--- Document {index:,} ---")
        print(document[:1_500])
        print("\n")


if __name__ == "__main__":
    main()