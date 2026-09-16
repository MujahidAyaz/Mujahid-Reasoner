from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer.json"
INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "tokens"

SEQUENCE_LENGTH = 512


def load_documents(path: Path):
    """Yield documents from a JSONL file."""

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            document = json.loads(line)

            if isinstance(document, str) and document.strip():
                yield document


def tokenize_file(
    input_path: Path,
    output_path: Path,
    tokenizer: Tokenizer,
) -> dict:
    """Tokenize documents and save a packed binary token stream."""

    eos_id = tokenizer.token_to_id("<eos>")

    if eos_id is None:
        raise ValueError("Tokenizer does not contain <eos>.")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_tokens = 0
    document_count = 0

    with output_path.open("wb") as output_file:
        for document in load_documents(input_path):
            encoded = tokenizer.encode(document)

            token_ids = encoded.ids

            if not token_ids:
                continue

            # Explicitly separate documents.
            token_ids.append(eos_id)

            array = np.asarray(
                token_ids,
                dtype=np.uint32,
            )

            output_file.write(
                array.tobytes()
            )

            total_tokens += len(token_ids)
            document_count += 1

    sequence_count = (
        total_tokens // SEQUENCE_LENGTH
    )

    return {
        "documents": document_count,
        "tokens": total_tokens,
        "full_sequences": sequence_count,
        "discarded_tokens": (
            total_tokens
            - sequence_count * SEQUENCE_LENGTH
        ),
    }


def main() -> None:
    """Prepare packed token streams for training."""

    print("=" * 70)
    print("Mujahid-Reasoner Token Preparation")
    print("=" * 70)

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}"
        )

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    print(f"Tokenizer : {TOKENIZER_PATH}")
    print(f"Sequence  : {SEQUENCE_LENGTH}")
    print()

    results = {}

    for split in ("train", "validation"):
        input_path = (
            INPUT_DIR / f"{split}.jsonl"
        )

        output_path = (
            OUTPUT_DIR / f"{split}.bin"
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input dataset not found: {input_path}"
            )

        print(f"Processing {split}...")

        results[split] = tokenize_file(
            input_path=input_path,
            output_path=output_path,
            tokenizer=tokenizer,
        )

        print(
            f"  Documents : "
            f"{results[split]['documents']:,}"
        )
        print(
            f"  Tokens    : "
            f"{results[split]['tokens']:,}"
        )
        print(
            f"  Sequences : "
            f"{results[split]['full_sequences']:,}"
        )
        print(
            f"  Discarded : "
            f"{results[split]['discarded_tokens']:,}"
        )
        print()

    manifest = {
        "tokenizer": {
            "path": str(
                TOKENIZER_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "vocab_size": tokenizer.get_vocab_size(),
            "eos_token_id": tokenizer.token_to_id(
                "<eos>"
            ),
        },
        "sequence_length": SEQUENCE_LENGTH,
        "dtype": "uint32",
        "splits": results,
    }

    manifest_path = (
        OUTPUT_DIR / "token_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
        )

    print("Token preparation complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()