
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer.json"

SEQUENCE_LENGTH = 512
TOKEN_DTYPE = np.uint16


def load_config() -> dict:
    """Load the data pipeline configuration."""

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def tokenize_file(
    input_path: Path,
    output_path: Path,
    tokenizer: Tokenizer,
) -> dict:
    """Tokenize documents and save a packed binary token stream."""

    eos_id = tokenizer.token_to_id("<eos>")

    if eos_id is None:
        raise ValueError(
            "Tokenizer does not contain <eos>."
        )

    vocab_size = tokenizer.get_vocab_size()

    if vocab_size > np.iinfo(TOKEN_DTYPE).max + 1:
        raise ValueError(
            f"Tokenizer vocabulary size ({vocab_size:,}) "
            f"does not fit inside {TOKEN_DTYPE}."
        )

    if eos_id > np.iinfo(TOKEN_DTYPE).max:
        raise ValueError(
            f"<eos> token ID ({eos_id}) "
            f"does not fit inside {TOKEN_DTYPE}."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_tokens = 0
    document_count = 0

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as input_file, output_path.open(
        "wb"
    ) as output_file:

        for line in input_file:
            line = line.strip()

            if not line:
                continue

            document = json.loads(line)

            if (
                not isinstance(document, str)
                or not document.strip()
            ):
                continue

            token_ids = tokenizer.encode(document).ids

            if not token_ids:
                continue

            token_ids.append(eos_id)

            array = np.asarray(
                token_ids,
                dtype=TOKEN_DTYPE,
            )

            output_file.write(
                array.tobytes()
            )

            total_tokens += len(token_ids)
            document_count += 1

            if document_count % 10_000 == 0:
                print(
                    f"  Tokenized documents: "
                    f"{document_count:,} | "
                    f"Tokens: {total_tokens:,}"
                )

    sequence_count = (
        total_tokens // SEQUENCE_LENGTH
    )

    discarded_tokens = (
        total_tokens
        - sequence_count * SEQUENCE_LENGTH
    )

    return {
        "documents": document_count,
        "tokens": total_tokens,
        "full_sequences": sequence_count,
        "discarded_tokens": discarded_tokens,
    }


def main() -> None:
    """Prepare packed token streams for training."""

    config = load_config()

    processed_dir = (
        PROJECT_ROOT
        / config["output"]["processed_dir"]
    )

    output_dir = (
        processed_dir / "tokens"
    )

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

    vocab_size = tokenizer.get_vocab_size()

    print(
        f"Tokenizer : {TOKENIZER_PATH}"
    )
    print(
        f"Vocabulary: {vocab_size:,}"
    )
    print(
        f"Input     : {processed_dir}"
    )
    print(
        f"Output    : {output_dir}"
    )
    print(
        f"Sequence  : {SEQUENCE_LENGTH}"
    )
    print(
        f"Dtype     : {TOKEN_DTYPE}"
    )
    print()

    results = {}

    for split in (
        "train",
        "validation",
    ):
        input_path = (
            processed_dir
            / f"{split}.jsonl"
        )

        output_path = (
            output_dir
            / f"{split}.bin"
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input dataset not found: "
                f"{input_path}"
            )

        print(
            f"Processing {split}..."
        )

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
            "vocab_size": vocab_size,
            "eos_token_id": tokenizer.token_to_id(
                "<eos>"
            ),
        },
        "sequence_length": SEQUENCE_LENGTH,
        "dtype": "uint16",
        "splits": results,
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        output_dir
        / "token_manifest.json"
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

    print(
        "Token preparation complete."
    )
    print(
        f"Output directory: {output_dir}"
    )
    print(
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
