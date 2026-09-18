from __future__ import annotations

import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.generator import (
    GenerationConfig,
    TextGenerator,
)
from src.model.config import load_model_config
from src.model.model import MujahidReasonerModel


CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "runs"
    / "mujahid-reasoner"
    / "checkpoints"
    / "best.pt"
)

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "tokenizer"
    / "tokenizer.json"
)

MODEL_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "model.yaml"
)


def main() -> None:
    prompt = input("\nPrompt: ").strip()

    if not prompt:
        raise ValueError("Prompt must not be empty.")

    device = "cpu"

    model_config = load_model_config(
        MODEL_CONFIG_PATH
    )

    model = MujahidReasonerModel(model_config)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    generator = TextGenerator(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    generation_config = GenerationConfig(
        max_new_tokens=100,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    )

    print("\nGenerating with KV cache...\n")

    output = generator.generate(
        prompt=prompt,
        config=generation_config,
        use_cache=True,
    )

    print("=" * 60)
    print(output)
    print("=" * 60)


if __name__ == "__main__":
    main()