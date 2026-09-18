from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.generator import GenerationConfig, TextGenerator
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

TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer.json"

MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"


PROMPT = (
    "The future of artificial intelligence is "
    "changing the way people work, learn, and solve problems."
)

MAX_NEW_TOKENS = 50


def load_generator() -> TextGenerator:
    device = "cpu"

    model_config = load_model_config(MODEL_CONFIG_PATH)
    model = MujahidReasonerModel(model_config)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    return TextGenerator(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


def benchmark(
    generator: TextGenerator,
    use_cache: bool,
) -> tuple[float, int, float, str]:
    config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
    )

    start = time.perf_counter()

    output = generator.generate(
        prompt=PROMPT,
        config=config,
        use_cache=use_cache,
    )

    elapsed = time.perf_counter() - start

    encoded_output = generator.tokenizer.encode(output)

    prompt_tokens = len(
        generator.tokenizer.encode(PROMPT).ids
    )

    generated_tokens = max(
        0,
        len(encoded_output.ids) - prompt_tokens,
    )

    tokens_per_second = (
        generated_tokens / elapsed
        if elapsed > 0
        else 0.0
    )

    return (
        elapsed,
        generated_tokens,
        tokens_per_second,
        output,
    )


def main() -> None:
    print("=" * 70)
    print("MUJAHID-REASONER KV CACHE BENCHMARK")
    print("=" * 70)

    print("\nLoading model...")
    generator = load_generator()

    print(f"Prompt: {PROMPT}")
    print(f"Requested new tokens: {MAX_NEW_TOKENS}")

    print("\nRunning without KV cache...")
    no_cache_time, no_cache_tokens, no_cache_tps, no_cache_output = (
        benchmark(
            generator,
            use_cache=False,
        )
    )

    print("\nRunning with KV cache...")
    cache_time, cache_tokens, cache_tps, cache_output = benchmark(
        generator,
        use_cache=True,
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"\nWithout KV cache:"
        f"\n  Time: {no_cache_time:.3f}s"
        f"\n  Tokens: {no_cache_tokens}"
        f"\n  Throughput: {no_cache_tps:.2f} tok/s"
    )

    print(
        f"\nWith KV cache:"
        f"\n  Time: {cache_time:.3f}s"
        f"\n  Tokens: {cache_tokens}"
        f"\n  Throughput: {cache_tps:.2f} tok/s"
    )

    if cache_time > 0:
        speedup = no_cache_time / cache_time
    else:
        speedup = 0.0

    print(
        f"\nKV-cache speedup: {speedup:.2f}x"
    )

    print("\n" + "=" * 70)
    print("NON-CACHED OUTPUT")
    print("=" * 70)
    print(no_cache_output)

    print("\n" + "=" * 70)
    print("CACHED OUTPUT")
    print("=" * 70)
    print(cache_output)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()