from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.generator import GenerationConfig, TextGenerator
from src.model.config import load_model_config
from src.model.model import MujahidReasonerModel
from tokenizers import Tokenizer


CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "runs"
    / "mujahid-reasoner"
    / "checkpoints"
    / "best.pt"
)

MODEL_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "model.yaml"
)

TOKENIZER_PATH = (
    PROJECT_ROOT / "tokenizer" / "tokenizer.json"
)

PROMPTS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "benchmarks"
    / "prompts.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "benchmarks"
    / "baseline_results.json"
)

SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model() -> MujahidReasonerModel:
    config = load_model_config(
        MODEL_CONFIG_PATH
    )

    model = MujahidReasonerModel(config)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def load_prompts() -> list[dict[str, str]]:
    with PROMPTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        prompts = json.load(file)

    if not isinstance(prompts, list):
        raise ValueError(
            "prompts.json must contain a list."
        )

    return prompts


def main() -> None:
    set_seed(SEED)

    print("=" * 70)
    print("MUJAHID-REASONER GENERATION BENCHMARK")
    print("=" * 70)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}"
        )

    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(
            f"Prompts not found: {PROMPTS_PATH}"
        )

    model = load_model()

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    generator = TextGenerator(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
    )

    prompts = load_prompts()

    generation_config = GenerationConfig(
        max_new_tokens=50,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    )

    results: list[dict] = []

    total_generated_tokens = 0
    total_generation_time = 0.0

    for index, item in enumerate(prompts, start=1):
        prompt_id = item["id"]
        category = item["category"]
        prompt = item["prompt"]

        print()
        print(
            f"[{index}/{len(prompts)}] "
            f"{prompt_id} ({category})"
        )
        print(f"Prompt: {prompt}")

        set_seed(SEED)

        start_time = time.perf_counter()

        generated_text = generator.generate(
            prompt=prompt,
            config=generation_config,
        )

        elapsed = (
            time.perf_counter() - start_time
        )

        prompt_tokens = len(
            tokenizer.encode(prompt).ids
        )

        total_tokens = len(
            tokenizer.encode(generated_text).ids
        )

        generated_tokens = max(
            0,
            total_tokens - prompt_tokens,
        )

        tokens_per_second = (
            generated_tokens / elapsed
            if elapsed > 0
            else 0.0
        )

        print(f"Output: {generated_text}")
        print(
            f"Generated tokens: {generated_tokens}"
        )
        print(
            f"Time: {elapsed:.2f}s"
        )
        print(
            f"Throughput: "
            f"{tokens_per_second:.2f} tok/s"
        )

        result = {
            "id": prompt_id,
            "category": category,
            "prompt": prompt,
            "output": generated_text,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "generation_seconds": elapsed,
            "tokens_per_second": tokens_per_second,
        }

        results.append(result)

        total_generated_tokens += generated_tokens
        total_generation_time += elapsed

    average_tokens_per_second = (
        total_generated_tokens
        / total_generation_time
        if total_generation_time > 0
        else 0.0
    )

    report = {
        "model": {
            "name": "mujahid-reasoner",
            "checkpoint": str(
                CHECKPOINT_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "parameters": model.parameter_count(),
        },
        "generation": {
            "seed": SEED,
            "max_new_tokens": generation_config.max_new_tokens,
            "temperature": generation_config.temperature,
            "top_k": generation_config.top_k,
            "top_p": generation_config.top_p,
        },
        "summary": {
            "prompt_count": len(results),
            "total_generated_tokens": total_generated_tokens,
            "total_generation_seconds": total_generation_time,
            "average_tokens_per_second": average_tokens_per_second,
        },
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(
        f"Prompts              : {len(results)}"
    )
    print(
        f"Generated tokens     : "
        f"{total_generated_tokens:,}"
    )
    print(
        f"Total generation time: "
        f"{total_generation_time:.2f}s"
    )
    print(
        f"Average throughput   : "
        f"{average_tokens_per_second:.2f} tok/s"
    )
    print("-" * 70)
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()