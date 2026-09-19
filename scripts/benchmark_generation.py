from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable

import torch
import yaml
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.ablation import (
    AblationSuite,
    format_ablation_suite,
    run_ablation,
)
from src.inference.benchmarks import (
    BenchmarkSuiteResult,
    GenerationBenchmarkResult,
    format_benchmark_suite,
)
from src.inference.config import GenerationConfig
from src.inference.generator import TextGenerator
from src.model.config import load_model_config
from src.model.model import MujahidReasonerModel


DEFAULT_PROMPTS = (
    "Explain how a transformer works.",
    "What is gradient descent?",
    "Write a Python function to calculate Fibonacci numbers.",
    "Solve this mathematical problem: 12 * 17 + 8.",
    "Explain the difference between TCP and UDP.",
    "What is retrieval augmented generation?",
    "Write a Python function that checks whether a number is prime.",
    "Explain why attention is important in modern language models.",
)

HOMOGENEOUS_PROMPT = (
    "Explain the architecture of a modern decoder-only transformer "
    "language model in simple technical terms."
)

WARMUP_RUNS = 1
MEASUREMENT_RUNS = 3


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a project-relative path against the repository root."""

    candidate = Path(path)

    if candidate.is_absolute():
        return candidate

    return PROJECT_ROOT / candidate


def load_yaml(path: str | Path) -> dict:
    """Load a YAML configuration file."""

    config_path = resolve_project_path(path)

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Configuration file must contain a mapping: {config_path}"
        )

    return data


def load_generator() -> TextGenerator:
    """Load the trained Mujahid-Reasoner checkpoint."""

    model_config = load_model_config(
        resolve_project_path(
            "configs/model.yaml"
        )
    )

    tokenizer_path = resolve_project_path(
        "tokenizer/tokenizer.json"
    )

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}"
        )

    tokenizer = Tokenizer.from_file(
        str(tokenizer_path)
    )

    model = MujahidReasonerModel(
        model_config
    )

    checkpoint_path = resolve_project_path(
        "experiments/runs/mujahid-reasoner/"
        "checkpoints/best.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Unsupported checkpoint format."
        )

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if state_dict is None:
        state_dict = checkpoint.get(
            "model"
        )

    if state_dict is None:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(device)
    model.eval()

    return TextGenerator(
        model=model,
        tokenizer=tokenizer,
        device=str(device),
    )


def build_generation_config(
    *,
    do_sample: bool = False,
) -> GenerationConfig:
    """Build a deterministic generation configuration."""

    return GenerationConfig(
        min_new_tokens=25,
        max_new_tokens=25,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        do_sample=do_sample,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop_sequences=(),
    )


def synchronize(
    device: torch.device,
) -> None:
    """Synchronize CUDA before timing measurements."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_repeated(
    benchmark_fn: Callable[
        [],
        GenerationBenchmarkResult,
    ],
    runs: int = MEASUREMENT_RUNS,
) -> GenerationBenchmarkResult:
    """
    Run the same benchmark multiple times and return
    the median measurement.
    """

    if runs <= 0:
        raise ValueError(
            "runs must be positive."
        )

    measurements = [
        benchmark_fn()
        for _ in range(runs)
    ]

    elapsed = median(
        result.elapsed_seconds
        for result in measurements
    )

    generated_tokens = measurements[0].generated_tokens

    if any(
        result.generated_tokens != generated_tokens
        for result in measurements
    ):
        raise RuntimeError(
            "Benchmark runs generated different "
            "numbers of tokens."
        )

    reference = measurements[0]

    return GenerationBenchmarkResult(
        name=reference.name,
        batch_size=reference.batch_size,
        prompt_count=reference.prompt_count,
        generated_tokens=generated_tokens,
        elapsed_seconds=elapsed,
        tokens_per_second=(
            generated_tokens / elapsed
            if elapsed > 0.0
            else 0.0
        ),
    )


def benchmark_sequential(
    generator: TextGenerator,
    prompts: tuple[str, ...],
    config: GenerationConfig,
    *,
    use_cache: bool,
    name: str,
) -> GenerationBenchmarkResult:
    """Benchmark sequential generation."""

    for _ in range(WARMUP_RUNS):
        generator.generate_with_stats(
            prompts[0],
            config=config,
            use_cache=use_cache,
        )

    synchronize(generator.device)

    start = perf_counter()

    generated_tokens = 0

    for prompt in prompts:
        result = generator.generate_with_stats(
            prompt,
            config=config,
            use_cache=use_cache,
        )

        generated_tokens += (
            result.generated_token_count
        )

    synchronize(generator.device)

    elapsed = perf_counter() - start

    return GenerationBenchmarkResult(
        name=name,
        batch_size=1,
        prompt_count=len(prompts),
        generated_tokens=generated_tokens,
        elapsed_seconds=elapsed,
        tokens_per_second=(
            generated_tokens / elapsed
            if elapsed > 0.0
            else 0.0
        ),
    )


def benchmark_batch(
    generator: TextGenerator,
    prompts: tuple[str, ...],
    config: GenerationConfig,
    batch_size: int,
    *,
    name: str,
) -> GenerationBenchmarkResult:
    """Benchmark batched generation."""

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    warmup_prompts = prompts[
        : min(batch_size, len(prompts))
    ]

    for _ in range(WARMUP_RUNS):
        generator.generate_batch_with_stats(
            warmup_prompts,
            config=config,
        )

    synchronize(generator.device)

    start = perf_counter()

    generated_tokens = 0
    output_count = 0

    for start_index in range(
        0,
        len(prompts),
        batch_size,
    ):
        batch_prompts = prompts[
            start_index:
            start_index + batch_size
        ]

        result = generator.generate_batch_with_stats(
            batch_prompts,
            config=config,
        )

        generated_tokens += (
            result.total_generated_tokens
        )

        output_count += len(
            result.output.texts
        )

    synchronize(generator.device)

    elapsed = perf_counter() - start

    if output_count != len(prompts):
        raise RuntimeError(
            "Batch generation returned an unexpected "
            "number of outputs."
        )

    return GenerationBenchmarkResult(
        name=name,
        batch_size=batch_size,
        prompt_count=len(prompts),
        generated_tokens=generated_tokens,
        elapsed_seconds=elapsed,
        tokens_per_second=(
            generated_tokens / elapsed
            if elapsed > 0.0
            else 0.0
        ),
    )


def build_batch_suite(
    generator: TextGenerator,
    prompts: tuple[str, ...],
    config: GenerationConfig,
    prefix: str,
) -> BenchmarkSuiteResult:
    """Build a batch-size benchmark suite."""

    results: list[
        GenerationBenchmarkResult
    ] = []

    for batch_size in (
        1,
        2,
        4,
        8,
    ):
        result = benchmark_repeated(
            lambda batch_size=batch_size: benchmark_batch(
                generator,
                prompts,
                config,
                batch_size,
                name=f"{prefix}_bs{batch_size}",
            )
        )

        results.append(result)

    return BenchmarkSuiteResult(
        results=tuple(results)
    )


def build_cache_ablation(
    generator: TextGenerator,
    prompts: tuple[str, ...],
    config: GenerationConfig,
) -> AblationSuite:
    """Compare cached and non-cached sequential generation."""

    return run_ablation(
        name="KV Cache",
        variants=(
            (
                "no_cache",
                lambda: benchmark_repeated(
                    lambda: benchmark_sequential(
                        generator,
                        prompts,
                        config,
                        use_cache=False,
                        name="no_cache",
                    )
                ),
            ),
            (
                "cache",
                lambda: benchmark_repeated(
                    lambda: benchmark_sequential(
                        generator,
                        prompts,
                        config,
                        use_cache=True,
                        name="cache",
                    )
                ),
            ),
        ),
    )


def build_decoding_ablation(
    generator: TextGenerator,
    prompts: tuple[str, ...],
) -> AblationSuite:
    """Compare deterministic greedy and sampling generation."""

    greedy_config = build_generation_config(
        do_sample=False
    )

    sampling_config = build_generation_config(
        do_sample=True
    )

    return run_ablation(
        name="Decoding Strategy",
        variants=(
            (
                "greedy",
                lambda: benchmark_repeated(
                    lambda: benchmark_sequential(
                        generator,
                        prompts,
                        greedy_config,
                        use_cache=True,
                        name="greedy",
                    )
                ),
            ),
            (
                "sampling",
                lambda: benchmark_repeated(
                    lambda: benchmark_sequential(
                        generator,
                        prompts,
                        sampling_config,
                        use_cache=True,
                        name="sampling",
                    )
                ),
            ),
        ),
    )


def result_to_dict(
    result: GenerationBenchmarkResult,
) -> dict:
    """Convert a benchmark result to JSON."""

    return {
        "name": result.name,
        "batch_size": result.batch_size,
        "prompt_count": result.prompt_count,
        "generated_tokens": result.generated_tokens,
        "elapsed_seconds": result.elapsed_seconds,
        "tokens_per_second": result.tokens_per_second,
        "average_latency_seconds": (
            result.average_latency_seconds
        ),
    }


def suite_to_dict(
    suite: BenchmarkSuiteResult,
) -> dict:
    """Convert a benchmark suite to JSON."""

    baseline = suite.results[0]

    return {
        "results": [
            result_to_dict(result)
            for result in suite.results
        ],
        "throughput_ratio_against_baseline": {
            result.name: (
                result.tokens_per_second
                / baseline.tokens_per_second
                if baseline.tokens_per_second > 0.0
                else 0.0
            )
            for result in suite.results
        },
    }


def ablation_to_dict(
    suite: AblationSuite,
) -> dict:
    """Convert an ablation suite to JSON."""

    results = [
        result_to_dict(result)
        for result in suite.results
    ]

    comparisons = {}

    for result in suite.results[1:]:
        comparison = suite.compare(
            result.name
        )

        comparisons[result.name] = {
            "throughput_ratio": (
                comparison.throughput_ratio
            ),
            "throughput_difference_percent": (
                comparison.throughput_difference_percent
            ),
            "latency_ratio": (
                comparison.latency_ratio
            ),
            "latency_difference_percent": (
                comparison.latency_difference_percent
            ),
        }

    return {
        "results": results,
        "comparisons": comparisons,
    }


def build_report(
    *,
    device: torch.device,
    config: GenerationConfig,
    mixed_suite: BenchmarkSuiteResult,
    equal_suite: BenchmarkSuiteResult,
    cache_ablation: AblationSuite,
    decoding_ablation: AblationSuite,
) -> dict:
    """Build the complete experiment report."""

    return {
        "metadata": {
            "device": str(device),
            "warmup_runs": WARMUP_RUNS,
            "measurement_runs": MEASUREMENT_RUNS,
            "generation_tokens": config.max_new_tokens,
            "generation_config": {
                "min_new_tokens": config.min_new_tokens,
                "max_new_tokens": config.max_new_tokens,
                "do_sample": config.do_sample,
                "temperature": config.temperature,
                "top_k": config.top_k,
                "top_p": config.top_p,
                "repetition_penalty": (
                    config.repetition_penalty
                ),
                "frequency_penalty": (
                    config.frequency_penalty
                ),
                "presence_penalty": (
                    config.presence_penalty
                ),
            },
        },
        "benchmarks": {
            "mixed_length": suite_to_dict(
                mixed_suite
            ),
            "equal_length": suite_to_dict(
                equal_suite
            ),
        },
        "ablations": {
            "kv_cache": ablation_to_dict(
                cache_ablation
            ),
            "decoding": ablation_to_dict(
                decoding_ablation
            ),
        },
    }


def print_separator() -> None:
    print("-" * 72)


def main() -> None:
    """Run all Phase 5.9 inference experiments."""

    print(
        "Loading Mujahid-Reasoner..."
    )

    generator = load_generator()

    print(
        f"Device: {generator.device}"
    )

    config = build_generation_config()

    print(
        f"Generation tokens: "
        f"{config.max_new_tokens}"
    )

    print(
        f"Warmup runs: {WARMUP_RUNS}"
    )

    print(
        f"Measurement runs: {MEASUREMENT_RUNS}"
    )

    mixed_prompts = DEFAULT_PROMPTS

    equal_prompts = tuple(
        HOMOGENEOUS_PROMPT
        for _ in range(len(DEFAULT_PROMPTS))
    )

    # --------------------------------------------------------------
    # Batch benchmarks
    # --------------------------------------------------------------

    print()
    print(
        "=" * 72
    )
    print(
        "MIXED-LENGTH BATCH BENCHMARK"
    )
    print(
        "=" * 72
    )

    mixed_suite = build_batch_suite(
        generator,
        mixed_prompts,
        config,
        "mixed",
    )

    print(
        format_benchmark_suite(
            mixed_suite
        )
    )

    print()
    print(
        "=" * 72
    )
    print(
        "EQUAL-LENGTH BATCH BENCHMARK"
    )
    print(
        "=" * 72
    )

    equal_suite = build_batch_suite(
        generator,
        equal_prompts,
        config,
        "equal",
    )

    print(
        format_benchmark_suite(
            equal_suite
        )
    )

    # --------------------------------------------------------------
    # KV cache ablation
    # --------------------------------------------------------------

    print()
    print(
        "=" * 72
    )
    print(
        "KV CACHE ABLATION"
    )
    print(
        "=" * 72
    )

    cache_ablation = build_cache_ablation(
        generator,
        equal_prompts,
        config,
    )

    print(
        format_ablation_suite(
            cache_ablation
        )
    )

    # --------------------------------------------------------------
    # Decoding ablation
    # --------------------------------------------------------------

    print()
    print(
        "=" * 72
    )
    print(
        "DECODING ABLATION"
    )
    print(
        "=" * 72
    )

    decoding_ablation = build_decoding_ablation(
        generator,
        equal_prompts,
    )

    print(
        format_ablation_suite(
            decoding_ablation
        )
    )

    # --------------------------------------------------------------
    # JSON report
    # --------------------------------------------------------------

    report = build_report(
        device=generator.device,
        config=config,
        mixed_suite=mixed_suite,
        equal_suite=equal_suite,
        cache_ablation=cache_ablation,
        decoding_ablation=decoding_ablation,
    )

    output_path = resolve_project_path(
        "experiments/runs/mujahid-reasoner/"
        "generation_benchmark.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print()
    print(
        "=" * 72
    )
    print(
        "Benchmark report saved to:"
    )
    print(output_path)
    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()