from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence

import torch


@dataclass(frozen=True)
class GenerationBenchmarkResult:
    """
    Benchmark result for a single generation strategy.
    """

    name: str
    batch_size: int
    prompt_count: int
    generated_tokens: int
    elapsed_seconds: float
    tokens_per_second: float

    @property
    def average_latency_seconds(self) -> float:
        """Return average latency per prompt."""
        if self.prompt_count == 0:
            return 0.0

        return self.elapsed_seconds / self.prompt_count

    def speedup_against(
        self,
        baseline: GenerationBenchmarkResult,
    ) -> float:
        """
        Return throughput speedup relative to a baseline result.
        """

        if baseline.tokens_per_second <= 0.0:
            return 0.0

        return self.tokens_per_second / baseline.tokens_per_second


@dataclass(frozen=True)
class BenchmarkSuiteResult:
    """
    Collection of benchmark results from one benchmark suite.
    """

    results: tuple[GenerationBenchmarkResult, ...]

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError("results must not be empty.")

    @property
    def fastest(self) -> GenerationBenchmarkResult:
        """Return the result with the highest throughput."""
        return max(
            self.results,
            key=lambda result: result.tokens_per_second,
        )


def _synchronize_device(device: torch.device) -> None:
    """
    Synchronize an accelerator before/after timing when required.
    """

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_generation(
    *,
    name: str,
    generate_fn: Callable[
        [Sequence[str]],
        tuple[Sequence[str], int],
    ],
    prompts: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> GenerationBenchmarkResult:
    """
    Benchmark a generation callable.

    The generation callable must return:

        (generated_texts, exact_generated_token_count)

    This avoids estimating token counts by re-tokenizing decoded text.
    """

    if not name.strip():
        raise ValueError("name must not be empty.")

    if not prompts:
        raise ValueError("prompts must not be empty.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    _synchronize_device(device)

    start_time = perf_counter()

    outputs, generated_tokens = generate_fn(prompts)

    _synchronize_device(device)

    elapsed_seconds = perf_counter() - start_time

    if len(outputs) != len(prompts):
        raise ValueError(
            "generate_fn must return exactly one output per prompt."
        )

    if generated_tokens < 0:
        raise ValueError(
            "generated token count must be non-negative."
        )

    tokens_per_second = (
        generated_tokens / elapsed_seconds
        if elapsed_seconds > 0.0
        else 0.0
    )

    return GenerationBenchmarkResult(
        name=name,
        batch_size=batch_size,
        prompt_count=len(prompts),
        generated_tokens=generated_tokens,
        elapsed_seconds=elapsed_seconds,
        tokens_per_second=tokens_per_second,
    )


def format_benchmark_result(
    result: GenerationBenchmarkResult,
) -> str:
    """
    Format one benchmark result as a readable text report.
    """

    return "\n".join(
        [
            f"Benchmark: {result.name}",
            f"Batch size: {result.batch_size}",
            f"Prompts: {result.prompt_count}",
            f"Generated tokens: {result.generated_tokens}",
            f"Elapsed: {result.elapsed_seconds:.3f}s",
            f"Tokens/sec: {result.tokens_per_second:.2f}",
            (
                "Average latency/prompt: "
                f"{result.average_latency_seconds:.3f}s"
            ),
        ]
    )


def format_benchmark_suite(
    suite: BenchmarkSuiteResult,
) -> str:
    """
    Format a complete benchmark suite as readable text.
    """

    lines = [
        "=" * 72,
        "MUJAHID-REASONER GENERATION BENCHMARK",
        "=" * 72,
    ]

    for result in suite.results:
        lines.extend(
            [
                "",
                format_benchmark_result(result),
                "-" * 72,
            ]
        )

    return "\n".join(lines)