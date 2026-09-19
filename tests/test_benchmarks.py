from __future__ import annotations

import pytest
import torch

from src.inference.benchmarks import (
    BenchmarkSuiteResult,
    GenerationBenchmarkResult,
    benchmark_generation,
    format_benchmark_result,
    format_benchmark_suite,
)


class FakeTokenizer:
    class Encoded:
        def __init__(self, token_count: int) -> None:
            self.ids = list(range(token_count))

    def encode(self, text: str) -> Encoded:
        return self.Encoded(len(text.split()))


def make_device() -> torch.device:
    return torch.device("cpu")


def make_generator_fn(
    generated_token_count: int = 10,
):
    def generate(
        prompts,
    ):
        outputs = [
            f"generated output {index}"
            for index, _ in enumerate(prompts)
        ]

        return outputs, generated_token_count

    return generate


def test_generation_benchmark_result_average_latency() -> None:
    result = GenerationBenchmarkResult(
        name="test",
        batch_size=2,
        prompt_count=4,
        generated_tokens=100,
        elapsed_seconds=5.0,
        tokens_per_second=20.0,
    )

    assert result.average_latency_seconds == 1.25


def test_generation_benchmark_result_zero_prompt_latency() -> None:
    result = GenerationBenchmarkResult(
        name="test",
        batch_size=1,
        prompt_count=0,
        generated_tokens=0,
        elapsed_seconds=0.0,
        tokens_per_second=0.0,
    )

    assert result.average_latency_seconds == 0.0


def test_speedup_against_baseline() -> None:
    baseline = GenerationBenchmarkResult(
        name="baseline",
        batch_size=1,
        prompt_count=4,
        generated_tokens=100,
        elapsed_seconds=10.0,
        tokens_per_second=10.0,
    )

    result = GenerationBenchmarkResult(
        name="optimized",
        batch_size=4,
        prompt_count=4,
        generated_tokens=100,
        elapsed_seconds=5.0,
        tokens_per_second=20.0,
    )

    assert result.speedup_against(baseline) == 2.0


def test_speedup_against_zero_throughput() -> None:
    baseline = GenerationBenchmarkResult(
        name="baseline",
        batch_size=1,
        prompt_count=1,
        generated_tokens=0,
        elapsed_seconds=1.0,
        tokens_per_second=0.0,
    )

    result = GenerationBenchmarkResult(
        name="optimized",
        batch_size=2,
        prompt_count=2,
        generated_tokens=10,
        elapsed_seconds=1.0,
        tokens_per_second=10.0,
    )

    assert result.speedup_against(baseline) == 0.0


def test_benchmark_suite_requires_results() -> None:
    with pytest.raises(ValueError, match="results must not be empty"):
        BenchmarkSuiteResult(results=())


def test_benchmark_suite_fastest() -> None:
    slow = GenerationBenchmarkResult(
        name="slow",
        batch_size=1,
        prompt_count=2,
        generated_tokens=20,
        elapsed_seconds=2.0,
        tokens_per_second=10.0,
    )

    fast = GenerationBenchmarkResult(
        name="fast",
        batch_size=4,
        prompt_count=2,
        generated_tokens=40,
        elapsed_seconds=1.0,
        tokens_per_second=40.0,
    )

    suite = BenchmarkSuiteResult(
        results=(slow, fast)
    )

    assert suite.fastest.name == "fast"


def test_benchmark_generation_returns_exact_token_count() -> None:
    prompts = (
        "hello world",
        "test prompt",
    )

    result = benchmark_generation(
        name="test",
        generate_fn=make_generator_fn(37),
        prompts=prompts,
        device=make_device(),
        batch_size=2,
    )

    assert result.generated_tokens == 37
    assert result.prompt_count == 2
    assert result.batch_size == 2
    assert result.tokens_per_second > 0.0


def test_benchmark_generation_validates_name() -> None:
    with pytest.raises(ValueError, match="name must not be empty"):
        benchmark_generation(
            name="",
            generate_fn=make_generator_fn(),
            prompts=("hello",),
            device=make_device(),
            batch_size=1,
        )


def test_benchmark_generation_validates_prompts() -> None:
    with pytest.raises(ValueError, match="prompts must not be empty"):
        benchmark_generation(
            name="test",
            generate_fn=make_generator_fn(),
            prompts=(),
            device=make_device(),
            batch_size=1,
        )


def test_benchmark_generation_validates_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        benchmark_generation(
            name="test",
            generate_fn=make_generator_fn(),
            prompts=("hello",),
            device=make_device(),
            batch_size=0,
        )


def test_benchmark_generation_validates_output_count() -> None:
    def generate(prompts):
        return ["only one"], 10

    with pytest.raises(
        ValueError,
        match="exactly one output per prompt",
    ):
        benchmark_generation(
            name="test",
            generate_fn=generate,
            prompts=("hello", "world"),
            device=make_device(),
            batch_size=2,
        )


def test_benchmark_generation_validates_token_count() -> None:
    def generate(prompts):
        return ["output"], -1

    with pytest.raises(
        ValueError,
        match="generated token count must be non-negative",
    ):
        benchmark_generation(
            name="test",
            generate_fn=generate,
            prompts=("hello",),
            device=make_device(),
            batch_size=1,
        )


def test_benchmark_generation_zero_elapsed_produces_zero_throughput(
    monkeypatch,
) -> None:
    import src.inference.benchmarks as benchmarks

    monkeypatch.setattr(
        benchmarks,
        "perf_counter",
        lambda: 1.0,
    )

    result = benchmark_generation(
        name="test",
        generate_fn=make_generator_fn(10),
        prompts=("hello",),
        device=make_device(),
        batch_size=1,
    )

    assert result.elapsed_seconds == 0.0
    assert result.tokens_per_second == 0.0


def test_format_benchmark_result() -> None:
    result = GenerationBenchmarkResult(
        name="batched_bs4",
        batch_size=4,
        prompt_count=8,
        generated_tokens=200,
        elapsed_seconds=2.5,
        tokens_per_second=80.0,
    )

    formatted = format_benchmark_result(result)

    assert "Benchmark: batched_bs4" in formatted
    assert "Batch size: 4" in formatted
    assert "Prompts: 8" in formatted
    assert "Generated tokens: 200" in formatted
    assert "Elapsed: 2.500s" in formatted
    assert "Tokens/sec: 80.00" in formatted
    assert "Average latency/prompt: 0.312s" in formatted


def test_format_benchmark_suite() -> None:
    result = GenerationBenchmarkResult(
        name="sequential",
        batch_size=1,
        prompt_count=2,
        generated_tokens=20,
        elapsed_seconds=1.0,
        tokens_per_second=20.0,
    )

    suite = BenchmarkSuiteResult(
        results=(result,)
    )

    formatted = format_benchmark_suite(suite)

    assert "MUJAHID-REASONER GENERATION BENCHMARK" in formatted
    assert "Benchmark: sequential" in formatted
    assert "Batch size: 1" in formatted


def test_benchmark_generation_uses_exact_returned_count() -> None:
    tokenizer = FakeTokenizer()

    def generate(prompts):
        return ["this output has five words"], 99

    result = benchmark_generation(
        name="exact-count",
        generate_fn=generate,
        prompts=("short",),
        device=make_device(),
        batch_size=1,
    )

    estimated_count = len(
        tokenizer.encode(
            "this output has five words"
        ).ids
    )

    assert estimated_count == 5
    assert result.generated_tokens == 99


def test_benchmark_generation_supports_multiple_prompts() -> None:
    prompts = (
        "prompt one",
        "prompt two",
        "prompt three",
        "prompt four",
    )

    result = benchmark_generation(
        name="multi-prompt",
        generate_fn=make_generator_fn(100),
        prompts=prompts,
        device=make_device(),
        batch_size=4,
    )

    assert result.prompt_count == 4
    assert result.batch_size == 4
    assert result.generated_tokens == 100


def test_benchmark_generation_empty_output_with_zero_tokens() -> None:
    def generate(prompts):
        return [""] * len(prompts), 0

    result = benchmark_generation(
        name="empty",
        generate_fn=generate,
        prompts=("hello",),
        device=make_device(),
        batch_size=1,
    )

    assert result.generated_tokens == 0
    assert result.tokens_per_second == 0.0


def test_benchmark_generation_preserves_output_order() -> None:
    prompts = (
        "first",
        "second",
        "third",
    )

    def generate(prompts_to_generate):
        outputs = [
            f"output-{index}"
            for index, _ in enumerate(prompts_to_generate)
        ]

        return outputs, 30

    captured = generate(prompts)

    assert captured[0] == [
        "output-0",
        "output-1",
        "output-2",
    ]
    assert captured[1] == 30