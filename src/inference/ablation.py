from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from src.inference.benchmarks import GenerationBenchmarkResult


@dataclass(frozen=True)
class AblationComparison:
    """
    Comparison between a baseline and an alternative benchmark.

    The baseline is treated only as the reference measurement.
    No configuration is considered intrinsically better; the
    comparison simply reports measurable differences.
    """

    baseline: GenerationBenchmarkResult
    alternative: GenerationBenchmarkResult

    @property
    def throughput_ratio(self) -> float:
        """
        Alternative throughput divided by baseline throughput.

        A value above 1.0 means the alternative produced more
        generated tokens per second during the measurement.
        """

        if self.baseline.tokens_per_second <= 0.0:
            return 0.0

        return (
            self.alternative.tokens_per_second
            / self.baseline.tokens_per_second
        )

    @property
    def throughput_difference_percent(self) -> float:
        """
        Percentage difference in throughput relative to baseline.
        """

        if self.baseline.tokens_per_second <= 0.0:
            return 0.0

        return (
            (
                self.alternative.tokens_per_second
                - self.baseline.tokens_per_second
            )
            / self.baseline.tokens_per_second
        ) * 100.0

    @property
    def latency_ratio(self) -> float:
        """
        Alternative average latency divided by baseline latency.
        """

        baseline_latency = (
            self.baseline.average_latency_seconds
        )

        if baseline_latency <= 0.0:
            return 0.0

        return (
            self.alternative.average_latency_seconds
            / baseline_latency
        )

    @property
    def latency_difference_percent(self) -> float:
        """
        Percentage difference in average latency relative to baseline.
        """

        baseline_latency = (
            self.baseline.average_latency_seconds
        )

        if baseline_latency <= 0.0:
            return 0.0

        return (
            (
                self.alternative.average_latency_seconds
                - baseline_latency
            )
            / baseline_latency
        ) * 100.0


@dataclass(frozen=True)
class AblationSuite:
    """
    Collection of benchmark measurements for one experimental
    dimension.

    Example dimensions:

        cache:
            cache
            no_cache

        batching:
            batch_1
            batch_2
            batch_4
            batch_8

        decoding:
            greedy
            sampling
    """

    name: str
    results: tuple[GenerationBenchmarkResult, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "name must not be empty."
            )

        if not self.results:
            raise ValueError(
                "results must not be empty."
            )

        prompt_count = self.results[0].prompt_count
        generated_tokens = self.results[0].generated_tokens

        for result in self.results:
            if result.prompt_count != prompt_count:
                raise ValueError(
                    "All ablation results must use the same "
                    "prompt count."
                )

            if result.generated_tokens != generated_tokens:
                raise ValueError(
                    "All ablation results must generate the "
                    "same number of tokens."
                )

    @property
    def baseline(self) -> GenerationBenchmarkResult:
        """Return the first result as the reference measurement."""

        return self.results[0]

    def compare(
        self,
        alternative_name: str,
    ) -> AblationComparison:
        """Compare one result against the baseline."""

        for result in self.results[1:]:
            if result.name == alternative_name:
                return AblationComparison(
                    baseline=self.baseline,
                    alternative=result,
                )

        raise ValueError(
            f"Unknown ablation result: {alternative_name}"
        )


def run_ablation(
    *,
    name: str,
    variants: Sequence[
        tuple[
            str,
            Callable[[], GenerationBenchmarkResult],
        ]
    ],
) -> AblationSuite:
    """
    Execute a collection of controlled benchmark variants.

    Each variant must return a GenerationBenchmarkResult.

    The first variant becomes the baseline/reference.
    """

    if not name.strip():
        raise ValueError(
            "name must not be empty."
        )

    if not variants:
        raise ValueError(
            "variants must not be empty."
        )

    variant_names = [
        variant_name
        for variant_name, _ in variants
    ]

    if any(
        not variant_name.strip()
        for variant_name in variant_names
    ):
        raise ValueError(
            "variant names must not be empty."
        )

    if len(set(variant_names)) != len(variant_names):
        raise ValueError(
            "variant names must be unique."
        )

    results: list[GenerationBenchmarkResult] = []

    for variant_name, benchmark_fn in variants:
        if not callable(benchmark_fn):
            raise ValueError(
                f"Benchmark function for '{variant_name}' "
                "must be callable."
            )

        result = benchmark_fn()

        if result.name != variant_name:
            raise ValueError(
                f"Benchmark result name '{result.name}' "
                f"does not match variant name "
                f"'{variant_name}'."
            )

        results.append(result)

    return AblationSuite(
        name=name,
        results=tuple(results),
    )


def format_ablation_comparison(
    comparison: AblationComparison,
) -> str:
    """
    Format a human-readable baseline-vs-alternative comparison.
    """

    baseline = comparison.baseline
    alternative = comparison.alternative

    throughput_ratio = (
        comparison.throughput_ratio
    )

    throughput_difference = (
        comparison.throughput_difference_percent
    )

    latency_ratio = (
        comparison.latency_ratio
    )

    latency_difference = (
        comparison.latency_difference_percent
    )

    return "\n".join(
        [
            f"Baseline: {baseline.name}",
            (
                f"  Throughput: "
                f"{baseline.tokens_per_second:.2f} tok/s"
            ),
            (
                f"  Latency/prompt: "
                f"{baseline.average_latency_seconds:.3f}s"
            ),
            "",
            f"Alternative: {alternative.name}",
            (
                f"  Throughput: "
                f"{alternative.tokens_per_second:.2f} tok/s"
            ),
            (
                f"  Latency/prompt: "
                f"{alternative.average_latency_seconds:.3f}s"
            ),
            "",
            (
                f"Throughput ratio: "
                f"{throughput_ratio:.3f}x"
            ),
            (
                f"Throughput difference: "
                f"{throughput_difference:+.2f}%"
            ),
            (
                f"Latency ratio: "
                f"{latency_ratio:.3f}x"
            ),
            (
                f"Latency difference: "
                f"{latency_difference:+.2f}%"
            ),
        ]
    )


def format_ablation_suite(
    suite: AblationSuite,
) -> str:
    """
    Format the complete ablation suite.
    """

    lines = [
        "=" * 72,
        f"ABLATION: {suite.name}",
        "=" * 72,
        "",
    ]

    for result in suite.results:
        lines.extend(
            [
                f"Variant: {result.name}",
                (
                    f"  Batch size: "
                    f"{result.batch_size}"
                ),
                (
                    f"  Prompts: "
                    f"{result.prompt_count}"
                ),
                (
                    f"  Generated tokens: "
                    f"{result.generated_tokens}"
                ),
                (
                    f"  Elapsed: "
                    f"{result.elapsed_seconds:.3f}s"
                ),
                (
                    f"  Throughput: "
                    f"{result.tokens_per_second:.2f} tok/s"
                ),
                (
                    f"  Latency/prompt: "
                    f"{result.average_latency_seconds:.3f}s"
                ),
                "",
            ]
        )

    if len(suite.results) > 1:
        lines.append("-" * 72)
        lines.append("COMPARISONS")
        lines.append("-" * 72)
        lines.append("")

        for result in suite.results[1:]:
            comparison = suite.compare(
                result.name
            )

            lines.append(
                f"{suite.baseline.name} vs "
                f"{result.name}"
            )
            lines.append(
                format_ablation_comparison(
                    comparison
                )
            )
            lines.append("")

    return "\n".join(lines)