from __future__ import annotations

import pytest

from src.inference.ablation import (
    AblationComparison,
    AblationSuite,
    format_ablation_comparison,
    format_ablation_suite,
    run_ablation,
)
from src.inference.benchmarks import (
    GenerationBenchmarkResult,
)


def make_result(
    name: str,
    tokens_per_second: float,
    elapsed_seconds: float,
    prompt_count: int = 8,
    generated_tokens: int = 200,
) -> GenerationBenchmarkResult:
    return GenerationBenchmarkResult(
        name=name,
        batch_size=1,
        prompt_count=prompt_count,
        generated_tokens=generated_tokens,
        elapsed_seconds=elapsed_seconds,
        tokens_per_second=tokens_per_second,
    )


def test_ablation_comparison_throughput_ratio() -> None:
    baseline = make_result(
        "baseline",
        tokens_per_second=50.0,
        elapsed_seconds=4.0,
    )

    alternative = make_result(
        "alternative",
        tokens_per_second=100.0,
        elapsed_seconds=2.0,
    )

    comparison = AblationComparison(
        baseline=baseline,
        alternative=alternative,
    )

    assert comparison.throughput_ratio == pytest.approx(2.0)


def test_ablation_comparison_throughput_difference() -> None:
    baseline = make_result(
        "baseline",
        tokens_per_second=50.0,
        elapsed_seconds=4.0,
    )

    alternative = make_result(
        "alternative",
        tokens_per_second=75.0,
        elapsed_seconds=2.6667,
    )

    comparison = AblationComparison(
        baseline=baseline,
        alternative=alternative,
    )

    assert comparison.throughput_difference_percent == pytest.approx(
        50.0
    )


def test_ablation_comparison_latency_ratio() -> None:
    baseline = make_result(
        "baseline",
        tokens_per_second=50.0,
        elapsed_seconds=4.0,
    )

    alternative = make_result(
        "alternative",
        tokens_per_second=100.0,
        elapsed_seconds=2.0,
    )

    comparison = AblationComparison(
        baseline=baseline,
        alternative=alternative,
    )

    assert comparison.latency_ratio == pytest.approx(0.5)


def test_ablation_comparison_latency_difference() -> None:
    baseline = make_result(
        "baseline",
        tokens_per_second=50.0,
        elapsed_seconds=4.0,
    )

    alternative = make_result(
        "alternative",
        tokens_per_second=100.0,
        elapsed_seconds=2.0,
    )

    comparison = AblationComparison(
        baseline=baseline,
        alternative=alternative,
    )

    assert comparison.latency_difference_percent == pytest.approx(
        -50.0
    )


def test_ablation_suite_uses_first_result_as_baseline() -> None:
    baseline = make_result(
        "baseline",
        50.0,
        4.0,
    )

    alternative = make_result(
        "alternative",
        75.0,
        2.666,
    )

    suite = AblationSuite(
        name="throughput",
        results=(baseline, alternative),
    )

    assert suite.baseline is baseline


def test_ablation_suite_compare() -> None:
    baseline = make_result(
        "baseline",
        50.0,
        4.0,
    )

    alternative = make_result(
        "alternative",
        100.0,
        2.0,
    )

    suite = AblationSuite(
        name="throughput",
        results=(baseline, alternative),
    )

    comparison = suite.compare(
        "alternative"
    )

    assert comparison.baseline is baseline
    assert comparison.alternative is alternative


def test_ablation_suite_rejects_empty_results() -> None:
    with pytest.raises(
        ValueError,
        match="results must not be empty",
    ):
        AblationSuite(
            name="test",
            results=(),
        )


def test_ablation_suite_rejects_empty_name() -> None:
    result = make_result(
        "baseline",
        50.0,
        4.0,
    )

    with pytest.raises(
        ValueError,
        match="name must not be empty",
    ):
        AblationSuite(
            name="",
            results=(result,),
        )


def test_ablation_suite_requires_same_prompt_count() -> None:
    baseline = make_result(
        "baseline",
        50.0,
        4.0,
        prompt_count=8,
    )

    alternative = make_result(
        "alternative",
        75.0,
        2.666,
        prompt_count=4,
    )

    with pytest.raises(
        ValueError,
        match="same prompt count",
    ):
        AblationSuite(
            name="test",
            results=(baseline, alternative),
        )


def test_ablation_suite_requires_same_generated_tokens() -> None:
    baseline = make_result(
        "baseline",
        50.0,
        4.0,
        generated_tokens=200,
    )

    alternative = make_result(
        "alternative",
        75.0,
        2.666,
        generated_tokens=100,
    )

    with pytest.raises(
        ValueError,
        match="same number of tokens",
    ):
        AblationSuite(
            name="test",
            results=(baseline, alternative),
        )


def test_ablation_suite_rejects_unknown_variant() -> None:
    baseline = make_result(
        "baseline",
        50.0,
        4.0,
    )

    suite = AblationSuite(
        name="test",
        results=(baseline,),
    )

    with pytest.raises(
        ValueError,
        match="Unknown ablation result",
    ):
        suite.compare("missing")


def test_run_ablation_executes_all_variants() -> None:
    calls: list[str] = []

    def baseline() -> GenerationBenchmarkResult:
        calls.append("baseline")
        return make_result(
            "baseline",
            50.0,
            4.0,
        )

    def alternative() -> GenerationBenchmarkResult:
        calls.append("alternative")
        return make_result(
            "alternative",
            100.0,
            2.0,
        )

    suite = run_ablation(
        name="cache",
        variants=(
            ("baseline", baseline),
            ("alternative", alternative),
        ),
    )

    assert calls == [
        "baseline",
        "alternative",
    ]

    assert len(suite.results) == 2


def test_run_ablation_rejects_duplicate_names() -> None:
    def benchmark() -> GenerationBenchmarkResult:
        return make_result(
            "same",
            50.0,
            4.0,
        )

    with pytest.raises(
        ValueError,
        match="variant names must be unique",
    ):
        run_ablation(
            name="test",
            variants=(
                ("same", benchmark),
                ("same", benchmark),
            ),
        )


def test_run_ablation_requires_matching_result_name() -> None:
    def benchmark() -> GenerationBenchmarkResult:
        return make_result(
            "wrong_name",
            50.0,
            4.0,
        )

    with pytest.raises(
        ValueError,
        match="does not match variant name",
    ):
        run_ablation(
            name="test",
            variants=(
                ("expected_name", benchmark),
            ),
        )


def test_run_ablation_rejects_empty_variants() -> None:
    with pytest.raises(
        ValueError,
        match="variants must not be empty",
    ):
        run_ablation(
            name="test",
            variants=(),
        )


def test_format_ablation_comparison() -> None:
    baseline = make_result(
        "baseline",
        50.0,
        4.0,
    )

    alternative = make_result(
        "alternative",
        100.0,
        2.0,
    )

    comparison = AblationComparison(
        baseline=baseline,
        alternative=alternative,
    )

    output = format_ablation_comparison(
        comparison
    )

    assert "Baseline: baseline" in output
    assert "Alternative: alternative" in output
    assert "2.000x" in output
    assert "+100.00%" in output
    assert "-50.00%" in output


def test_format_ablation_suite() -> None:
    baseline = make_result(
        "baseline",
        50.0,
        4.0,
    )

    alternative = make_result(
        "alternative",
        100.0,
        2.0,
    )

    suite = AblationSuite(
        name="cache",
        results=(baseline, alternative),
    )

    output = format_ablation_suite(
        suite
    )

    assert "ABLATION: cache" in output
    assert "Variant: baseline" in output
    assert "Variant: alternative" in output
    assert "COMPARISONS" in output


def test_zero_baseline_throughput_is_safe() -> None:
    baseline = make_result(
        "baseline",
        0.0,
        4.0,
    )

    alternative = make_result(
        "alternative",
        100.0,
        2.0,
    )

    comparison = AblationComparison(
        baseline=baseline,
        alternative=alternative,
    )

    assert comparison.throughput_ratio == 0.0
    assert comparison.throughput_difference_percent == 0.0


def test_zero_baseline_latency_is_safe() -> None:
    baseline = make_result(
        "baseline",
        100.0,
        0.0,
    )

    alternative = make_result(
        "alternative",
        100.0,
        2.0,
    )

    comparison = AblationComparison(
        baseline=baseline,
        alternative=alternative,
    )

    assert comparison.latency_ratio == 0.0
    assert comparison.latency_difference_percent == 0.0