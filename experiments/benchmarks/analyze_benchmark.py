from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "benchmarks"
    / "baseline_results.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "benchmarks"
    / "benchmark_analysis.json"
)


def load_results() -> dict:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark results not found: {RESULTS_PATH}"
        )

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Benchmark results must be a JSON object.")

    if "model" not in data:
        raise KeyError("Missing 'model' section.")

    if "summary" not in data:
        raise KeyError("Missing 'summary' section.")

    if "results" not in data:
        raise KeyError("Missing 'results' section.")

    return data


def tokenize_text(text: str) -> list[str]:
    """Tokenization used only for degeneration analysis."""
    return re.findall(
        r"\w+|[^\w\s]",
        text.lower(),
        flags=re.UNICODE,
    )


def repetition_ratio(tokens: list[str]) -> float:
    """
    Fraction of tokens that are repeated beyond
    their first occurrence.
    """
    if not tokens:
        return 0.0

    counts = Counter(tokens)

    repeated_tokens = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    return repeated_tokens / len(tokens)


def unique_token_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0

    return len(set(tokens)) / len(tokens)


def analyze_prompt(result: dict) -> dict:
    output = result.get("output", "")
    analysis_tokens = tokenize_text(output)

    return {
        "id": result["id"],
        "category": result["category"],
        "generated_tokens": int(result["generated_tokens"]),
        "generation_seconds": float(
            result["generation_seconds"]
        ),
        "tokens_per_second": float(
            result["tokens_per_second"]
        ),
        "repetition_ratio": repetition_ratio(
            analysis_tokens
        ),
        "unique_token_ratio": unique_token_ratio(
            analysis_tokens
        ),
    }


def aggregate_categories(
    results: list[dict],
) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)

    for result in results:
        grouped[result["category"]].append(result)

    categories = {}

    for category, items in sorted(grouped.items()):
        count = len(items)

        total_tokens = sum(
            item["generated_tokens"]
            for item in items
        )

        total_seconds = sum(
            item["generation_seconds"]
            for item in items
        )

        categories[category] = {
            "prompts": count,
            "generated_tokens": total_tokens,
            "generation_seconds": total_seconds,
            "average_generated_tokens": (
                total_tokens / count
            ),
            "average_generation_seconds": (
                total_seconds / count
            ),
            "average_tokens_per_second": (
                sum(
                    item["tokens_per_second"]
                    for item in items
                )
                / count
            ),
            "average_repetition_ratio": (
                sum(
                    item["repetition_ratio"]
                    for item in items
                )
                / count
            ),
            "average_unique_token_ratio": (
                sum(
                    item["unique_token_ratio"]
                    for item in items
                )
                / count
            ),
        }

    return categories


def main() -> None:
    data = load_results()

    model = data["model"]
    summary = data["summary"]

    analyzed_results = [
        analyze_prompt(result)
        for result in data["results"]
    ]

    categories = aggregate_categories(
        analyzed_results
    )

    total_tokens = sum(
        result["generated_tokens"]
        for result in analyzed_results
    )

    total_seconds = sum(
        result["generation_seconds"]
        for result in analyzed_results
    )

    overall_throughput = (
        total_tokens / total_seconds
        if total_seconds > 0
        else 0.0
    )

    average_repetition = (
        sum(
            result["repetition_ratio"]
            for result in analyzed_results
        )
        / len(analyzed_results)
    )

    average_unique = (
        sum(
            result["unique_token_ratio"]
            for result in analyzed_results
        )
        / len(analyzed_results)
    )

    report = {
        "model": {
            "name": model["name"],
            "checkpoint": model["checkpoint"],
            "parameters": model["parameters"],
        },
        "generation_config": data.get(
            "generation",
            {},
        ),
        "benchmark": {
            "prompt_count": len(analyzed_results),
            "total_generated_tokens": total_tokens,
            "total_generation_seconds": total_seconds,
            "overall_tokens_per_second": (
                overall_throughput
            ),
            "average_repetition_ratio": (
                average_repetition
            ),
            "average_unique_token_ratio": (
                average_unique
            ),
        },
        "categories": categories,
        "prompts": analyzed_results,
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print("=" * 70)
    print("MUJAHID-REASONER BENCHMARK ANALYSIS")
    print("=" * 70)

    print(
        f"Model                : {model['name']}"
    )
    print(
        f"Parameters           : {model['parameters']:,}"
    )
    print(
        f"Prompts              : {summary['prompt_count']}"
    )
    print(
        f"Generated tokens     : {total_tokens}"
    )
    print(
        f"Generation time      : {total_seconds:.2f}s"
    )
    print(
        f"Overall throughput   : "
        f"{overall_throughput:.2f} tok/s"
    )
    print(
        f"Avg repetition       : "
        f"{average_repetition:.2%}"
    )
    print(
        f"Avg unique tokens    : "
        f"{average_unique:.2%}"
    )

    print("-" * 70)
    print("CATEGORY RESULTS")
    print("-" * 70)

    for category, metrics in categories.items():
        print(
            f"{category:<12} "
            f"tokens={metrics['generated_tokens']:<4} "
            f"time={metrics['generation_seconds']:.2f}s  "
            f"speed={metrics['average_tokens_per_second']:.2f} tok/s  "
            f"repetition={metrics['average_repetition_ratio']:.2%}  "
            f"unique={metrics['average_unique_token_ratio']:.2%}"
        )

    print("-" * 70)
    print(
        f"Saved: {REPORT_PATH}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()