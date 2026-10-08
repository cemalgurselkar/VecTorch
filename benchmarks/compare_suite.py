from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SuiteMetric:
    name: str
    higher_is_better: bool


METRICS = (
    SuiteMetric("search_p50_ms", False),
    SuiteMetric("search_p95_ms", False),
    SuiteMetric("search_p99_ms", False),
    SuiteMetric("search_qps", True),
    SuiteMetric("ingestion_vectors_per_second", True),
    SuiteMetric("save_seconds", False),
    SuiteMetric("reopen_seconds", False),
    SuiteMetric("peak_rss_after_search_bytes", False),
)


def load_suite(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"Invalid suite report: {path}")
    if data.get("status") != "complete":
        raise ValueError(f"Suite report is not complete: {path}")
    return data


def compare_suites(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    _validate_comparable_configs(baseline, candidate)
    baseline_scenarios = _scenarios_by_size(baseline)
    candidate_scenarios = _scenarios_by_size(candidate)
    if baseline_scenarios.keys() != candidate_scenarios.keys():
        raise ValueError("Suite reports must contain the same dataset sizes")

    rows: list[dict[str, Any]] = []
    for dataset_size in baseline_scenarios:
        before = baseline_scenarios[dataset_size]
        after = candidate_scenarios[dataset_size]
        changes: dict[str, float | None] = {}
        for metric in METRICS:
            baseline_value = _numeric_value(before, metric.name)
            candidate_value = _numeric_value(after, metric.name)
            if baseline_value in {None, 0} or candidate_value is None:
                improvement = None
            else:
                change = ((candidate_value - baseline_value) / baseline_value) * 100
                improvement = change if metric.higher_is_better else -change
            changes[metric.name] = improvement

        rows.append(
            {
                "dataset_size": dataset_size,
                "improvement_percent": changes,
            }
        )
    return rows


def _validate_comparable_configs(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    fields = (
        "dimension",
        "query_count",
        "oracle_query_count",
        "warmup_queries",
        "k",
        "metric",
        "seed",
        "include_metadata",
    )
    baseline_config = baseline.get("base_config")
    candidate_config = candidate.get("base_config")
    if not isinstance(baseline_config, dict) or not isinstance(
        candidate_config,
        dict,
    ):
        raise TypeError("Suite reports must contain base_config objects")

    mismatches = [
        field
        for field in fields
        if baseline_config.get(field) != candidate_config.get(field)
    ]
    if mismatches:
        raise ValueError(
            "Suite configurations differ for: " + ", ".join(mismatches)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two completed Vectorch scaling suite manifests."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = compare_suites(
        load_suite(args.baseline),
        load_suite(args.candidate),
    )

    print(f"{'vectors':>12} {'p50':>10} {'p99':>10} {'QPS':>10} {'insert':>10}")
    for row in rows:
        changes = row["improvement_percent"]
        print(
            f"{row['dataset_size']:>12,} "
            f"{_format_percent(changes['search_p50_ms']):>10} "
            f"{_format_percent(changes['search_p99_ms']):>10} "
            f"{_format_percent(changes['search_qps']):>10} "
            f"{_format_percent(changes['ingestion_vectors_per_second']):>10}"
        )


def _scenarios_by_size(suite: dict[str, Any]) -> dict[int, dict[str, Any]]:
    scenarios = suite.get("scenarios")
    if not isinstance(scenarios, list):
        raise TypeError("Suite scenarios must be a list")

    result: dict[int, dict[str, Any]] = {}
    for scenario in scenarios:
        if not isinstance(scenario, dict) or scenario.get("status") != "complete":
            raise ValueError("Suite contains an incomplete scenario")
        dataset_size = scenario.get("dataset_size")
        if isinstance(dataset_size, bool) or not isinstance(dataset_size, int):
            raise TypeError("Scenario dataset_size must be an integer")
        if dataset_size in result:
            raise ValueError(f"Duplicate suite dataset size: {dataset_size}")
        result[dataset_size] = scenario
    return result


def _numeric_value(scenario: dict[str, Any], name: str) -> float | None:
    value = scenario.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Suite metric {name!r} must be numeric")
    return float(value)


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}%"


if __name__ == "__main__":
    main()
