import json

import numpy as np
import pytest

from benchmarks.compare import compare_reports
from benchmarks.compare_suite import compare_suites
from benchmarks.config import BenchmarkConfig, get_profile
from benchmarks.dataset import generate_dataset
from benchmarks.metrics import (
    correctness_summary,
    exact_top_k,
    latency_summary,
    recall_at_k,
    score_error_summary,
)
from benchmarks.runner import run_benchmark, write_report
from benchmarks.suite import DEFAULT_SIZES, scenario_command, validate_sizes


def tiny_config(**overrides):
    config = BenchmarkConfig(
        profile="test",
        dataset_size=32,
        dimension=8,
        query_count=5,
        oracle_query_count=5,
        warmup_queries=2,
        k=3,
        metric="cosine",
        seed=123,
        include_metadata=True,
    )
    return config.with_overrides(**overrides)


def test_profiles_are_explicit_and_valid():
    assert get_profile("quick").profile == "quick"
    assert get_profile("standard").dataset_size > get_profile("quick").dataset_size

    with pytest.raises(ValueError, match="Unknown profile"):
        get_profile("missing")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_size", 0),
        ("dimension", -1),
        ("query_count", 0),
        ("oracle_query_count", 0),
        ("warmup_queries", -1),
        ("k", 0),
        ("metric", "invalid"),
        ("seed", True),
        ("include_metadata", "yes"),
    ],
)
def test_config_rejects_invalid_values(field, value):
    with pytest.raises((TypeError, ValueError)):
        tiny_config(**{field: value})


def test_dataset_generation_is_reproducible_and_contiguous():
    first = generate_dataset(tiny_config())
    second = generate_dataset(tiny_config())

    np.testing.assert_array_equal(first.vectors, second.vectors)
    np.testing.assert_array_equal(first.queries, second.queries)
    assert first.metadata == second.metadata
    assert first.vectors.dtype == np.float32
    assert first.queries.dtype == np.float32
    assert first.vectors.flags.c_contiguous
    assert first.queries.flags.c_contiguous


def test_queries_are_identical_across_scaling_sizes():
    small = generate_dataset(tiny_config(dataset_size=16))
    large = generate_dataset(tiny_config(dataset_size=32))

    np.testing.assert_array_equal(small.vectors, large.vectors[:16])
    np.testing.assert_array_equal(small.queries, large.queries)


def test_latency_summary_uses_milliseconds():
    summary = latency_summary([0.001, 0.002, 0.003])

    assert summary["min_ms"] == pytest.approx(1.0)
    assert summary["mean_ms"] == pytest.approx(2.0)
    assert summary["p50_ms"] == pytest.approx(2.0)
    assert summary["max_ms"] == pytest.approx(3.0)


@pytest.mark.parametrize("metric", ["cosine", "dot", "l2"])
def test_exact_oracle_returns_top_k(metric):
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=np.float32,
    )
    query = np.array([1.0, 0.0], dtype=np.float32)

    ids, scores = exact_top_k(query, vectors, metric=metric, k=2)

    assert ids.tolist()[0] == 0
    assert len(ids) == 2
    assert len(scores) == 2


def test_correctness_metrics():
    assert recall_at_k([1, 2], [2, 3]) == 0.5
    summary = correctness_summary([[1, 2], [3, 4]], [[2, 1], [3, 5]])

    assert summary["mean_recall_at_k"] == 0.75
    assert summary["minimum_recall_at_k"] == 0.5
    assert summary["exact_set_match_rate"] == 0.5

    score_summary = score_error_summary(
        [[1, 2]],
        [[0.75, 0.5]],
        [[2, 1]],
        [[0.5, 0.75]],
    )
    assert score_summary["compared_score_count"] == 2
    assert score_summary["maximum_absolute_score_error"] == 0.0


def test_tiny_benchmark_report_has_expected_schema(tmp_path):
    report = run_benchmark(tiny_config())
    output = tmp_path / "report.json"
    write_report(report, output)
    restored = json.loads(output.read_text(encoding="utf-8"))

    assert restored["schema_version"] == 1
    assert restored["config"]["seed"] == 123
    assert restored["results"]["ingestion"]["vectors_per_second"] > 0
    assert restored["results"]["search"]["p99_ms"] > 0
    assert restored["results"]["correctness"]["mean_recall_at_k"] == 1.0
    assert restored["results"]["correctness"]["maximum_absolute_score_error"] == 0.0
    assert restored["results"]["persistence"]["database_size_bytes"] > 0


def test_report_comparison_respects_metric_direction():
    baseline = {
        "schema_version": 1,
        "results": {
            "ingestion": {"vectors_per_second": 100.0},
            "search": {"qps": 10.0, "p50_ms": 10.0, "p95_ms": 20.0, "p99_ms": 30.0},
            "persistence": {
                "save_seconds": 2.0,
                "reopen_seconds": 1.0,
                "database_size_bytes": 1_000,
            },
            "memory": {"peak_rss_after_search_bytes": 2_000},
        },
    }
    candidate = {
        "schema_version": 1,
        "results": {
            "ingestion": {"vectors_per_second": 120.0},
            "search": {"qps": 12.0, "p50_ms": 8.0, "p95_ms": 16.0, "p99_ms": 24.0},
            "persistence": {
                "save_seconds": 1.6,
                "reopen_seconds": 0.8,
                "database_size_bytes": 800,
            },
            "memory": {"peak_rss_after_search_bytes": 1_600},
        },
    }

    comparisons = compare_reports(baseline, candidate)

    assert all(item["improvement_percent"] == pytest.approx(20.0) for item in comparisons)


def test_scaling_suite_defaults_are_unique_and_ascending():
    validate_sizes(DEFAULT_SIZES)
    assert DEFAULT_SIZES == (
        10_000,
        25_000,
        50_000,
        100_000,
        250_000,
        500_000,
        1_000_000,
    )


def test_scaling_suite_uses_isolated_runner_command(tmp_path):
    config = tiny_config(include_metadata=False)
    output = tmp_path / "vectors-10000.json"

    command = scenario_command(
        config,
        dataset_size=10_000,
        output_path=output,
    )

    assert command[1:3] == ["-m", "benchmarks.runner"]
    assert command[command.index("--dataset-size") + 1] == "10000"
    assert command[command.index("--output") + 1] == str(output)
    assert "--no-metadata" in command


@pytest.mark.parametrize(
    "sizes",
    [(), (25_000, 10_000), (10_000, 10_000), (0, 10_000)],
)
def test_scaling_suite_rejects_invalid_sizes(sizes):
    with pytest.raises(ValueError):
        validate_sizes(sizes)


def test_suite_comparison_reports_improvement_for_each_size():
    def suite_with(p50, p95, p99, qps, ingestion, save, reopen, rss):
        return {
            "schema_version": 1,
            "status": "complete",
            "base_config": {
                "dimension": 128,
                "query_count": 100,
                "oracle_query_count": 10,
                "warmup_queries": 10,
                "k": 10,
                "metric": "cosine",
                "seed": 42,
                "include_metadata": False,
            },
            "scenarios": [
                {
                    "dataset_size": 10_000,
                    "status": "complete",
                    "search_p50_ms": p50,
                    "search_p95_ms": p95,
                    "search_p99_ms": p99,
                    "search_qps": qps,
                    "ingestion_vectors_per_second": ingestion,
                    "save_seconds": save,
                    "reopen_seconds": reopen,
                    "peak_rss_after_search_bytes": rss,
                }
            ],
        }

    baseline = suite_with(10, 20, 30, 100, 1_000, 2, 1, 2_000)
    candidate = suite_with(8, 16, 24, 120, 1_200, 1.6, 0.8, 1_600)

    rows = compare_suites(baseline, candidate)

    assert rows[0]["dataset_size"] == 10_000
    assert all(
        value == pytest.approx(20.0)
        for value in rows[0]["improvement_percent"].values()
    )
