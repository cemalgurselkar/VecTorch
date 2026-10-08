from __future__ import annotations

import argparse
import json
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vectorch import Vectorch

from .config import BenchmarkConfig, get_profile
from .dataset import generate_dataset
from .metrics import (
    correctness_summary,
    exact_top_k,
    latency_summary,
    score_error_summary,
)
from .system import (
    cpu_utilization_percent,
    current_rss_bytes,
    directory_size_bytes,
    environment_info,
    peak_rss_bytes,
)

REPORT_SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    benchmark_started = time.perf_counter()
    benchmark_cpu_started = time.process_time()
    rss_at_start = current_rss_bytes()

    dataset_started = time.perf_counter()
    dataset = generate_dataset(config)
    dataset_seconds = time.perf_counter() - dataset_started
    rss_after_dataset = current_rss_bytes()

    with tempfile.TemporaryDirectory(prefix="vectorch-benchmark-") as temporary:
        database_path = Path(temporary) / "database"
        database = Vectorch(database_path)
        collection = database.create_collection(
            "benchmark",
            dimension=config.dimension,
            metric=config.metric,
            index="flat",
        )

        ingestion_started = time.perf_counter()
        ingestion_cpu_started = time.process_time()
        for internal_id in range(config.dataset_size):
            collection.add(
                internal_id,
                dataset.vectors[internal_id],
                dataset.metadata[internal_id],
            )
        ingestion_seconds = time.perf_counter() - ingestion_started
        ingestion_cpu_seconds = time.process_time() - ingestion_cpu_started
        rss_after_ingestion = current_rss_bytes()

        save_started = time.perf_counter()
        collection.save()
        save_seconds = time.perf_counter() - save_started
        database.close()

        database_size = directory_size_bytes(database_path)

        reopen_started = time.perf_counter()
        reopened = Vectorch(database_path)
        collection = reopened.get_collection("benchmark")
        reopen_seconds = time.perf_counter() - reopen_started
        rss_after_reopen = current_rss_bytes()

        try:
            cold_started = time.perf_counter()
            collection.search(dataset.queries[0], config.k)
            cold_query_seconds = time.perf_counter() - cold_started

            for index in range(config.warmup_queries):
                query = dataset.queries[index % config.query_count]
                collection.search(query, config.k)

            latencies: list[float] = []
            actual_ids: list[list[int]] = []
            actual_scores: list[list[float]] = []
            search_started = time.perf_counter()
            search_cpu_started = time.process_time()

            for query in dataset.queries:
                query_started = time.perf_counter()
                results = collection.search(query, config.k)
                latencies.append(time.perf_counter() - query_started)
                actual_ids.append([int(result.id) for result in results])
                actual_scores.append([result.score for result in results])

            search_seconds = time.perf_counter() - search_started
            search_cpu_seconds = time.process_time() - search_cpu_started
            peak_rss_after_search = peak_rss_bytes()

            oracle_started = time.perf_counter()
            expected_ids: list[list[int]] = []
            expected_scores: list[list[float]] = []
            oracle_queries = dataset.queries[: config.oracle_query_count]
            for query in oracle_queries:
                ids, scores = exact_top_k(
                    query,
                    dataset.vectors,
                    metric=config.metric,
                    k=config.k,
                )
                expected_ids.append(ids.tolist())
                expected_scores.append(scores.tolist())
            oracle_seconds = time.perf_counter() - oracle_started
        finally:
            reopened.close()

    total_seconds = time.perf_counter() - benchmark_started
    total_cpu_seconds = time.process_time() - benchmark_cpu_started
    search_summary = latency_summary(latencies)
    search_summary.update(
        {
            "cold_query_ms": cold_query_seconds * 1_000,
            "measured_seconds": search_seconds,
            "qps": config.query_count / search_seconds,
            "warmup_query_count": config.warmup_queries,
            "measured_query_count": config.query_count,
        }
    )
    oracle_actual_ids = actual_ids[: config.oracle_query_count]
    oracle_actual_scores = actual_scores[: config.oracle_query_count]
    correctness = correctness_summary(expected_ids, oracle_actual_ids)
    correctness.update(
        score_error_summary(
            expected_ids,
            expected_scores,
            oracle_actual_ids,
            oracle_actual_scores,
        )
    )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "environment": environment_info(REPOSITORY_ROOT),
        "config": config.as_dict(),
        "results": {
            "dataset_generation_seconds": dataset_seconds,
            "ingestion": {
                "seconds": ingestion_seconds,
                "cpu_seconds": ingestion_cpu_seconds,
                "cpu_utilization_percent": cpu_utilization_percent(
                    ingestion_cpu_seconds,
                    ingestion_seconds,
                ),
                "vectors_per_second": config.dataset_size / ingestion_seconds,
            },
            "index_build": {
                "seconds": 0.0,
                "note": "FlatIndex has no separate build phase",
            },
            "persistence": {
                "save_seconds": save_seconds,
                "reopen_seconds": reopen_seconds,
                "database_size_bytes": database_size,
            },
            "search": search_summary,
            "correctness": correctness,
            "oracle_seconds": oracle_seconds,
            "memory": {
                "rss_at_start_bytes": rss_at_start,
                "rss_after_dataset_bytes": rss_after_dataset,
                "rss_after_ingestion_bytes": rss_after_ingestion,
                "rss_after_reopen_bytes": rss_after_reopen,
                "peak_rss_after_search_bytes": peak_rss_after_search,
                "peak_rss_after_oracle_bytes": peak_rss_bytes(),
            },
            "cpu": {
                "total_cpu_seconds": total_cpu_seconds,
                "total_cpu_utilization_percent": cpu_utilization_percent(
                    total_cpu_seconds,
                    total_seconds,
                ),
                "search_cpu_seconds": search_cpu_seconds,
                "search_cpu_utilization_percent": cpu_utilization_percent(
                    search_cpu_seconds,
                    search_seconds,
                ),
            },
            "total_seconds": total_seconds,
        },
    }


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def default_report_path(profile: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPOSITORY_ROOT / "benchmarks" / "results" / f"{profile}-{timestamp}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the reproducible Vectorch FlatIndex baseline benchmark."
    )
    parser.add_argument("--profile", choices=("quick", "standard"), default="quick")
    parser.add_argument("--dataset-size", type=int)
    parser.add_argument("--dimension", type=int)
    parser.add_argument("--query-count", type=int)
    parser.add_argument("--oracle-query-count", type=int)
    parser.add_argument("--warmup-queries", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--metric", choices=("cosine", "dot", "l2"))
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--metadata",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include representative JSON metadata during ingestion.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile = get_profile(args.profile)
    oracle_query_count = args.oracle_query_count
    if oracle_query_count is None and args.query_count is not None:
        oracle_query_count = min(profile.oracle_query_count, args.query_count)

    config = profile.with_overrides(
        dataset_size=args.dataset_size,
        dimension=args.dimension,
        query_count=args.query_count,
        oracle_query_count=oracle_query_count,
        warmup_queries=args.warmup_queries,
        k=args.k,
        metric=args.metric,
        seed=args.seed,
        include_metadata=args.metadata,
    )
    output_path = args.output or default_report_path(config.profile)
    report = run_benchmark(config)
    write_report(report, output_path)

    results = report["results"]
    search = results["search"]
    correctness = results["correctness"]
    print(f"Report: {output_path}")
    print(f"Insert throughput: {results['ingestion']['vectors_per_second']:.2f} vec/s")
    print(f"Search p50/p95/p99: {search['p50_ms']:.3f} / {search['p95_ms']:.3f} / {search['p99_ms']:.3f} ms")
    print(f"Search QPS: {search['qps']:.2f}")
    print(f"Mean Recall@K: {correctness['mean_recall_at_k']:.6f}")


if __name__ == "__main__":
    main()
