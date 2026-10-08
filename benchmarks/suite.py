from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig, get_profile
from .runner import REPOSITORY_ROOT

SUITE_SCHEMA_VERSION = 1
DEFAULT_SIZES = (
    10_000,
    25_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
)


def scenario_command(
    config: BenchmarkConfig,
    *,
    dataset_size: int,
    output_path: Path,
) -> list[str]:
    metadata_flag = "--metadata" if config.include_metadata else "--no-metadata"
    return [
        sys.executable,
        "-m",
        "benchmarks.runner",
        "--profile",
        config.profile,
        "--dataset-size",
        str(dataset_size),
        "--dimension",
        str(config.dimension),
        "--query-count",
        str(config.query_count),
        "--oracle-query-count",
        str(config.oracle_query_count),
        "--warmup-queries",
        str(config.warmup_queries),
        "--k",
        str(config.k),
        "--metric",
        config.metric,
        "--seed",
        str(config.seed),
        metadata_flag,
        "--output",
        str(output_path),
    ]


def run_suite(
    config: BenchmarkConfig,
    *,
    sizes: tuple[int, ...],
    output_directory: Path,
) -> dict[str, Any]:
    validate_sizes(sizes)
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "suite.json"
    scenarios: list[dict[str, Any]] = []
    suite_started = datetime.now(UTC)

    manifest: dict[str, Any] = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "created_at_utc": suite_started.isoformat(),
        "status": "running",
        "process_isolation": True,
        "page_cache_reset": False,
        "base_config": config.as_dict(),
        "sizes": list(sizes),
        "scenarios": scenarios,
    }
    write_manifest(manifest, manifest_path)

    for position, dataset_size in enumerate(sizes, start=1):
        report_path = output_directory / f"vectors-{dataset_size}.json"
        print(
            f"[{position}/{len(sizes)}] Starting isolated benchmark: "
            f"{dataset_size:,} vectors",
            flush=True,
        )
        completed = subprocess.run(
            scenario_command(
                config,
                dataset_size=dataset_size,
                output_path=report_path,
            ),
            cwd=REPOSITORY_ROOT,
            check=False,
        )

        if completed.returncode != 0:
            scenarios.append(
                {
                    "dataset_size": dataset_size,
                    "status": "failed",
                    "return_code": completed.returncode,
                    "report": report_path.name,
                }
            )
            manifest["status"] = "failed"
            manifest["failed_dataset_size"] = dataset_size
            write_manifest(manifest, manifest_path)
            raise RuntimeError(
                f"Benchmark failed for {dataset_size:,} vectors "
                f"with exit code {completed.returncode}"
            )

        report = load_report(report_path)
        scenarios.append(scenario_summary(report, report_path.name))
        write_manifest(manifest, manifest_path)

    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
    write_manifest(manifest, manifest_path)
    return manifest


def scenario_summary(
    report: dict[str, Any],
    report_name: str,
) -> dict[str, Any]:
    config = report["config"]
    results = report["results"]
    search = results["search"]
    persistence = results["persistence"]
    memory = results["memory"]
    correctness = results["correctness"]
    return {
        "dataset_size": config["dataset_size"],
        "status": "complete",
        "report": report_name,
        "ingestion_vectors_per_second": results["ingestion"][
            "vectors_per_second"
        ],
        "search_p50_ms": search["p50_ms"],
        "search_p95_ms": search["p95_ms"],
        "search_p99_ms": search["p99_ms"],
        "search_p99_9_ms": search["p99_9_ms"],
        "search_qps": search["qps"],
        "save_seconds": persistence["save_seconds"],
        "reopen_seconds": persistence["reopen_seconds"],
        "database_size_bytes": persistence["database_size_bytes"],
        "peak_rss_after_search_bytes": memory["peak_rss_after_search_bytes"],
        "mean_recall_at_k": correctness["mean_recall_at_k"],
    }


def load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"Invalid benchmark report: {path}")
    return data


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def default_output_directory() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPOSITORY_ROOT / "benchmarks" / "results" / f"suite-{timestamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Vectorch scaling suite in a fresh Python process per size."
        )
    )
    parser.add_argument("--profile", choices=("quick", "standard"), default="standard")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
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
    )
    parser.add_argument("--output-directory", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile = get_profile(args.profile)
    oracle_query_count = args.oracle_query_count
    if oracle_query_count is None and args.query_count is not None:
        oracle_query_count = min(profile.oracle_query_count, args.query_count)

    config = profile.with_overrides(
        dimension=args.dimension,
        query_count=args.query_count,
        oracle_query_count=oracle_query_count,
        warmup_queries=args.warmup_queries,
        k=args.k,
        metric=args.metric,
        seed=args.seed,
        include_metadata=args.metadata,
    )
    output_directory = args.output_directory or default_output_directory()
    manifest = run_suite(
        config,
        sizes=tuple(args.sizes),
        output_directory=output_directory,
    )

    print(f"Suite report: {output_directory / 'suite.json'}")
    print(f"{'vectors':>12} {'p50 ms':>12} {'p99 ms':>12} {'QPS':>12}")
    for scenario in manifest["scenarios"]:
        print(
            f"{scenario['dataset_size']:>12,} "
            f"{scenario['search_p50_ms']:>12.3f} "
            f"{scenario['search_p99_ms']:>12.3f} "
            f"{scenario['search_qps']:>12.2f}"
        )


def validate_sizes(sizes: tuple[int, ...]) -> None:
    if not sizes:
        raise ValueError("At least one dataset size is required")
    if any(isinstance(size, bool) or not isinstance(size, int) for size in sizes):
        raise TypeError("Dataset sizes must be integers")
    if any(size <= 0 for size in sizes):
        raise ValueError("Dataset sizes must be positive integers")
    if tuple(sorted(set(sizes))) != sizes:
        raise ValueError("Dataset sizes must be unique and ascending")


if __name__ == "__main__":
    main()
