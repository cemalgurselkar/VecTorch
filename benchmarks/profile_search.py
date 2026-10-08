from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from vectorch import Vectorch

from .config import get_profile
from .dataset import generate_dataset
from .runner import REPOSITORY_ROOT
from .system import environment_info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile the Vectorch search hot path with cProfile."
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--text-output", type=Path)
    parser.add_argument("--limit", type=int, default=40)
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
    )
    dataset = generate_dataset(config)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or (
        REPOSITORY_ROOT / "benchmarks" / "profiles" / f"search-{timestamp}.prof"
    )
    text_output = args.text_output or output.with_suffix(".txt")
    manifest_output = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    text_output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="vectorch-profile-") as temporary:
        database = Vectorch(Path(temporary) / "database")
        collection = database.create_collection(
            "profile",
            dimension=config.dimension,
            metric=config.metric,
        )
        for internal_id in range(config.dataset_size):
            collection.add(
                internal_id,
                dataset.vectors[internal_id],
                dataset.metadata[internal_id],
            )

        for index in range(config.warmup_queries):
            collection.search(
                dataset.queries[index % config.query_count],
                config.k,
            )

        profiler = cProfile.Profile()
        profiler.enable()
        for query in dataset.queries:
            collection.search(query, config.k)
        profiler.disable()
        profiler.dump_stats(output)
        database.close()

    stream = io.StringIO()
    statistics = pstats.Stats(profiler, stream=stream)
    statistics.strip_dirs().sort_stats("cumulative").print_stats(args.limit)
    text_output.write_text(stream.getvalue(), encoding="utf-8")
    manifest_output.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "config": config.as_dict(),
                "environment": environment_info(REPOSITORY_ROOT),
                "profile_file": output.name,
                "summary_file": text_output.name,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Binary profile: {output}")
    print(f"Text summary: {text_output}")
    print(f"Profile manifest: {manifest_output}")


if __name__ == "__main__":
    main()
