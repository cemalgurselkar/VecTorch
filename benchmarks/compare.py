from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Metric:
    label: str
    path: tuple[str, ...]
    higher_is_better: bool


METRICS = (
    Metric("insert vec/s", ("results", "ingestion", "vectors_per_second"), True),
    Metric("search QPS", ("results", "search", "qps"), True),
    Metric("search p50 ms", ("results", "search", "p50_ms"), False),
    Metric("search p95 ms", ("results", "search", "p95_ms"), False),
    Metric("search p99 ms", ("results", "search", "p99_ms"), False),
    Metric("save seconds", ("results", "persistence", "save_seconds"), False),
    Metric("reopen seconds", ("results", "persistence", "reopen_seconds"), False),
    Metric("database bytes", ("results", "persistence", "database_size_bytes"), False),
    Metric(
        "peak search RSS bytes",
        ("results", "memory", "peak_rss_after_search_bytes"),
        False,
    ),
)


def load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"Unsupported benchmark report: {path}")
    return data


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for metric in METRICS:
        before = _nested_value(baseline, metric.path)
        after = _nested_value(candidate, metric.path)
        if before is None or after is None or before == 0:
            change_percent = None
            improvement_percent = None
        else:
            change_percent = ((after - before) / before) * 100
            improvement_percent = (
                change_percent if metric.higher_is_better else -change_percent
            )

        comparisons.append(
            {
                "metric": metric.label,
                "baseline": before,
                "candidate": after,
                "change_percent": change_percent,
                "improvement_percent": improvement_percent,
            }
        )
    return comparisons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two Vectorch benchmark JSON reports."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    baseline = load_report(args.baseline)
    candidate = load_report(args.candidate)
    comparisons = compare_reports(baseline, candidate)

    print(f"{'metric':<22} {'baseline':>14} {'candidate':>14} {'improvement':>13}")
    print("-" * 67)
    for item in comparisons:
        before = _format_value(item["baseline"])
        after = _format_value(item["candidate"])
        improvement = item["improvement_percent"]
        improvement_text = "n/a" if improvement is None else f"{improvement:+.2f}%"
        print(f"{item['metric']:<22} {before:>14} {after:>14} {improvement_text:>13}")


def _nested_value(data: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Report metric {'.'.join(path)} is not numeric")
    return float(value)


def _format_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4g}"


if __name__ == "__main__":
    main()
