from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    profile: str
    dataset_size: int
    dimension: int
    query_count: int
    oracle_query_count: int
    warmup_queries: int
    k: int
    metric: str
    seed: int
    include_metadata: bool

    def __post_init__(self) -> None:
        positive_values = {
            "dataset_size": self.dataset_size,
            "dimension": self.dimension,
            "query_count": self.query_count,
            "oracle_query_count": self.oracle_query_count,
            "k": self.k,
        }
        for name, value in positive_values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if isinstance(self.warmup_queries, bool) or not isinstance(
            self.warmup_queries,
            int,
        ):
            raise TypeError("warmup_queries must be an integer")
        if self.warmup_queries < 0:
            raise ValueError("warmup_queries must be a non-negative integer")

        if self.k > self.dataset_size:
            raise ValueError("k cannot be greater than dataset_size")

        if self.oracle_query_count > self.query_count:
            raise ValueError("oracle_query_count cannot exceed query_count")

        if self.metric not in {"cosine", "dot", "l2"}:
            raise ValueError(f"Unsupported metric: {self.metric!r}")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

        if not isinstance(self.include_metadata, bool):
            raise TypeError("include_metadata must be a boolean")

    def with_overrides(self, **overrides: Any) -> BenchmarkConfig:
        values = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **values)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


PROFILES: dict[str, BenchmarkConfig] = {
    "quick": BenchmarkConfig(
        profile="quick",
        dataset_size=2_000,
        dimension=64,
        query_count=50,
        oracle_query_count=50,
        warmup_queries=10,
        k=10,
        metric="cosine",
        seed=42,
        include_metadata=True,
    ),
    "standard": BenchmarkConfig(
        profile="standard",
        dataset_size=1_000_000,
        dimension=128,
        query_count=1_000,
        oracle_query_count=25,
        warmup_queries=50,
        k=10,
        metric="cosine",
        seed=42,
        include_metadata=False,
    ),
}


def get_profile(name: str) -> BenchmarkConfig:
    try:
        return PROFILES[name]
    except KeyError:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown profile {name!r}; choose one of: {choices}") from None
