from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Metric(str, Enum):
    COSINE = "cosine"
    L2 = "l2"
    DOT = "dot"


class IndexType(str, Enum):
    FLAT = "flat"
    HNSW = "hnsw"
    IVF = "ivf"


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    name: str
    dimension: int
    metric: Metric = Metric.COSINE
    index_type: IndexType = IndexType.FLAT
    index_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Collection name cannot be empty.")

        if self.dimension <= 0:
            raise ValueError("Dimension must be greater than 0.")


@dataclass(frozen=True, slots=True)
class SearchResult:
    id: str | int
    score: float
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Record:
    id: str | int
    vector: Any
    metadata: dict[str, Any] | None = None