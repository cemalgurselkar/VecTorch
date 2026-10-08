from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import BenchmarkConfig


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    vectors: NDArray[np.float32]
    queries: NDArray[np.float32]
    metadata: list[dict[str, Any] | None]


def generate_dataset(config: BenchmarkConfig) -> BenchmarkDataset:
    seed_sequence = np.random.SeedSequence(config.seed)
    vector_seed, query_seed = seed_sequence.spawn(2)
    vector_rng = np.random.default_rng(vector_seed)
    query_rng = np.random.default_rng(query_seed)

    vectors = vector_rng.standard_normal(
        (config.dataset_size, config.dimension),
        dtype=np.float32,
    )
    queries = query_rng.standard_normal(
        (config.query_count, config.dimension),
        dtype=np.float32,
    )

    if config.include_metadata:
        metadata: list[dict[str, Any] | None] = [
            {
                "source": f"document-{index // 8}",
                "chunk": index,
                "group": index % 16,
            }
            for index in range(config.dataset_size)
        ]
    else:
        metadata = [None] * config.dataset_size

    return BenchmarkDataset(
        vectors=np.ascontiguousarray(vectors),
        queries=np.ascontiguousarray(queries),
        metadata=metadata,
    )
