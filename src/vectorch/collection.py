from typing import Any

import numpy as np
from numpy.typing import NDArray

from .index.flat import FlatIndex
from .kernels.numpy import NumpyKernel
from .storage.engine import StorageEngine
from .types import CollectionConfig, IndexType, SearchResult
from .validation import validate_vector


class Collection:
    def __init__(self, config: CollectionConfig) -> None:
        self._config = config

        self._storage = StorageEngine(dimension=config.dimension)
        self._kernel = NumpyKernel(metric=config.metric)

        self._closed = False
        
        if config.index_type == IndexType.FLAT:
            self._index = FlatIndex(
                vectors=self._storage.vectors,
                metric=config.metric,
                kernel=self._kernel,
                deleted_mask=self._storage.deleted_mask
            )
        else:
            raise NotImplementedError(
                f"Index type '{config.index_type}' is not implemented."
            )

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def dimension(self) -> int:
        return self._config.dimension

    @property
    def config(self) -> CollectionConfig:
        return self._config

    def add(self, external_id: str | int, vector: NDArray[np.float32], metadata: dict[str, Any] | None = None) -> int:
        self._ensure_open()
        vector = validate_vector(vector, dimension=self._config.dimension, name="vector")
        return self._storage.add(
            external_id=external_id,
            vector=vector,
            metadata=metadata,
        )

    def search(self,query: NDArray[np.float32],k: int = 10) -> list[SearchResult]:
        
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError(f"k must be an integer, but is {type(k)}")
        
        if k <= 0:
            raise ValueError(f"k must be greater than 0, but is {k}")
        
        self._ensure_open()
        query = validate_vector(query, dimension=self._config.dimension, name="query")
        internal_ids, scores = self._index.search(query, k)

        results: list[SearchResult] = []

        for internal_id, score in zip(internal_ids, scores):
            iid = int(internal_id)

            # TODO(perf/correctness):
            # Deleted IDs should be filtered before top-k selection.
            if self._storage.is_deleted(iid):
                continue

            results.append(
                SearchResult(
                    id=self._storage.get_external_id(iid),
                    score=float(score),
                    metadata=self._storage.get_metadata(iid),
                )
            )

        return results

    def get(self,external_id: str | int) -> NDArray[np.float32]:
        
        self._ensure_open()
        internal_id = self._storage.get_internal_id(external_id)

        if self._storage.is_deleted(internal_id):
            raise KeyError(
                f"ID '{external_id}' has been deleted."
            )

        return self._storage.get_vector(internal_id).copy()

    def delete(self,external_id: str | int) -> None:
        self._ensure_open()
        self._storage.delete(external_id)

    def count(self) -> int:
        self._ensure_open()
        return self._storage.count()

    def total_count(self) -> int:
        self._ensure_open()
        return self._storage.total_count()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Collection is closed")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True