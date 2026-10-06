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

        self._storage = StorageEngine(
            dimension=config.dimension
        )

        self._kernel = NumpyKernel(
            metric=config.metric
        )

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
        vector = validate_vector(vector, dimension=self._config.dimension, name="vector")
        return self._storage.add(
            external_id=external_id,
            vector=vector,
            metadata=metadata,
        )

    def search(self,query: NDArray[np.float32],k: int = 10) -> list[SearchResult]:
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

    def get(
        self,
        external_id: str | int,
    ) -> NDArray[np.float32]:
        internal_id = self._storage.get_internal_id(external_id)

        if self._storage.is_deleted(internal_id):
            raise KeyError(
                f"ID '{external_id}' has been deleted."
            )

        return self._storage.get_vector(internal_id)

    def delete(
        self,
        external_id: str | int,
    ) -> None:
        self._storage.delete(external_id)

    def count(self) -> int:
        return len(self._storage)

    def close(self) -> None:
        pass