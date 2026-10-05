import numpy as np
from numpy.typing import NDArray
from collections.abc import Callable

from .base import Index
from ..storage.vectors import VectorStorage
from ..kernels.base import DistanceKernel
from ..types import Metric


class FlatIndex(Index):

    def __init__(self,vectors: VectorStorage,metric: Metric,kernel: DistanceKernel,deleted_mask: Callable[[], NDArray[np.bool_]]) -> None:
        self._vectors = vectors
        self._metric = metric
        self._kernel = kernel
        self._deleted_mask = deleted_mask

    def search(self,query: NDArray[np.float32],k: int,) -> tuple[NDArray[np.int64], NDArray[np.float32]]:

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        size = len(self._vectors)

        if size == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
            )

        # Zero-copy view of stored vectors.
        vectors = self._vectors.view()

        scores = self._kernel.compute(query, vectors)

        deleted = self._deleted_mask()
        valid_ids = np.flatnonzero(~deleted)

        if len(valid_ids) == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
            )

        k = min(k, len(valid_ids))

        valid_scores = scores[valid_ids]

        if self._metric == Metric.L2:
            selected = np.argpartition(
                valid_scores,
                k - 1,
            )[:k]

            order = np.argsort(
                valid_scores[selected]
            )

        else:
            selected = np.argpartition(
                -valid_scores,
                k - 1,
            )[:k]

            order = np.argsort(
                -valid_scores[selected]
            )

        selected = selected[order]

        internal_ids = valid_ids[selected]

        return (
            internal_ids.astype(np.int64),
            scores[internal_ids].astype(np.float32),
        )