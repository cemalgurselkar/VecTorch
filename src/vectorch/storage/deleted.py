"""
DeleteBitmap: logical deletion + ID stability.

Amaç:
- Silinen internal ID'leri takip etmek.
- Fiziksel silme yapmayarak internal ID'lerin yer değiştirmesini önlemek.

Tasarım:
- np.zeros -> başlangıçta bütün kayıtlar "not deleted" (False).
- np.bool_ -> baseline; gerçek bit-packed bitmap değil.
- Geometric growth -> allocation sayısını azaltır.

TODO(perf):
- np.bool_ vs bit-packed bitmap benchmark.
"""

import numpy as np
from numpy.typing import NDArray


class DeleteBitmap:
    def __init__(self, initial_capacity: int = 1024) -> None:
        if initial_capacity <= 0:
            raise ValueError(
                "initial_capacity must be greater than 0"
            )

        self._capacity = initial_capacity

        self._deleted: NDArray[np.bool_] = np.zeros(
            initial_capacity,
            dtype=np.bool_,
        )

    def mark(self, internal_id: int) -> None:
        if internal_id < 0:
            raise ValueError("internal_id cannot be negative")

        self._ensure_capacity(internal_id)
        self._deleted[internal_id] = True

    def unmark(self, internal_id: int) -> None:
        if internal_id < 0 or internal_id >= self._capacity:
            raise IndexError("internal_id out of range")

        self._deleted[internal_id] = False

    def contains(self, internal_id: int) -> bool:
        if internal_id < 0 or internal_id >= self._capacity:
            return False

        return bool(self._deleted[internal_id])

    def clear(self) -> None:
        self._deleted.fill(False)

    def view(self, size: int) -> NDArray[np.bool_]:
        if size < 0:
            raise ValueError("size cannot be negative")

        if size > self._capacity:
            self._ensure_capacity(size - 1)

        return self._deleted[:size]

    def _ensure_capacity(self, internal_id: int) -> None:
        if internal_id < 0:
            raise ValueError("internal_id cannot be negative")

        if internal_id < self._capacity:
            return

        new_capacity = self._capacity

        while internal_id >= new_capacity:
            new_capacity *= 2

        new_deleted = np.zeros(
            new_capacity,
            dtype=np.bool_,
        )

        new_deleted[:self._capacity] = self._deleted

        self._deleted = new_deleted
        self._capacity = new_capacity