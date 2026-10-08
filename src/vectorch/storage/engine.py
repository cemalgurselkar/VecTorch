"""
    Persistence ve Concurrency kısımlarına sonra gelecez. partial write için.
"""
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..validation import validate_external_id, validate_metadata
from .deleted import DeleteBitmap
from .ids import IDMap
from .metadata import MetadataStorage
from .vectors import VectorStorage


@dataclass(frozen=True, slots=True)
class StorageSnapshot:
    vectors: NDArray[np.float32]
    ids: list[str | int]
    metadata: list[dict[str, Any] | None]
    deleted: NDArray[np.bool_]


class StorageEngine:
    
    def __init__(self, dimension: int, initial_capacity: int = 1024) -> None:
        self._vectors = VectorStorage(dimension, initial_capacity)
        
        # HAS A ilişkisi kurduk burada.
        self._ids = IDMap()
        self._metadata = MetadataStorage()
        self._deleted = DeleteBitmap(initial_capacity)
    
    def add(
        self,
        external_id: str | int,
        vector: NDArray[np.float32],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        external_id = validate_external_id(external_id)
        metadata = validate_metadata(metadata)

        if self._ids.contains(external_id):
            raise ValueError(f"External ID {external_id} already exists.")
        
        internal_id = self._vectors.append(vector)
        mapped_id = self._ids.add(external_id)
        
        if mapped_id != internal_id:
            raise ValueError("Internal ID and external ID are not consistent.")
        
        if metadata is not None:
            self._metadata.set(internal_id, metadata)
            
        return internal_id
    
    def count(self) -> int:
        return len(self._vectors) - self._deleted.count()
    
    def total_count(self) -> int:
        return len(self._vectors)
    
    def get_metadata(self, internal_id) -> dict[str, Any] | None:
        return self._metadata.get(internal_id)
    
    def get_vector(self, internal_id) -> NDArray[np.float32]:
        return self._vectors.get(internal_id)
    
    def get_external_id(self, internal_id) -> str | int:
        return self._ids.external(internal_id)
    
    def get_internal_id(self, external_id) -> int:
        return self._ids.internal(validate_external_id(external_id))

    def delete(self, external_id: str | int) -> bool:
        internal_id = self.get_internal_id(external_id)

        if self._deleted.contains(internal_id):
            return False

        self._deleted.mark(internal_id)
        return True
    
    def is_deleted(self, internal_id) -> bool:
        return self._deleted.contains(internal_id)
    
    def deleted_mask(self) -> NDArray[np.bool_]:
        return self._deleted.view(len(self._vectors))
    
    def snapshot(self) -> StorageSnapshot:
        size = len(self._vectors)

        return StorageSnapshot(
            vectors=self._vectors.view().copy(),
            ids=[self._ids.external(i) for i in range(size)],
            metadata=[self._metadata.get(i) for i in range(size)],
            deleted=self._deleted.view(size).copy(),
        )

    @classmethod
    def from_snapshot(cls, dimension: int, snapshot: StorageSnapshot) -> "StorageEngine":
        count = len(snapshot.vectors)
        if snapshot.vectors.shape != (count, dimension):
            raise ValueError("Invalid vector snapshot shape")

        if len(snapshot.metadata) != count:
            raise ValueError("Invalid metadata snapshot size")

        if len(snapshot.ids) != count:
            raise ValueError("Invalid ID snapshot size")

        if snapshot.deleted.shape != (count,):
            raise ValueError("Invalid deleted snapshot size")

        if snapshot.deleted.dtype != np.bool_:
            raise ValueError("Invalid deleted snapshot dtype")

        engine = cls(
            dimension=dimension,
            initial_capacity=max(1024, count),
        )

        for internal_id in range(count):
            restored_id = engine.add(
                external_id=snapshot.ids[internal_id],
                vector=snapshot.vectors[internal_id],
                metadata=snapshot.metadata[internal_id],
            )

            if restored_id != internal_id:
                raise RuntimeError(
                    "Failed to restore stable internal IDs"
                )

            if snapshot.deleted[internal_id]:
                engine._deleted.mark(internal_id)

        return engine

    def __len__(self) -> int:
        return len(self._vectors)
    
    @property
    def vectors(self) -> VectorStorage:
        return self._vectors
