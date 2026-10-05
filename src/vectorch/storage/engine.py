"""
    Persistence ve Concurrency kısımlarına sonra gelecez. partial write için.
"""
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .vectors import VectorStorage
from .ids import IDMap
from .metadata import MetadataStorage
from .deleted import DeleteBitmap


class StorageEngine:
    
    def __init__(self, dimension: int, initial_capacity: int = 1024) -> None:
        self._vectors = VectorStorage(dimension, initial_capacity)
        
        self._ids = IDMap()
        self._metadata = MetadataStorage()
        self._delete = DeleteBitmap(initial_capacity)
    
    def add(self, external_id, vector, metadata: dict[str, Any] | None = None) -> int:
        if self._ids.contains(external_id):
            raise ValueError(f"External ID {external_id} already exists.")
        
        internal_id = self._vectors.append(vector)
        mapped_id = self._ids.add(external_id)
        
        if mapped_id != internal_id:
            raise ValueError("Internal ID and external ID are not consistent.")
        
        if metadata is not None:
            self._metadata.set(internal_id, metadata)
            
        return internal_id
    
    def get_metadata(self, internal_id) -> dict[str, Any]:
        return self._metadata.get(internal_id)
    
    def get_vector(self, internal_id) -> dict[str, Any]:
        return self._vectors.get(internal_id)
    
    def get_external_id(self, internal_id) -> str | int:
        return self._ids.external(internal_id)
    
    def get_internal_id(self, external_id) -> int:
        return self._ids.internal(external_id)
    
    def delete(self, external_id) -> None:
        internal_id = self._ids.internal(external_id)
        self._delete.mark(internal_id)
    
    def is_deleted(self, internal_id) -> bool:
        return self._delete.contains(internal_id)
    
    def deleted_mask(self) -> NDArray[np.bool_]:
        return self._delete.view(len(self._vectors))
    
    def __len__(self) -> int:
        return len(self._vectors)
    
    @property
    def vectors(self) -> VectorStorage:
        return self._vectors