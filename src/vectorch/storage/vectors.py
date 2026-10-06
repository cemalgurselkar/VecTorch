import numpy as np
from numpy.typing import NDArray


class VectorStorage:
    
    def __init__(self, dimension: int, initial_capacity: int = 10024) -> None:
        
        if dimension <= 0:
            raise ValueError("Dimension must be a positive integer.")
        
        if initial_capacity <= 0:
            raise ValueError("Initial capacity must be a positive integer.")
        
        self._dimension = dimension
        self._capacity = initial_capacity
        self._size = 0
        
        self._data: NDArray[np.float32] = np.empty(
            (initial_capacity, dimension), dtype=np.float32
        ) # np.empty because we will fill it with data later, and we don't want to initialize it with zeros.

    def view(self) -> NDArray[np.float32]:
        return self._data[:self._size]

    def append(self, vector: NDArray[np.float32]) -> int:
        
        self._validate_vector(vector)
        
        if self._size >= self._capacity:
            self._grow()
        
        internal_id = self._size
        self._data[internal_id] = vector
        self._size += 1
        
        return internal_id
    
    def get(self, internal_id: int) -> NDArray[np.float32]:
        self._validate_id(internal_id)
        return self._data[internal_id]
    
    def get_many(self, internal_ids: NDArray[np.integer]) -> NDArray[np.float32]:
        # To collect more vectors at once
        if np.any(internal_ids >= self._size) or np.any(internal_ids < 0):
            raise IndexError("One or more internal IDs are out of bounds.")
        
        return self._data[internal_ids]
    
    def _reserve(self, capacity: int) -> None:
        if capacity <= self._capacity:
            return 
        
        new_data = np.empty((capacity, self._dimension), dtype=np.float32)
        new_data[:self._size] = self._data[:self._size] # Contiguous Numpy Buffer (memcpy)
        
        self._data = new_data
        self._capacity = capacity
    
    def _grow(self) -> None:
        self._reserve(self._capacity * 2)
    
    def _validate_vector(self, vector: NDArray[np.float32]) -> None:
        if vector.ndim != 1:
            raise ValueError("Vector must be a 1D array.")
        
        if vector.shape[0] != self._dimension:
            raise ValueError(f"Vector dimension must be {self._dimension}.")
        
        if vector.dtype != np.float32:
            raise ValueError("Vector dtype must be np.float32.")
    
    def _validate_id(self, internal_id: int) -> None:
        if internal_id < 0 or internal_id >= self._size:
            raise IndexError("Internal ID is out of bounds.")  
    
    def __len__(self) -> int:
        return self._size
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def capacity(self) -> int:
        return self._capacity