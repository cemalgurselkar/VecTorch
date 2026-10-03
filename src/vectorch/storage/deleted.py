"""
DeleteBitmap:
amacımız silinen internal_idleri track etmek ve yerlerine false koyarak id yer değiştirmesini önlemek.
np.zeros → başlangıç state'i False olmalı.
np.bool_ → baseline; gerçek bit-packed bitmap değil.
Geometric growth → allocation sayısını azaltıyor.
TODO benchmark: bool array vs bit-packed.
"""



import numpy as np
from numpy.typing import NDArray

class DeleteBitmap:
    def __init__(self, initial_capacity: int = 10024) -> None:
        if initial_capacity <= 0:
            raise ValueError("Initial capacity must be a positive integer.")
        
        self._capacity = initial_capacity
        # np.zeros çünkü içersiini false olarak init etmemiz gerekiyor. empty init etmiyor belleği doldurmamak için. 
        self._deleted: NDArray[np.bool_] = np.zeros(initial_capacity, dtype=np.bool_)
    
    def mark(self, internal_id: int) -> None:
        self._ensure_capacity(internal_id)
        self._deleted[internal_id] = True
    
    def unmark(self, internal_id: int) -> None:
        if internal_id < 0 or internal_id >= self._capacity:
            raise IndexError("Internal ID is out of bounds.")
        
        self._deleted[internal_id] = False
    
    def contains(self, internal_id: int) -> bool:
        if internal_id < 0 or internal_id >= self._capacity:
            return False
        
        return bool(self._deleted[internal_id])
    
    def clear(self) -> None:
        self._deleted.fill(False)
        
    def _ensure_capacity(self, internal_id: int) -> None:
        if internal_id >= self._capacity:
            raise ValueError("Internal ID exceeds the current capacity. Please increase the capacity first.")
        
        if internal_id >= self._capacity:
            return
        
        new_capacity = self._capacity
        
        while internal_id >= new_capacity:
            new_capacity *= 2
        
        new_deleted = np.zeros(new_capacity, dtype=np.bool_)
        
        new_deleted[:self._capacity] = self._deleted
        
        self._deleted = new_deleted
        self._capacity = new_capacity