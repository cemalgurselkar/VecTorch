from pathlib import Path
from typing import Self

from .collection import Collection
from .manager import CollectionManager
from .types import CollectionConfig, IndexType, Metric


class Vectorch:
    
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._manager = CollectionManager(self._path)
        self._closed = False
    
    def __enter__(self) -> Self:
        self._ensure_open()
        return self
    
    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
    
    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Vectorch is closed")
        
    def create_collection(self, name: str, dimension: int, metric: str | Metric = Metric.COSINE, index: str | IndexType = IndexType.FLAT) -> Collection:
        self._ensure_open()
        try:
            normalized_metric = Metric(metric)
        except ValueError as exc:
            raise ValueError(f"Invalid metric '{metric}'") from exc
        
        try:
            normalized_index = IndexType(index)
        except ValueError as exc:
            raise ValueError(f"Invalid index '{index}'") from exc
        
        config = CollectionConfig(name=name,
                                  dimension=dimension,
                                  metric=normalized_metric,
                                  index_type=normalized_index)
        
        return self._manager.create(config)
    
    def get_collection(self, name: str) -> Collection:
        self._ensure_open()
        return self._manager.get(name)
    
    def drop_collection(self, name: str) -> None:
        self._ensure_open()
        self._manager.drop(name)
    
    def list_collections(self) -> list[str]:
        self._ensure_open()
        return self._manager.list()
    
    def close(self) -> None:
        if self._closed:
            return
        
        self._manager.close()
        self._closed = True