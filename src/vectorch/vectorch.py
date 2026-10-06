from pathlib import Path

from .collection import Collection
from .manager import CollectionManager
from .types import CollectionConfig, IndexType, Metric


class Vectorch:
    
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._manager = CollectionManager(self._path)
    
    def create_collection(self, name: str, dimension: int, metric: str | Metric = Metric.COSINE, index: str | IndexType = IndexType.FLAT) -> Collection:
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
        return self._manager.get(name)
    
    def drop_collection(self, name: str) -> None:
        self._manager.drop(name)
    
    def list_collections(self) -> list[str]:
        return self._manager.list()
    
    def close(self) -> None:
        self._manager.close()