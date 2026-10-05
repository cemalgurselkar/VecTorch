from pathlib import Path
from .collection import Collection
from .manager import CollectionManager
from .types import CollectionConfig

class Vectorch:
    
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._manager = CollectionManager(self._path)
    
    def create_collection(self, name: str, dimension: int, metric: str ="cosine", index: str = "flat") -> Collection:
        
        config = CollectionConfig(name=name,
                                  dimension=dimension,
                                  metric=metric,
                                  index_type=index)
        
        return self._manager.create(config)
    
    def get_collection(self, name: str) -> Collection:
        return self._manager.get(name)
    
    def drop_collection(self, name: str) -> None:
        self._manager.drop(name)
    
    def list_collection(self) -> list[str]:
        return self._manager.list()
    
    def close(self) -> None:
        self._manager.close()