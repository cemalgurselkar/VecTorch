from pathlib import Path

from .collection import Collection
from .types import CollectionConfig


class CollectionManager:
    def __init__(self, root_path:str | Path):
        self._collection: dict[str, Collection] = {}
        self._root_path = Path(root_path)
        self._closed = False
    
    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Collection manager is closed")
    
    def create(self, config: CollectionConfig) -> Collection:
        self._ensure_open()
        if config.name in self._collection:
            raise ValueError(f"Collection with name '{config.name}' already exists.")
        
        collection = Collection(config=config)
        
        self._collection[config.name] = collection
        return collection
    
    def get(self, name: str) -> Collection:
        self._ensure_open()
        try:
            return self._collection[name]
        except KeyError:
            raise ValueError(f"Collection with name '{name}' does not exist.") from None
        
    
    def drop(self, name: str) -> None:
        self._ensure_open()
        collection = self._collection.pop(name)
        collection.close()
    
    def list(self) -> list[str]:
        self._ensure_open()
        return list(self._collection.keys())
    
    def close(self) -> None:
        if self._closed:
            return
        
        for collection in self._collection.values():
            collection.close()
        
        self._collection.clear()
        self._closed = True