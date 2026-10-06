from pathlib import Path

from .collection import Collection
from .types import CollectionConfig


class CollectionManager:
    def __init__(self, root_path:str | Path):
        self._collection: dict[str, Collection] = {}
        self._root_path = Path(root_path)
    
    def create(self, config: CollectionConfig) -> Collection:
        if config.name in self._collection:
            raise ValueError(f"Collection with name '{config.name}' already exists.")
        
        collection = Collection(config=config)
        
        self._collection[config.name] = collection
        return collection
    
    def get(self, name: str) -> Collection:
        try:
            return self._collection[name]
        except KeyError:
            raise ValueError(f"Collection with name '{name}' does not exist.") from None
        
    
    def drop(self, name: str) -> None:
        collection = self.get(name)
        collection.close()
        
        del self._collection[name]
    
    def list(self) -> list[str]:
        return list(self._collection.keys())
    
    def close(self) -> None:
        for collection in self._collection.values():
            collection.close()
        
        self._collection.clear()